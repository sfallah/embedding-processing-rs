use std::path::PathBuf;
use tracing::{info, error, Instrument, debug};
use std::sync::Arc;
use clap::Parser;
use futures::future::join_all;
use rayon::prelude::*;
use tokio::task;

use anyhow::Result;
use async_channel::Sender;
use embedding_common::models::document::Document;
use embedding_common::models::embedding::{Embedding, EmbeddingDataType, EmbeddingUser};
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_common::utils::helpers::get_db_dir;

use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils;

use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_database::dao::dao_impl::{put_document, put_embedding, put_embedding_user, put_model, put_split, put_summary};
use embedding_database::db::rocksdb_impl::RocksDB;

use embedding_cli::Cli;
use embedding_common::models::model::Model;
use uuid::Uuid;
use embedding_common::config::{AppConfig};
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::services::embeddings::EmbeddingsRequest;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    setup_tracing(args.log_level);
    debug!("{:?}", args);

    if let Err(e) = args.validate() {
        error!("Invalid arguments: {}", e);
        std::process::exit(1);
    }
    info!("Starting up");

    let app_config = AppConfig::from_file_async(args.config_file).await?;

    let db_config = app_config.clone().database_config;

    let db_path_binding = get_db_dir(Some(&db_config.rocksdb_dir))?;
    let db_path = db_path_binding.to_str().unwrap();
    embedding_common::utils::helpers::create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let user_id = args.user_id.map_or(Uuid::new_v4(), |user_id| {
        Uuid::parse_str(&user_id).unwrap()
    });

    run_process_docs(
        db,
        &args.file_path,
        user_id,
        app_config.clone(),
    )
        .instrument(tracing::info_span!("run_process_docs"))
        .await
        ?;

    info!("Shutting down");
    Ok(())
}

#[tracing::instrument]
async fn run_process_docs(
    db: Arc<RocksDB>,
    file_path: &std::path::Path,
    user_id: Uuid,
    app_config: AppConfig,
) -> Result<()> {

    let model_config = app_config.model_config;
    let splitter_config = app_config.splitter_config;

    let splits_index = HnswIndex::load_index("splits".to_string(), app_config.index_config.clone())?;
    let summaries_index = HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone())?;

    let text_files = embedding_cli::file_io::list_files(&file_path, &vec!["txt".to_string()]).await?;

    let (embed, shutdown, handles, model) = app_utils::init(&model_config.gguf_file, model_config.instances).await?;


    let ctx = init_ctx(splitter_config.max_tokens, splitter_config.merge_level, model.n_embd as usize, model.model_id).await;

    let futures = text_files.iter().map(|file_path| {
        process_doc(
            db.clone(),
            splits_index.clone(),
            summaries_index.clone(),
            ctx.clone(),
            embed.clone(),
            file_path,
            user_id,
            model.clone(),
        )
    });
    join_all(futures).await.into_iter().collect::<Result<()>>()?;
    save_index(splits_index, summaries_index).await?;
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

async fn process_doc(
    db: Arc<RocksDB>,
    splits_index: HnswIndex,
    summaries_index: HnswIndex,
    ctx: Arc<ProcessingContext>,
    embed: Arc<Sender<EmbeddingsRequest>>,
    file_path: &PathBuf,
    user_id: Uuid,
    model: Arc<Model>,
) -> Result<()> {
    let doc = tokio::fs::read_to_string(file_path).await?;

    let res = process_document(
        ctx.clone(),
        embed.clone(),
        file_path.to_string_lossy().to_string(),
        doc.into_bytes().to_vec(),
    )
        .await?;

    println!("{:?}", res);

    let (document, splits, summaries, embeddings, embedding_users) = task::spawn_blocking(move || {
        let document = res.to_model();

        let splits: Vec<_> = res.splits.par_iter().map(|split| split.to_model()).collect();
        let summaries: Vec<_> = res.summaries.par_iter().map(|summary| summary.to_model()).collect();

        let embeddings: Vec<Embedding> = res.splits.par_iter()
            .filter_map(|split_dto| split_dto.to_embedding_model())
            .chain(
                res.summaries.par_iter()
                    .filter_map(|summary_dto| summary_dto.to_embedding_model())
            )
            .collect();

        let embedding_users = embeddings.iter().map(|embedding| {
            let embed_id = embedding.embedding_id;
            EmbeddingUser::new(embed_id, user_id)
        }).collect::<Vec<_>>();

        (document, splits, summaries, embeddings, embedding_users)
    }).await?;
    save_models_to_db(&db, &document, &splits, &summaries, &embeddings, &embedding_users, model.clone()).await?;
    add_index(splits_index, summaries_index, &embeddings).await
}

async fn add_index(splits_index: HnswIndex,
                   summaries_index: HnswIndex,
                   embeddings: &[Embedding]) -> Result<()> {
    let index_futures = embeddings.iter().map(|embedding| {
        match embedding.embedding_type {
            EmbeddingDataType::Split => {
                HnswIndex::async_upsert(splits_index.clone(), &embedding.embedding, embedding.embedding_id)
            }
            EmbeddingDataType::Summary => {
                HnswIndex::async_upsert(summaries_index.clone(), &embedding.embedding, embedding.embedding_id)
            }
        }
    });
    join_all(index_futures).await;
    Ok(())
}

async fn save_index(splits_index: HnswIndex,
                    summaries_index: HnswIndex) -> Result<()> {
    join_all(vec![
        HnswIndex::async_save(splits_index.clone()),
        HnswIndex::async_save(summaries_index.clone()),
    ]).await.into_iter().collect::<Result<()>>()?;
    Ok(())
}


async fn save_models_to_db(
    db: &Arc<RocksDB>,
    document: &Document,
    splits: &[Split],
    summaries: &[Summary],
    embeddings: &[Embedding],
    embedding_users: &[EmbeddingUser],
    model: Arc<Model>,
) -> Result<()> {
    put_model(db, model.clone().as_ref()).await?;

    let embedding_futures = embeddings.iter().map(|embedding| put_embedding(db, embedding));
    join_all(embedding_futures).await.into_iter().collect::<Result<()>>()?;

    let embedding_user_futures = embedding_users.iter().map(|embedding_user| put_embedding_user(db, embedding_user));
    join_all(embedding_user_futures).await.into_iter().collect::<Result<()>>()?;

    let split_futures = splits.iter().map(|split| put_split(db, split));
    join_all(split_futures).await.into_iter().collect::<Result<()>>()?;

    let summary_futures = summaries.iter().map(|summary| put_summary(db, summary));
    join_all(summary_futures).await.into_iter().collect::<Result<()>>()?;

    put_document(db, document).await?;


    Ok(())
}




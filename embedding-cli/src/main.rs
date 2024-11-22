use clap::Parser;
use futures::future::join_all;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task;
use tracing::{debug, error, info, trace};

use anyhow::Result;
use async_channel::Sender;
use embedding_common::models::document::Document;
use embedding_common::models::embedding::{Embedding, EmbeddingDataType, EmbeddingUser};
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_common::utils::helpers::get_db_dir;

use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils;

use embedding_database::dao::dao_impl::{
    put_document, put_embedding, put_embedding_user, put_model, put_split, put_summary,
};
use embedding_database::db::rocksdb_impl::RocksDB;
use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};

use embedding_cli::Cli;
use embedding_common::config::AppConfig;
use embedding_common::models::model::Model;
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use uuid::Uuid;
use embedding_database::dao::embedding_dao::get_splits_by_embedding_ids;

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
    embedding_common::utils::helpers::create_directory(db_path)
        .expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let user_id = args
        .user_id
        .map_or(Uuid::new_v4(), |user_id| Uuid::parse_str(&user_id).unwrap());

    info!("User ID: {}", user_id);
    match args.command {
        embedding_cli::Commands::Query { query, top_k } => {
            info!("Querying");
            run_query(db, user_id, app_config.clone(), query, top_k).await?;
        }
        embedding_cli::Commands::Index { ref file_path } => {
            run_process_docs(db, file_path, user_id, app_config.clone()).await?;
        }
    }
    info!("Shutting down");
    Ok(())
}

#[tracing::instrument(skip(db, app_config))]
async fn run_query(
    db: Arc<RocksDB>,
    user_id: Uuid,
    app_config: AppConfig,
    query: String,
    top_k: usize,
) -> Result<()> {
    let model_config = app_config.model_config;

    let splits_index =
        HnswIndex::load_index("splits".to_string(), app_config.index_config.clone()).await?;
    let splits_index = Arc::new(splits_index);

    //let summaries_index = HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone())?;

    let (embed, shutdown, handles, model) =
        app_utils::init(&model_config.gguf_file, model_config.instances).await?;


    let query_embd =
        async_get_embeddings(embed.clone(), &vec![query.to_string()], model.n_embd as usize).await?;

    trace!("Query embeddings: {:?}", query_embd.len());

    let (embd_ids, _scores) =
        splits_index.query_filter(&db, &vec![user_id.clone()], &query_embd.to_vec(), top_k)
            .await?;
    info!("embd_ids: {:?}", embd_ids);

    let splits = get_splits_by_embedding_ids(&db, embd_ids).await?;
    for split in splits {
        info!("Split: {:?}", split);
    }

    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

//#[tracing::instrument(skip(db,app_config))]
async fn run_process_docs(
    db: Arc<RocksDB>,
    file_path: &std::path::Path,
    user_id: Uuid,
    app_config: AppConfig,
) -> Result<()> {
    let model_config = app_config.model_config;
    let splitter_config = app_config.splitter_config;

    let splits_index =
        HnswIndex::load_index("splits".to_string(), app_config.index_config.clone()).await?;
    let splits_index = Arc::new(splits_index);
    let summaries_index =
        HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone()).await?;
    let summaries_index = Arc::new(summaries_index);

    let text_files =
        embedding_cli::file_io::list_files(&file_path, &vec!["txt".to_string()]).await?;

    let (embed, shutdown, handles, model) =
        app_utils::init(&model_config.gguf_file, model_config.instances).await?;

    let ctx = init_ctx(
        splitter_config.max_tokens,
        splitter_config.merge_level,
        model.n_embd as usize,
        model.model_id,
    )
        .await;

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
    join_all(futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;
    save_index(splits_index, summaries_index).await?;
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

//#[tracing::instrument(skip(db, splits_index, summaries_index, ctx, embed))]
async fn process_doc(
    db: Arc<RocksDB>,
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
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

    let (document, splits, summaries, embeddings, embedding_users) =
        task::spawn_blocking(move || {
            let document = res.to_model();

            let splits: Vec<_> = res
                .splits
                .par_iter()
                .map(|split| split.to_model())
                .collect();
            let summaries: Vec<_> = res
                .summaries
                .par_iter()
                .map(|summary| summary.to_model())
                .collect();

            let embeddings: Vec<Embedding> = res
                .splits
                .par_iter()
                .filter_map(|split_dto| split_dto.to_embedding_model())
                .chain(
                    res.summaries
                        .par_iter()
                        .filter_map(|summary_dto| summary_dto.to_embedding_model()),
                )
                .collect();

            let embedding_users = embeddings
                .iter()
                .map(|embedding| {
                    let embed_id = embedding.embedding_id;
                    EmbeddingUser::new(embed_id, user_id)
                })
                .collect::<Vec<_>>();

            (document, splits, summaries, embeddings, embedding_users)
        })
            .await?;
    save_models_to_db(
        &db,
        &document,
        &splits,
        &summaries,
        &embeddings,
        &embedding_users,
        model.clone(),
    )
        .await?;
    add_index(splits_index, summaries_index, &embeddings).await
}

#[tracing::instrument(skip(splits_index, summaries_index, embeddings))]
async fn add_index(
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
    embeddings: &[Embedding],
) -> Result<()> {
    let index_futures = embeddings
        .iter()
        .map(|embedding| match embedding.embedding_type {
            EmbeddingDataType::Split => splits_index.upsert(
                &embedding.embedding,
                embedding.embedding_id,
            ),
            EmbeddingDataType::Summary => summaries_index.upsert(
                &embedding.embedding,
                embedding.embedding_id,
            ),
        });
    join_all(index_futures).await;
    Ok(())
}

#[tracing::instrument(skip(splits_index, summaries_index))]
async fn save_index(splits_index: Arc<HnswIndex>, summaries_index: Arc<HnswIndex>) -> Result<()> {
    join_all(vec![
        splits_index.save(),
        summaries_index.save(),
    ])
        .await
        .into_iter()
        .collect::<Result<()>>()?;
    Ok(())
}

//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
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

    let embedding_futures = embeddings
        .iter()
        .map(|embedding| put_embedding(db, embedding));
    join_all(embedding_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let embedding_user_futures = embedding_users
        .iter()
        .map(|embedding_user| put_embedding_user(db, embedding_user));

    join_all(embedding_user_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let split_futures = splits.iter().map(|split| put_split(db, split));
    join_all(split_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let summary_futures = summaries.iter().map(|summary| put_summary(db, summary));
    join_all(summary_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    put_document(db, document).await?;

    Ok(())
}

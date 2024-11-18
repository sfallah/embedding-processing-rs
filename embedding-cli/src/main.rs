use tracing::{info, error, Instrument};
use std::sync::Arc;
use anyhow::anyhow;
use clap::Parser;
use futures::future::join_all;
use rayon::prelude::*;
use tokio::task;

use anyhow::Result;
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

use embedding_cli::Args;
use embedding_common::models::model::Model;
use uuid::Uuid;
use embedding_index::hnsw_index::HnswIndex;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        error!("Error: {}", e);
        std::process::exit(1);
    }
    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let db_path_binding = get_db_dir(Some(&args.db_path))?;
    let db_path = db_path_binding.to_str().unwrap();
    embedding_common::utils::helpers::create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let user_id = args.user_id.map_or(Uuid::new_v4(), |user_id| {
        Uuid::parse_str(&user_id).unwrap()
    });

    run_doc_processing(
        db,
        &args.model_path,
        args.max_tokens,
        args.merge_level,
        args.np,
        args.n_embd,
        &args.file_path,
        user_id,
        args.index_dir,
    )
        .instrument(tracing::info_span!("run_doc_processing"))
        .await
        ?;

    info!("Shutting down");
    Ok(())
}

#[tracing::instrument]
async fn run_doc_processing(
    db: Arc<RocksDB>,
    model_path: &str,
    max_tokens: usize,
    merge_level: Option<usize>,
    np: usize,
    n_embd: usize,
    file_path: &std::path::Path,
    user_id: Uuid,
    index_dir: String,
) -> Result<()> {
    let (embed, shutdown, handles, model) = app_utils::init(&model_path, np).await?;

    let ctx = init_ctx(max_tokens, merge_level, n_embd, model.model_id).await;

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
    create_index(index_dir, n_embd as i32, &embeddings).await?;
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

async fn create_index(index_dir: String, n_embd: i32, embeddings: &[Embedding]) -> Result<()> {
    let splits_index = HnswIndex::new(n_embd as usize)?;
    let summaries_index = HnswIndex::new(n_embd as usize)?;
    embeddings.iter().for_each(|embedding| {
        match embedding.embedding_type {
            EmbeddingDataType::Split => {
                splits_index.add(&embedding.embedding, embedding.embedding_id).unwrap();
            }
            EmbeddingDataType::Summary => {
                summaries_index.add(&embedding.embedding, embedding.embedding_id).unwrap();
            }
        }
    });
    embedding_common::utils::helpers::create_directory(index_dir.as_str()).map_err(|e| anyhow!("Failed to create index directory: {:?}", e))?;
    let splits_index_file = format!("{}/splits_index.usearch", index_dir);
    splits_index.save(splits_index_file.as_str()).map_err(|e| anyhow!("Failed to save index: {:?}", e))?;
    let summaries_index_file = format!("{}/summaries_index.usearch", index_dir);
    summaries_index.save(summaries_index_file.as_str()).map_err(|e| anyhow!("Failed to save index: {:?}", e))?;
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




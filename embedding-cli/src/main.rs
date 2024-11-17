use tracing::{info, error, Instrument};
use std::sync::Arc;
use anyhow::anyhow;
use clap::Parser;
use futures::future::{join_all, try_join_all, FutureExt};
use rayon::prelude::*;
use tokio::task;

use embedding_common::models::document::Document;
use embedding_common::models::embedding::{Embedding, EmbeddingDataType, EmbeddingUser};
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_common::utils::helpers::get_db_dir;

use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::async_get_embeddings;
use embedding_processing::utils::app_utils;

use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_database::dao::dao_impl::{put_document, put_embedding, put_embedding_user, put_split, put_summary};
use embedding_database::db::rocksdb_impl::RocksDB;

use embedding_cli::Args;
use embedding_common::models::model::Model;
use uuid::Uuid;
use embedding_index::hnsw_index::HnswIndex;
use rayon::prelude::*;

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        error!("Error: {}", e);
        std::process::exit(1);
    }
    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let db_path_binding = get_db_dir(Some(&args.db_path.unwrap_or_else(|| "rocks_db_dir".to_string())))
        .unwrap();
    let db_path = db_path_binding.to_str().unwrap();
    embedding_common::utils::helpers::create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await.unwrap();
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
        user_id
    )
        .instrument(tracing::info_span!("run_doc_processing"))
        .await
        .unwrap();

    info!("Shutting down");
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
) -> anyhow::Result<()> {
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

    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;

    Ok(())
}

async fn create_index(_path:String,n_embd:i32, embeddings: &[Embedding]) -> anyhow::Result<()> {
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
    splits_index.save("output/splits_index.usearch").map_err(|e| anyhow!("Failed to save index: {:?}", e))?;
    summaries_index.save("output/summaries_index.usearch").map_err(|e| anyhow!("Failed to save index: {:?}", e))?;
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
) -> anyhow::Result<()> {

    let futures = embeddings.iter()
        .map(|embedding| put_embedding(db, embedding).boxed())
        .chain(embedding_users.iter().map(|user| put_embedding_user(db, user).boxed()))
        .chain(splits.iter().map(|split| put_split(db, split).boxed()))
        .chain(summaries.iter().map(|summary| put_summary(db, summary).boxed()))
        .chain(std::iter::once(put_document(db, document).boxed()));

    try_join_all(futures).await?;

    Ok(())
}




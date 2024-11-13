use tracing::{info, error, Instrument};
use std::sync::Arc;
use clap::Parser;
use futures::future::join_all;
use rayon::prelude::*;
use tokio::task;

use embedding_common::models::document::Document;
use embedding_common::models::embedding::Embedding;
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_common::utils::helpers::get_db_dir;

use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::async_get_embeddings;
use embedding_processing::utils::app_utils;

use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use embedding_database::dao::dao_impl::{put_document, put_embedding, put_split, put_summary};
use embedding_database::db::rocksdb_impl::RocksDB;

use embedding_cli::Args;

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        error!("Error: {}", e);
        std::process::exit(1);
    }
    setup_tracing(args.log_level.to_tracing_level());

    info!("Starting up");

    let db_path_binding = get_db_dir(Some(&args.db_path.unwrap_or_else(|| "rocksdb_dir".to_string())))
        .unwrap();
    let db_path = db_path_binding.to_str().unwrap();
    embedding_common::utils::helpers::create_directory(db_path).expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await.unwrap();
    let db = Arc::new(rocksdb);

    run_doc_processing(
        db,
        &args.model_path,
        args.max_tokens,
        args.merge_level,
        args.np,
        args.n_embd,
        &args.file_path
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
) -> anyhow::Result<()> {
    let (embed, shutdown, handles) = app_utils::init(&model_path, np).await?;

    let ctx = init_ctx(max_tokens, merge_level, n_embd).await;

    let doc = tokio::fs::read_to_string(file_path).await?;

    let res = process_document(
        ctx.clone(),
        embed.clone(),
        file_path.to_string_lossy().to_string(),
        doc.into_bytes().to_vec(),
    )
        .await?;

    println!("{:?}", res);

    let (document, splits, summaries, embeddings) = task::spawn_blocking(move || {
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

        (document, splits, summaries, embeddings)
    }).await?;

    save_models_to_db(&db, &document, &splits, &summaries, &embeddings).await?;

    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;

    Ok(())
}

async fn _run_embeddings(model_path: &str, np: usize) -> anyhow::Result<()> {
    let (embed, shutdown, handles) =
        app_utils::init(&model_path, np).await?;
    let text = "This is a test".to_string();
    let embeddings = async_get_embeddings(embed.clone(), &vec![text.clone()], 512).await?;
    println!("{:?}", embeddings);
    let _embeddings = async_get_embeddings(embed.clone(), &vec![text], 512).await?;
    println!("got second embeddings");
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

async fn save_models_to_db(
    db: &Arc<RocksDB>,
    document: &Document,
    splits: &[Split],
    summaries: &[Summary],
    embeddings: &[Embedding],
) -> anyhow::Result<()> {
    let embedding_futures = embeddings.iter().map(|embedding| put_embedding(db, embedding));
    join_all(embedding_futures).await.into_iter().collect::<Result<(), _>>()?;

    let split_futures = splits.iter().map(|split| put_split(db, split));
    join_all(split_futures).await.into_iter().collect::<Result<(), _>>()?;

    let summary_futures = summaries.iter().map(|summary| put_summary(db, summary));
    join_all(summary_futures).await.into_iter().collect::<Result<(), _>>()?;

    put_document(db, document).await?;

    Ok(())
}




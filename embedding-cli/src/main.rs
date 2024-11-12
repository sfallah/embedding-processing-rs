use std::sync::Arc;
use clap::Parser;
use futures::future::join_all;
use embedding_processing::processing::documents::process_document;
use embedding_processing::services::embeddings::async_get_embeddings;
use embedding_processing::utils::app_utils;

use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};
use tracing::Instrument;
use tracing::info;
use embedding_cli::Args;
use embedding_common::utils::helpers::get_db_dir;
use embedding_database::dao::dao_impl::{put_document, put_embedding, put_split, put_summary};
use embedding_database::db::rocksdb_impl::RocksDB;
use embedding_database::models::document::Document;
use embedding_database::models::embedding::{Embedding, EmbeddingDataType};
use embedding_database::models::split::Split;
use embedding_database::models::summary::Summary;
use embedding_processing::dtos::document_dto::DocumentDto;

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        eprintln!("Error: {}", e);
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

    run_doc_processing(db, &args.model_path, args.max_tokens, args.np, &args.file_path)
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
    np: usize,
    file_path: &std::path::Path,
) -> anyhow::Result<()> {
    let (embed, shutdown, handles) = app_utils::init(&model_path, np).await?;

    let ctx = init_ctx().await;

    let doc = tokio::fs::read_to_string("embedding-processing/tests/test_data/superlinear.txt").await?;

    let res = process_document(
        ctx.clone(),
        embed.clone(),
        file_path.to_string_lossy().to_string(),
        doc.into_bytes().to_vec(),
    )
        .await?;

    println!("{:?}", res);

    save_document(db.clone(), &res).await?;

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

pub async fn save_document(db: Arc<RocksDB>, doc_dto: &DocumentDto) -> anyhow::Result<()> {
    let mut splits = Vec::new();
    let mut summaries = Vec::new();
    let mut embeddings = Vec::new();
    let mut split_ids = Vec::new();
    let mut document_summary_ids = Vec::new();

    for split_dto in &doc_dto.splits {
        let split_embedding_id = if let Some(embedding_dto) = &split_dto.embedding {
            let embedding = Embedding {
                embedding_id: embedding_dto.embedding_id,
                data_id: split_dto.split_id,
                embedding_type: EmbeddingDataType::Split,
                embedding: embedding_dto.embedding.clone(),
            };
            embeddings.push(embedding);
            Some(embedding_dto.embedding_id)
        } else {
            None
        };

        let mut split_summary_ids = Vec::new();

        for summary_dto in &split_dto.summaries {
            let summary_embedding_id = if let Some(embedding_dto) = &summary_dto.embedding {
                let embedding = Embedding {
                    embedding_id: embedding_dto.embedding_id,
                    data_id: summary_dto.summary_id,
                    embedding_type: EmbeddingDataType::Summary,
                    embedding: embedding_dto.embedding.clone(),
                };
                embeddings.push(embedding);
                embedding_dto.embedding_id
            } else {
                0
            };

            let summary = Summary {
                summary_id: summary_dto.summary_id,
                document_id: summary_dto.document_id,
                split_id: summary_dto.split_id,
                split_sequence_id: summary_dto.split_sequence_id,
                embedding_id: summary_embedding_id,
                text_content: summary_dto.text_content.clone(),
                token_len: summary_dto.token_len,
                centrality: summary_dto.centrality,
            };
            summaries.push(summary);
            split_summary_ids.push(summary_dto.summary_id);
        }

        let split = Split {
            split_id: split_dto.split_id,
            sequence_id: split_dto.sequence_id,
            doc_id: split_dto.doc_id,
            embedding_id: split_embedding_id.unwrap_or(0),
            text_content: split_dto.text_content.clone(),
            token_len: split_dto.token_len,
            summary_ids: if split_summary_ids.is_empty() {
                None
            } else {
                Some(split_summary_ids)
            },
        };
        splits.push(split);
        split_ids.push(split_dto.split_id);
    }

    for summary_dto in &doc_dto.summaries {
        let summary_embedding_id = if let Some(embedding_dto) = &summary_dto.embedding {
            let embedding = Embedding {
                embedding_id: embedding_dto.embedding_id,
                data_id: summary_dto.summary_id,
                embedding_type: EmbeddingDataType::Summary,
                embedding: embedding_dto.embedding.clone(),
            };
            embeddings.push(embedding);
            embedding_dto.embedding_id
        } else {
            0
        };

        let summary = Summary {
            summary_id: summary_dto.summary_id,
            document_id: summary_dto.document_id,
            split_id: summary_dto.split_id,
            split_sequence_id: summary_dto.split_sequence_id,
            embedding_id: summary_embedding_id,
            text_content: summary_dto.text_content.clone(),
            token_len: summary_dto.token_len,
            centrality: summary_dto.centrality,
        };
        summaries.push(summary);
        document_summary_ids.push(summary_dto.summary_id);
    }

    let document = Document {
        document_id: doc_dto.document_id,
        document_url: doc_dto.document_url.clone(),
        split_ids,
        summary_ids: if document_summary_ids.is_empty() {
            None
        } else {
            Some(document_summary_ids)
        }
    };

    save_models_to_db(&db, &document, &splits, &summaries, &embeddings).await?;

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




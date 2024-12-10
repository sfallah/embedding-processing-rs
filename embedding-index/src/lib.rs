use crate::hnsw_index::HnswIndex;
use embedding_common::prelude::{DocumentDto, EmbeddingDataType};
use embedding_database::prelude::{get_all_embeddings, RocksDB};
use futures::future::join_all;
use std::sync::Arc;
use tracing::{error, info};

pub mod hnsw_index;
pub mod index_record;
pub mod utils;

use index_record::IndexRecord;

#[tracing::instrument(skip(splits_index, summaries_index, doc_dto))]
pub async fn add_to_indices(
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
    doc_dto: &DocumentDto,
) -> anyhow::Result<()> {
    let mut split_entries = Vec::new();
    let mut summary_entries = Vec::new();
    for split_dto in doc_dto.splits.iter() {
        match split_dto.embedding.as_ref() {
            Some(embd) => {
                split_entries.push(IndexRecord::new(embd.embedding_id, embd.embedding.clone()));
            }
            None => {
                error!("Split embedding is missing");
            }
        }
        for summary_dto in split_dto.summaries.iter() {
            match summary_dto.embedding.as_ref() {
                Some(embd) => {
                    summary_entries
                        .push(IndexRecord::new(embd.embedding_id, embd.embedding.clone()));
                }
                None => {
                    error!("Summary embedding is missing");
                }
            }
        }
    }

    splits_index
        .upsert_batch_records(Arc::new(split_entries))
        .await?;
    summaries_index
        .upsert_batch_records(Arc::new(summary_entries))
        .await?;
    Ok(())
}

#[tracing::instrument(skip(splits_index, summaries_index))]
pub async fn save_index(
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
) -> anyhow::Result<()> {
    join_all(vec![splits_index.save(), summaries_index.save()])
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;
    Ok(())
}

pub async fn initialize_index_from_db(
    db: &Arc<RocksDB>,
    split_index: &HnswIndex,
    summary_index: &HnswIndex,
) -> anyhow::Result<()> {
    let embedding_data = match get_all_embeddings(db).await {
        Ok(embedding_data) => embedding_data,
        Err(e) => {
            let error_message = format!("Failed to get all embedding data: {}", e);
            error!("{}", error_message);
            return Err(anyhow::anyhow!(error_message));
        }
    };

    if embedding_data.is_empty() {
        info!("No embeddings found in the database");
        return Ok(());
    }

    let mut split_no = 0;
    let mut summary_no = 0;

    let mut split_labels = Vec::new();
    let mut summary_labels = Vec::new();

    let mut split_embeddings = Vec::new();
    let mut summary_embeddings = Vec::new();

    for embedding in embedding_data {
        if embedding.embedding_type == EmbeddingDataType::Split {
            split_labels.push(embedding.embedding_id);
            split_embeddings.push(embedding.embedding);
        } else {
            summary_labels.push(embedding.embedding_id);
            summary_embeddings.push(embedding.embedding);
        }
    }
    if let Err(e) = split_index
        .upsert_batch(&split_embeddings, &split_labels)
        .await
    {
        error!("Failed to add split embeddings: {}", e);
    } else {
        split_no = split_labels.len();
    }

    if let Err(e) = summary_index
        .upsert_batch(&summary_embeddings, &summary_labels)
        .await
    {
        error!("Failed to add summary embeddings: {}", e);
    } else {
        summary_no = summary_labels.len();
    }
    info!(
        "Initialized index from db, splits: {}, summaries: {}",
        split_no, summary_no
    );
    Ok(())
}

use std::sync::Arc;
use futures::future::join_all;
use tracing::{error, info};
use embedding_common::prelude::{DocumentDto, EmbeddingDataType};
use embedding_database::prelude::{get_all_embeddings, RocksDB};
use crate::hnsw_index::HnswIndex;

pub mod hnsw_index;
pub mod utils;


#[tracing::instrument(skip(splits_index, summaries_index, doc_dto))]
pub async fn add_to_indices(
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
    doc_dto: &DocumentDto,
) -> anyhow::Result<()> {
    let splits_futures = doc_dto.splits.iter().flat_map(|split| {
        split
            .embedding
            .as_ref()
            .map(|embd| splits_index.upsert(&embd.embedding, embd.embedding_id))
    });
    join_all(splits_futures).await;

    let summaries_futures = doc_dto.summaries.iter().flat_map(|summary| {
        summary
            .embedding
            .as_ref()
            .map(|embd| summaries_index.upsert(&embd.embedding, embd.embedding_id))
    });
    join_all(summaries_futures).await;
    Ok(())
}

#[tracing::instrument(skip(splits_index, summaries_index))]
pub async fn save_index(splits_index: Arc<HnswIndex>, summaries_index: Arc<HnswIndex>) -> anyhow::Result<()> {
    join_all(vec![splits_index.save(), summaries_index.save()])
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;
    Ok(())
}

pub async fn initialize_index_from_db(db: &Arc<RocksDB>, split_index: &HnswIndex, summary_index: &HnswIndex) {
    let embedding_data = match get_all_embeddings(db).await {
        Ok(embedding_data) => embedding_data,
        Err(e) => {
            error!("Failed to get all embedding data: {}", e);
            return;
        }
    };

    if embedding_data.is_empty() {
        info!("No embedding data found, returning early.");
        return;
    }

    for embedding in embedding_data {
        if embedding.embedding_type == EmbeddingDataType::Split {
            split_index.upsert(&embedding.embedding, embedding.embedding_id).await.expect("Failed to add split embedding");
        } else {
            summary_index.upsert(&embedding.embedding, embedding.embedding_id).await.expect("Failed to add summary embedding");
        }
    }
    Ok::<(), anyhow::Error>(()).expect("TODO: panic message");
}
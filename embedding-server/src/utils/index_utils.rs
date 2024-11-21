use std::sync::Arc;
use tracing::{error, info};
use embedding_common::models::embedding::EmbeddingDataType;
use embedding_database::dao::embedding_dao::get_all_embeddings;
use embedding_database::db::rocksdb_impl::RocksDB;
use embedding_index::hnsw_index::HnswIndex;

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
            split_index.add(&embedding.embedding, embedding.data_id).expect("Failed to add split embedding");
        } else {
            summary_index.add(&embedding.embedding, embedding.data_id).expect("Failed to add summary embedding");
        }
    }
    Ok::<(), anyhow::Error>(()).expect("TODO: panic message");
}
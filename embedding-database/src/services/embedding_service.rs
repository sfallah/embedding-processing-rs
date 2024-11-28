use std::sync::Arc;
use embedding_common::prelude::EmbeddingDto;
use crate::dao::embedding_dao::get_embeddings;
use crate::prelude::{get_embedding, RocksDB};

pub async fn get_all_embeddings_full(db: &Arc<RocksDB>, embedding_ids: &[u64]) -> anyhow::Result<Vec<EmbeddingDto>> {
    let embeddings = get_embeddings(&db, embedding_ids).await?;
    let embeddings_dtos:Vec<_> = embeddings.iter().map(|embedding| embedding.to_dto()).collect();
    Ok(embeddings_dtos)
}

pub async fn get_embedding_full(db: &Arc<RocksDB>, embedding_id: &u64) -> anyhow::Result<Option<EmbeddingDto>> {
    let embedding = get_embedding(&db, embedding_id).await?;
    Ok(embedding.map(|e| e.to_dto()))
}
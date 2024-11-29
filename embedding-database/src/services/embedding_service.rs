use std::sync::Arc;
use indexmap::IndexMap;
use embedding_common::prelude::EmbeddingDto;
use crate::dao::embedding_dao::get_embeddings;
use crate::prelude::{get_embedding, RocksDB};

pub async fn get_embeddings_map(db: &Arc<RocksDB>, embedding_ids: &[u64]) -> anyhow::Result<IndexMap<u64, EmbeddingDto>> {
    let embeddings = get_embeddings(&db, embedding_ids).await?;
    let embeddings_map = IndexMap::from_iter( embeddings.into_iter().map(|e| (e.embedding_id, e.to_dto())));
    Ok(embeddings_map)
}



pub async fn get_embedding_full(db: &Arc<RocksDB>, embedding_id: &u64) -> anyhow::Result<Option<EmbeddingDto>> {
    let embedding = get_embedding(&db, embedding_id).await?;
    Ok(embedding.map(|e| e.to_dto()))
}
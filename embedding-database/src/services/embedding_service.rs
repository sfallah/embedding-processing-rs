use crate::dao::embedding_dao::{get_embedding, get_embeddings};
use crate::db::column_families::ColumnFamilyType;
use crate::db::db_record::DbRecordValue;
use crate::db::rocksdb_impl::RocksDB;
use embedding_common::prelude::{Embedding, EmbeddingDto, Serde};
use indexmap::IndexMap;
use std::sync::Arc;

pub fn get_embeddings_map(
    db: &Arc<RocksDB>,
    embedding_ids: &[u64],
) -> anyhow::Result<IndexMap<u64, EmbeddingDto>> {
    let embeddings = get_embeddings(&db, embedding_ids)?;
    let embeddings_map =
        IndexMap::from_iter(embeddings.into_iter().map(|e| (e.embedding_id, e.to_dto())));
    Ok(embeddings_map)
}

pub fn get_embedding_full(
    db: &Arc<RocksDB>,
    embedding_id: &u64,
) -> anyhow::Result<Option<EmbeddingDto>> {
    let embedding = get_embedding(&db, embedding_id)?;
    Ok(embedding.map(|e| e.to_dto()))
}

pub fn to_embedding_record(embedding: &Embedding) -> anyhow::Result<DbRecordValue> {
    Ok(DbRecordValue::new(
        ColumnFamilyType::Embeddings,
        embedding.embedding_id,
        embedding.pack()?,
    ))
}

pub fn get_all_embeddings(db: &Arc<RocksDB>) -> anyhow::Result<Vec<Embedding>> {
    crate::dao::embedding_dao::get_all_embeddings(&db)
}

use crate::db::column_families::ColumnFamilyType;
use crate::db::db_record::DbRecordValue;
use crate::db::rocksdb_impl::RocksDB;
use embedding_common::prelude::{EmbeddingUser, Serde};
use std::sync::Arc;
use uuid::Uuid;

pub fn has_embedding_user(
    db: &Arc<RocksDB>,
    embed_id: u64,
    user_uuids: Vec<Uuid>,
) -> anyhow::Result<bool> {
    crate::dao::embedding_user_dao::has_embedding_user(db, embed_id, user_uuids)
}

pub async fn to_embedding_user_record(
    embedding_user: &EmbeddingUser,
) -> anyhow::Result<DbRecordValue> {
    Ok(DbRecordValue::new(
        ColumnFamilyType::EmbeddingUsers,
        embedding_user.embed_id,
        embedding_user.pack()?,
    ))
}

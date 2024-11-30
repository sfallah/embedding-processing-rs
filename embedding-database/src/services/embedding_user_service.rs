use crate::db::rocksdb_impl::RocksDB;
use std::sync::Arc;
use uuid::Uuid;

pub fn has_embedding_user(
    db: &Arc<RocksDB>,
    embed_id: u64,
    user_uuids: Vec<Uuid>,
) -> anyhow::Result<bool> {
    crate::dao::embedding_user_dao::has_embedding_user(db, embed_id, user_uuids)
}

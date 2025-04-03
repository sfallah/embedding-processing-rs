use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;

pub fn get_embedding_filter_info(
    db: &Arc<RocksDB>,
    embed_id: u64,
) -> anyhow::Result<Option<EmbeddingFilterInfo>> {
    match db.get_sync(ColumnFamilyType::EmbeddingFilterInfo, &embed_id)? {
        Some(data) => {
            let embedding_user = EmbeddingFilterInfo::unpack(&data)
                .map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            Ok(Some(embedding_user))
        }
        None => Ok(None),
    }
}

pub fn include_in_search(
    db: &Arc<RocksDB>,
    embed_id: u64,
    user_uuids: &Vec<Uuid>,
    doc_ids: &Vec<u64>,
) -> bool {
    if user_uuids.is_empty() {
        return false;
    }
    let res = match get_embedding_filter_info(&db.clone(), embed_id) {
        Ok(Some(embd_info)) => {
            let in_user_uuids = user_uuids.contains(&embd_info.user_uuid);
            if doc_ids.is_empty() {
                in_user_uuids
            } else {
                doc_ids.contains(&embd_info.doc_id) && in_user_uuids
            }
        }
        _ => false,
    };
    res
}

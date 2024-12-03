use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;

#[tracing::instrument(skip(db))]
pub async fn put_embedding_user(
    db: &Arc<RocksDB>,
    embedding_user: &EmbeddingUser,
) -> anyhow::Result<()> {
    let embed_id = &embedding_user.embed_id;
    let data = embedding_user
        .pack()
        .map_err(|e| anyhow!("Failed to pack embedding: {}", e))?;
    db.put(ColumnFamilyType::EmbeddingUsers, embed_id, &data)
        .await?;
    Ok(())
}

pub fn get_embedding_user(
    db: &Arc<RocksDB>,
    embed_id: u64,
) -> anyhow::Result<Option<EmbeddingUser>> {
    match db.get_sync(ColumnFamilyType::EmbeddingUsers, &embed_id)? {
        Some(data) => {
            let embedding_user = EmbeddingUser::unpack(&data)
                .map_err(|e| anyhow!("Failed to unpack embedding: {}", e))?;
            Ok(Some(embedding_user))
        }
        None => Ok(None),
    }
}

pub fn has_embedding_user(
    db: &Arc<RocksDB>,
    embed_id: u64,
    user_uuids: Vec<Uuid>,
) -> anyhow::Result<bool> {
    let res = match get_embedding_user(&db.clone(), embed_id) {
        Ok(opt) => {
            //info!("Embedding user: {:?}", opt);
            opt.map(|eu| user_uuids.contains(&eu.user_uuid))
                .unwrap_or(false)
        }
        Err(_) => false,
    };
    Ok(res)
}

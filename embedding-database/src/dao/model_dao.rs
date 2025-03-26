use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;

#[allow(unused)]
pub fn put_model(db: &Arc<RocksDB>, model: &Model) -> anyhow::Result<()> {
    let model_id = &model.model_id;
    let data = model
        .pack()
        .map_err(|e| anyhow!("Failed to pack model: {}", e))?;
    db.put(ColumnFamilyType::Models, model_id, &data)?;
    Ok(())
}

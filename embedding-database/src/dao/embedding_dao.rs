use std::sync::Arc;
use embedding_common::models::embedding::Embedding;
use embedding_common::models::split::Split;
use embedding_common::Serde;
use crate::dao::dao_impl::put_embedding;
use crate::db::column_families::ColumnFamilyType;
use crate::db::rocksdb_impl::RocksDB;

pub async fn put_embeddings(
    db: &Arc<RocksDB>,
    embeddings: Vec<Embedding>,
) -> anyhow::Result<()> {
    let futures = embeddings.iter().map(|embedding| put_embedding(db, embedding));
    futures::future::join_all(futures).await.into_iter().collect::<anyhow::Result<()>>()?;
    Ok(())
}

pub async fn get_splits_by_embedding_ids(
    db: Arc<RocksDB>,
    embedding_ids: Vec<u64>,
) -> anyhow::Result<Vec<Split>> {
    let embeddings = db.multi_get(ColumnFamilyType::Embeddings, &embedding_ids)
        .await?;
    let mut split_ids = Vec::new();
    for embed_opt in embeddings.iter() {
        match embed_opt {
            Some(embedding) => {
                let embedding: Embedding = Embedding::unpack(embedding)?;
                split_ids.push(embedding.data_id);
            }
            None => (),
        }
    }
    let splits = db.multi_get(ColumnFamilyType::Splits, &split_ids)
        .await?
        .into_iter()
        .filter_map(|opt| opt)
        .map(|bytes| Split::unpack(&bytes).unwrap())
        .collect();
    Ok(splits)
}
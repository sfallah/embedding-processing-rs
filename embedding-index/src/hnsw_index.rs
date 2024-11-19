use crate::utils::{create_dir, from_config, index_file};
use anyhow::anyhow;
use embedding_common::config::IndexConfig;
use embedding_database::dao::dao_impl::has_embedding_user;
use embedding_database::db::rocksdb_impl::RocksDB;
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc};
use usearch::{new_index, Index};
use uuid::Uuid;

#[derive(Clone)]
pub struct HnswIndex {
    pub index: Arc<Index>,
    pub index_name: String,
    pub index_config: IndexConfig,
}

impl HnswIndex {
    pub fn load_index(index_name: String, index_config: IndexConfig) -> anyhow::Result<Self> {
        let options = from_config(&index_config);
        let index = match new_index(&options) {
            Err(e) => {
                return Err(anyhow::Error::msg(format!(
                    "Failed to create index: {:?}",
                    e
                )))
            }
            Ok(index) => index,
        };

        create_dir(index_config.index_dir.as_str())?;

        let index_file = index_file(index_name.clone(), &index_config);
        let file_path = Path::new(index_file.as_str());
        if file_path.exists() && file_path.is_file() {
            index.load(file_path.to_str().unwrap()).map_err(|e| anyhow!("Failed to load index: {:?}", e))?;
        } else {
            index
                .reserve(64)
                .map_err(|e| anyhow!("Failed to reserve index: {:?}", e))?;
        }

        let inner = Arc::new(index);
        let index_config = index_config.clone();
        Ok(Self { index: inner, index_name, index_config })
    }

    pub fn add(&self, rust_vec: &Vec<f32>, label: u64) -> anyhow::Result<()> {
        self.check_expand_capacity( 1)?;
        self.index
            .add(label, rust_vec)
            .map_err(|e| anyhow!("Failed to add item: {:?}", e))
    }

    pub fn upsert(&self, rust_vec: &Vec<f32>, label: u64) -> anyhow::Result<bool> {
        let contains = self.index.contains(label);
        if contains {
            self.index
                .remove(label)
                .map_err(|e| anyhow!("Failed to delete item: {:?}", e))?;
        }
        self.add(rust_vec, label)?;
        Ok(contains)
    }

    pub fn check_expand_capacity(&self, num: usize) -> anyhow::Result<()> {
        if self.index.capacity() <= self.index.size() + num {
            let num = if num > 64 { num } else { 64 };
            self.index
                .reserve(self.index.size() + num)
                .map_err(|e| anyhow!("Failed to reserve index: {:?}", e))
        } else {
            Ok(())
        }
    }

    pub fn batch_add(&self, embeddings: &Vec<Vec<f32>>, labels: &Vec<u64>) -> anyhow::Result<()> {
        self.check_expand_capacity(embeddings.len())?;
        embeddings.par_iter().enumerate().for_each(|(i, vec)| {
            self.index
                .add(labels[i], vec)
                .map_err(|e| anyhow::Error::msg(format!("Failed to add item: {:?}", e)))
                .unwrap();
        });
        Ok(())
    }

    pub fn delete(&self, label: u64) -> anyhow::Result<usize> {
        self.index
            .remove(label)
            .map_err(|e| anyhow::Error::msg(format!("Failed to delete item: {:?}", e)))
    }

    pub fn query_filter(
        &self,
        db: Arc<RocksDB>,
        user_uuid: &Uuid,
        query: &Vec<f32>,
        k: usize,
    ) -> anyhow::Result<(Vec<u64>, Vec<f32>)> {
        let matches = self.index
            .filtered_search(query, k, |key| {
                let embed_id: u64 = key.into();
                has_embedding_user(db.clone(), embed_id, user_uuid).unwrap()
            })
            .map_err(|e| anyhow!("Failed to query index: {:?}", e))?;
        Ok((matches.keys, matches.distances))
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let location = index_file(self.index_name.clone(), &self.index_config);
        self.index
            .save(location.as_str())
            .map_err(|e| anyhow!("Failed to save index: {:?}", e))
    }

    pub fn size(&self) -> usize {
        self.index.size()
    }

    pub fn capacity(&self) -> usize {
        self.index.capacity()
    }

    pub async fn async_upsert(index: Self, rust_vec: &Vec<f32>, label: u64) -> anyhow::Result<bool> {
        let embedding = rust_vec.clone();
        tokio::task::spawn(async move {
            index.upsert(&embedding, label)
        }).await?
    }

    pub async fn async_save(index: Self) -> anyhow::Result<()> {
        tokio::task::spawn(async move {
            index.save()
        }).await?
    }
}

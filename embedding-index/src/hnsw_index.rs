use std::path::Path;
use crate::utils::{create_dir, from_config, index_file};
use anyhow::anyhow;
use embedding_common::config::IndexConfig;
use embedding_database::dao::dao_impl::has_embedding_user;
use embedding_database::db::rocksdb_impl::RocksDB;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};
use uuid::Uuid;

pub struct HnswIndex {
    pub index: Arc<Mutex<Index>>,
}

impl HnswIndex {
    pub fn load_index(index_name: String, index_config: &IndexConfig) -> anyhow::Result<Self> {
        let options = from_config(index_config);
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

        let index_file = index_file(index_name, index_config);
        let file_path = Path::new(index_file.as_str());
        if file_path.exists() && file_path.is_file() {
            index.load(file_path.to_str().unwrap()).map_err(|e| anyhow!("Failed to load index: {:?}", e))?;
        } else {
            index
                .reserve(64)
                .map_err(|e| anyhow!("Failed to reserve index: {:?}", e))?;
        }

        let inner = Arc::new(Mutex::new(index));
        Ok(Self { index: inner })
    }

    pub fn add(&self, rust_vec: &Vec<f32>, label: u64) -> anyhow::Result<()> {
        let index = self
            .index
            .lock()
            .map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        Self::check_expand_capacity(&index, 1)?;
        index
            .add(label, rust_vec)
            .map_err(|e| anyhow::Error::msg(format!("Failed to add item: {:?}", e)))
    }

    pub fn check_expand_capacity(index: &Index, num: usize) -> anyhow::Result<()> {
        if index.capacity() <= index.size() + num {
            let num = if num > 64 { num } else { 64 };
            index
                .reserve(index.size() + num)
                .map_err(|e| anyhow!("Failed to reserve index: {:?}", e))
        } else {
            Ok(())
        }
    }

    pub fn batch_add(&self, embeddings: &Vec<Vec<f32>>, labels: &Vec<u64>) -> anyhow::Result<()> {
        let index = self
            .index
            .lock()
            .map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        Self::check_expand_capacity(&index, embeddings.len())?;
        embeddings.par_iter().enumerate().for_each(|(i, vec)| {
            index
                .add(labels[i], vec)
                .map_err(|e| anyhow::Error::msg(format!("Failed to add item: {:?}", e)))
                .unwrap();
        });
        Ok(())
    }

    pub fn delete(&self, label: u64) -> anyhow::Result<usize> {
        let index = self
            .index
            .lock()
            .map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        index
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
        let index = self
            .index
            .lock()
            .map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        let matches = index
            .filtered_search(query, k, |key| {
                let embed_id: u64 = key.into();
                has_embedding_user(db.clone(), embed_id, user_uuid).unwrap()
            })
            .map_err(|e| anyhow::Error::msg(format!("Failed to query index: {:?}", e)))?;
        Ok((matches.keys, matches.distances))
    }

    pub fn save(&self, location: &str) -> anyhow::Result<()> {
        let index = self
            .index
            .lock()
            .map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        index
            .save(location)
            .map_err(|e| anyhow::Error::msg(format!("Failed to save index: {:?}", e)))
    }

    pub fn size(&self) -> usize {
        let index = self.index.lock().unwrap();
        index.size()
    }

    pub fn capacity(&self) -> usize {
        let index = self.index.lock().unwrap();
        index.capacity()
    }
}

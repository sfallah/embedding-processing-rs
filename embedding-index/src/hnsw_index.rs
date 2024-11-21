use crate::utils::{create_dir, from_config, index_file};
use anyhow::anyhow;
use embedding_common::config::IndexConfig;
use embedding_database::dao::dao_impl::has_embedding_user;
use embedding_database::db::rocksdb_impl::RocksDB;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::task::spawn_blocking;
use usearch::{new_index, Index};
use uuid::Uuid;

use anyhow::Result;
use tracing::{error, info};

#[derive(Clone)]
pub struct HnswIndex {
    pub index: Arc<Mutex<Index>>,
    pub index_name: String,
    pub index_config: IndexConfig,
}

impl HnswIndex {
    #[tracing::instrument]
    fn create_load_index(index_name: String, index_config: IndexConfig) -> Result<Self> {
        let options = from_config(&index_config);
        let index = match new_index(&options) {
            Err(e) => {
                error!("Failed to create index: {:?}", e);
                return Err(anyhow!("Failed to create index: {:?}", e));
            }
            Ok(index) => index,
        };

        create_dir(index_config.index_dir.as_str())?;

        let index_file = index_file(index_name.clone(), &index_config);
        let file_path = Path::new(index_file.as_str());
        if file_path.exists() && file_path.is_file() {
            if let Err(e) = index.load(file_path.canonicalize()?.to_str().unwrap()) {
                error!("Failed to load index: {:?}", e);
                return Err(anyhow!("Failed to load index: {:?}", e));
            }
            info!("Loaded existing index: {}", index_file);
        } else {
            if let Err(e) = index.reserve(64) {
                error!("Failed to reserve index: {:?}", e);
                return Err(anyhow!("Failed to reserve index: {:?}", e));
            }
            info!("Created and reserved new index for: {}", index_file);
        }
        let inner = Arc::new(Mutex::new(index));
        let index_config = index_config.clone();
        Ok(Self {
            index: inner,
            index_name,
            index_config,
        })
    }

    #[tracing::instrument]
    pub async fn load_index(index_name: String, index_config: IndexConfig) -> Result<Self> {
        let index_config = index_config.clone();
        let index_name = index_name.clone();
        spawn_blocking(move || HnswIndex::create_load_index(index_name, index_config)).await?
    }

    pub async fn get_by_label(&self, label: u64) -> Result<Vec<f32>> {
        let index = self.index.clone();
        let dimensions = self.index_config.dimensions;
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            let mut embedding = Vec::with_capacity(dimensions);
            match index.export(label, &mut embedding) {
                Ok(_) => Ok(embedding.to_vec()),
                Err(e) => Err(anyhow!("Failed to get item: {:?}", e)),
            }
        })
            .await?
    }

    pub async fn add(&self, embd: &Vec<f32>, label: u64) -> Result<()> {
        let index = self.index.clone();
        let embd = embd.clone();
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            if index.capacity() <= index.size() + 1 {
                if let Err(e) = index.reserve(index.size() + 1) {
                    error!("Failed to reserve index: {:?}", e);
                    return Err(anyhow!("Failed to reserve index: {:?}", e));
                }
            }
            index
                .add(label, &embd)
                .map_err(|e| anyhow!("Failed to add item: {:?}", e))
        })
            .await?
    }

    #[tracing::instrument(skip(self, embeddings, labels))]
    pub async fn add_batch(&self, embeddings: &Vec<Vec<f32>>, labels: &Vec<u64>) -> Result<()> {
        let index = self.index.clone();
        let embeddings = embeddings.clone();
        let labels = labels.clone();
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            let num = embeddings.len();
            if index.capacity() <= index.size() + num {
                let num = if num > 64 { num } else { 64 };
                if let Err(e) = index.reserve(index.size() + num) {
                    error!("Failed to reserve index: {:?}", e);
                    return Err(anyhow!("Failed to reserve index: {:?}", e));
                }
                info!("Reserved {} more capacity, cur capacity: {}", num, index.capacity());
            }
            for (embd, label) in embeddings.iter().zip(labels.iter()) {
                if let Err(e) = index.add(*label, embd) {
                    error!("Failed to add item: {:?}", e);
                    return Err(anyhow!("Failed to add item: {:?}", e));
                }
            }
            info!("Added {} items to index", num);
            Ok(())
        })
            .await?
    }
    pub async fn upsert(&self, embd: &Vec<f32>, label: u64) -> Result<bool> {
        let index = self.index.clone();
        let embd = embd.clone();
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            let exists = index.contains(label);
            if exists {
                if let Err(e) = index.remove(label) {
                    error!("Failed to delete existing item: {:?}", e);
                    return Err(anyhow!("Failed to delete item: {:?}", e));
                }
            }
            if index.capacity() <= index.size() + 1 {
                if let Err(e) = index.reserve(index.size() + 1) {
                    error!("Failed to reserve index: {:?}", e);
                    return Err(anyhow!("Failed to reserve index: {:?}", e));
                }
            }
            if let Err(e) = index.add(label, &embd) {
                error!("Failed to add item: {:?}", e);
                return Err(anyhow!("Failed to add item: {:?}", e));
            }
            Ok(exists)
        })
            .await?
    }

    pub async fn upsert_batch(&self, embeddings: &Vec<Vec<f32>>, labels: &Vec<u64>) -> Result<Vec<u64>> {
        let index = self.index.clone();
        let embeddings = embeddings.clone();
        let labels = labels.clone();
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            let mut removed_keys = Vec::new();
            for label in labels.iter() {
                if index.contains(*label) {
                    if let Err(e) = index.remove(*label) {
                        error!("Failed to delete existing item: {:?}", e);
                        return Err(anyhow!("Failed to delete item: {:?}", e));
                    }
                }
                removed_keys.push(*label);
            }
            let num = labels.len();
            if index.capacity() <= index.size() + num {
                if let Err(e) = index.reserve(index.size() + num) {
                    error!("Failed to reserve index: {:?}", e);
                    return Err(anyhow!("Failed to reserve index: {:?}", e));
                }
            }
            for (embd, label) in embeddings.iter().zip(labels.iter()) {
                if let Err(e) = index.add(*label, embd) {
                    error!("Failed to add item: {:?}", e);
                    return Err(anyhow!("Failed to add item: {:?}", e));
                }
            }
            Ok(removed_keys)
        })
            .await?
    }

    pub async fn delete(&self, label: u64) -> Result<usize> {
        let index = self.index.clone();
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            index
                .remove(label)
                .map_err(|e| anyhow::Error::msg(format!("Failed to delete item: {:?}", e)))
        })
            .await?
    }

    pub async fn query_filter(
        &self,
        db: &Arc<RocksDB>,
        user_uuids: &Vec<Uuid>,
        query: &Vec<f32>,
        k: usize,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let index = self.index.clone();
        let db = db.clone();
        let query = query.clone();
        let user_uuids = user_uuids.clone();
        let matches = spawn_blocking(move || {
            let index = index.lock().unwrap();
            index
                .filtered_search(&query, k, |key| {
                    let embed_id: u64 = key.into();
                    has_embedding_user(&db, embed_id, user_uuids.clone()).unwrap()
                })
                .map_err(|e| anyhow!("Failed to query index: {:?}", e))
        })
            .await??;
        Ok((matches.keys, matches.distances))
    }

    pub async fn save(&self) -> Result<()> {
        let index = self.index.clone();
        let index_file = index_file(self.index_name.clone(), &self.index_config);
        spawn_blocking(move || {
            let index = index.lock().unwrap();
            let index_file_path = Path::new(index_file.as_str());
            if !index_file_path.exists() {
                let location = index_file_path.to_str().unwrap().to_string();
                if let Err(e) = index.save(location.clone().as_str()) {
                    error!(
                        "Failed to save to new index file: {:?} error: {:?}",
                        location, e
                    );
                    return Err(anyhow!(
                        "Failed to save to new index file: {:?} error: {:?}",
                        location,
                        e
                    ));
                }
                info!("Saved new index to: {}", location);
            } else {
                let old_file_path = index_file_path.canonicalize()?;
                let file_name = old_file_path.file_name().unwrap().to_str().unwrap();
                let backup_file = format!(
                    "{}/backup_{}",
                    old_file_path.parent().unwrap().to_str().unwrap(),
                    file_name
                ).to_string();
                if let Err(e) = std::fs::rename(old_file_path.clone(), backup_file.clone()) {
                    let old_file = old_file_path.to_str().unwrap();
                    error!(
                        "Failed to rename old index file: {:?}, to: {:?}, error: {:?}",
                        old_file, backup_file, e
                    );
                    return Err(anyhow!(
                        "Failed to rename old index file: {:?}, to: {:?}, error: {:?}",
                        old_file,
                        backup_file,
                        e
                    ));
                }
                info!("Backed up existing index-file to file: {}", backup_file);

                let location = index_file_path.to_str().unwrap().to_string();
                if let Err(e) = index.save(location.clone().as_str()) {
                    error!(
                        "Failed to save to index file: {:?} error: {:?}",
                        location, e
                    );
                    return Err(anyhow!(
                        "Failed to save to index file: {:?} error: {:?}",
                        location,
                        e
                    ));
                }
                if let Err(e) = std::fs::remove_file(backup_file.clone()) {
                    error!(
                        "Failed to remove backup index file: {:?} error: {:?}",
                        backup_file, e
                    );
                    return Err(anyhow!(
                        "Failed to remove backup index file: {:?} error: {:?}",
                        backup_file,
                        e
                    ));
                }
                info!("Overwrote existing index to: {}", location);
            }
            Ok(())
        })
            .await?
    }

    pub async fn size(&self) -> Result<usize> {
        let index = self.index.lock().unwrap();
        Ok(index.size())
    }

    pub async fn capacity(&self) -> Result<usize> {
        let index = self.index.lock().unwrap();
        Ok(index.capacity())
    }
}

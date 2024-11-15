use std::sync::{Arc, Mutex};
use anyhow::anyhow;
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};
use usearch::ffi::new_native_index;
use uuid::Uuid;
use embedding_database::dao::dao_impl::has_embedding_user;
use embedding_database::db::rocksdb_impl::RocksDB;

pub struct HnswIndex {
    pub index: Arc<Mutex<Index>>,
}

impl HnswIndex {
    pub fn new(
        dimensions: usize,
    ) -> anyhow::Result<Self> {
        let options = IndexOptions {
            dimensions, // necessary for most metric kinds
            metric: MetricKind::L2sq, // or ::L2sq, ::Cos ...
            quantization: ScalarKind::F32, // or ::F32, ::F16, ::I8, ::B1x8 ...
            connectivity: 0, // zero for auto
            expansion_add: 0, // zero for auto
            expansion_search: 0, // zero for auto
            multi: false,
        };

        let index = match new_index(&options) {
            Err(e) =>
                return Err(anyhow::Error::msg(format!("Failed to create index: {:?}", e))),
            Ok(index) => index,
        };
        index.reserve(1000).map_err(|e| anyhow!("Failed to reserve index: {:?}", e))?;
        //index.reserve(1000).map_err(|e| anyhow!("Failed to reserve index: {:?}", e))?;

        let inner = Arc::new(Mutex::new(index));
        Ok(Self { index: inner })
    }

    pub fn load(
        location: &str,
        dimensions: usize,
    ) -> anyhow::Result<Self> {
        let options = IndexOptions {
            dimensions, // necessary for most metric kinds
            metric: MetricKind::L2sq, // or ::L2sq, ::Cos ...
            quantization: ScalarKind::F32, // or ::F32, ::F16, ::I8, ::B1x8 ...
            connectivity: 0, // zero for auto
            expansion_add: 0, // zero for auto
            expansion_search: 0, // zero for auto
            multi: false,
        };

        let index = match new_index(&options) {
            Err(e) =>
                return Err(anyhow::Error::msg(format!("Failed to create index: {:?}", e))),
            Ok(index) => index,
        };
        match index.load(location) {
            Err(e) =>
                return Err(anyhow::Error::msg(format!("Failed to load index: {:?}", e))),
            Ok(_) => (),
        };
        let inner = Arc::new(Mutex::new(index));
        Ok(Self { index: inner })
    }

    pub fn add(&self, rust_vec: &Vec<f32>, label: u64) -> anyhow::Result<()> {
        let index = self.index.lock().map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        index.add(label, rust_vec).map_err(|e| anyhow::Error::msg(format!("Failed to add item: {:?}", e)))
    }

    pub fn delete(&self, label: u64) -> anyhow::Result<usize> {
        let index = self.index.lock().map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        index.remove(label).map_err(|e| anyhow::Error::msg(format!("Failed to delete item: {:?}", e)))
    }

    pub fn query_filter(&self, db: &Arc<RocksDB>, user_uuid: &Uuid, query: &Vec<f32>, k: usize) -> anyhow::Result<(Vec<u64>, Vec<f32>)> {
        let index = self.index.lock().map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        let matches = index.filtered_search(query, k, |key| {
            let embed_id: u64 = key.into();
            has_embedding_user(db, embed_id, user_uuid).unwrap()
        }).map_err(|e| anyhow::Error::msg(format!("Failed to query index: {:?}", e)))?;
        Ok((matches.keys, matches.distances))
    }

    pub fn save(&self, location: &str) -> anyhow::Result<()> {
        let index = self.index.lock().map_err(|e| anyhow!("Failed to lock index: {:?}", e))?;
        index.save(location).map_err(|e| anyhow::Error::msg(format!("Failed to save index: {:?}", e)))
    }
}

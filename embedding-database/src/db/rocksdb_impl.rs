use byteorder::{ByteOrder, LittleEndian};
use rocksdb::{
    ColumnFamilyDescriptor, DBCompressionType, DBWithThreadMode, IteratorMode, MultiThreaded,
    Options, WriteBatch,
};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use tokio::task;
use tracing::info;
use tracing::instrument;
use crate::db::column_families::ColumnFamilyType;

#[derive(Debug)]
pub struct RocksDB {
    pub db: Arc<DBWithThreadMode<MultiThreaded>>,
}

impl RocksDB {
    fn configure_options() -> Options {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.increase_parallelism(num_cpus::get() as i32);
        opts.optimize_level_style_compaction(0);
        opts.set_max_open_files(-1);
        opts.set_compression_type(DBCompressionType::Lz4);

        let cache = rocksdb::Cache::new_lru_cache(1 << 30);
        opts.set_row_cache(&cache);
        opts.set_arena_block_size(16 * 1024);

        opts
    }

    fn key_to_bytes(key: &u64) -> [u8; 8] {
        let mut key_bytes = [0u8; 8];
        LittleEndian::write_u64(&mut key_bytes, *key);
        key_bytes
    }

    fn open_column_families(
        path: &str,
        cfs: Vec<&str>,
    ) -> Result<DBWithThreadMode<MultiThreaded>> {
        let opts = Self::configure_options();
        let cf_descriptors: Vec<_> = cfs
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect();

        let db = DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(&opts, path, cf_descriptors)?;
        Ok(db)
    }

    /// Opens the RocksDB database asynchronously.
    #[instrument]
    pub async fn open(path: &str) -> Result<Self> {
        let cfs = vec!["default", "documents", "splits", "summaries", "embeddings"];
        let path = path.to_string();
        let path_clone = path.clone();

        info!("Opening RocksDB at path: {}", path);

        let db = task::spawn_blocking(move || -> Result<DBWithThreadMode<MultiThreaded>> {
            Self::open_column_families(&path, cfs)
        })
            .await??;

        info!("Successfully opened RocksDB at path: {}", path_clone);

        Ok(RocksDB {
            db: Arc::new(db),
        })
    }

    /// Asynchronously puts a key-value pair into the specified cf.
    #[instrument(skip(self, value))]
    pub async fn put(&self, cf: ColumnFamilyType, key: &u64, value: &[u8]) -> Result<()> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let value_len = value.len();
        let value = value.to_vec();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        info!("Putting key: {} into cf: {}", key, cf_name_clone);

        task::spawn_blocking(move || -> Result<()> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
            db.put_cf(&cf, &key_bytes, &value)?;
            Ok(())
        })
            .await??;

        info!(
            "Successfully put key: {} ({} bytes) into cf: {}",
            key, value_len, cf_name_clone
        );

        Ok(())
    }

    /// Asynchronously gets the value associated with a key from the specified cf.
    #[instrument(skip(self))]
    pub async fn get(&self, cf: ColumnFamilyType, key: &u64) -> Result<Option<Vec<u8>>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        info!("Getting key: {} from cf: {}", key, cf_name_clone);

        let result = task::spawn_blocking(move || -> Result<Option<Vec<u8>>> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
            let value = db.get_cf(&cf, &key_bytes)?;
            Ok(value)
        })
            .await??;

        match &result {
            Some(value) => info!(
        "Successfully got key: {} ({} bytes) from cf: {}",
        key,
        value.len(),
        cf_name_clone
    ),
            None => info!("Key: {} not found in cf: {}", key, cf_name_clone),
        }

        Ok(result)
    }

    /// Asynchronously retrieves all values from the specified cf.
    #[instrument(skip(self))]
    pub async fn get_all(&self, cf: ColumnFamilyType) -> Result<Vec<Vec<u8>>> {
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        info!("Getting all values from cf: {}", cf_name_clone);

        let values = task::spawn_blocking(move || -> Result<Vec<Vec<u8>>> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;

            let iter = db.iterator_cf(&cf, IteratorMode::Start);
            let mut values = Vec::new();

            for item in iter {
                let (_key, value) = item?;
                values.push(value.to_vec());
            }

            Ok(values)
        })
            .await??;

        info!(
            "Successfully retrieved {} values from cf: {}",
            values.len(),
            cf_name_clone
        );

        Ok(values)
    }

    /// Asynchronously gets multiple values associated with the provided keys from the specified cf.
    #[instrument(skip(self, keys))]
    pub async fn multi_get(&self, cf: ColumnFamilyType, keys: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let db = self.db.clone();
        let keys = keys.to_vec();
        let keys_len = keys.len();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        info!(
            "Performing multi_get for {} keys in cf: {}",
            keys_len, cf_name_clone
        );

        let results = task::spawn_blocking(move || -> Result<Vec<Option<Vec<u8>>>> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;

            let mut results = Vec::with_capacity(keys.len());
            for key in keys {
                let key_bytes = Self::key_to_bytes(&key);
                let value = db.get_cf(&cf, &key_bytes)?;
                results.push(value);
            }
            Ok(results)
        })
            .await??;

        info!(
            "multi_get completed for {} keys in column family: {}",
            keys_len, cf_name_clone
        );


        Ok(results)
    }

    /// Asynchronously deletes a key-value pair from the specified cf.
    #[instrument(skip(self))]
    pub async fn delete(&self, cf: ColumnFamilyType, key: &u64) -> Result<()> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        info!("Deleting key: {} from cf: {}", key, cf_name_clone);

        task::spawn_blocking(move || -> Result<()> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
            db.delete_cf(&cf, &key_bytes)?;
            Ok(())
        })
            .await??;

        info!("Successfully deleted key: {} from cf: {}", key, cf_name_clone);

        Ok(())
    }

    /// Asynchronously deletes multiple key-value pairs from the specified cf.
    #[instrument(skip(self, keys))]
    pub async fn delete_many(&self, cf: ColumnFamilyType, keys: &[u64]) -> Result<()> {
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();
        let keys = keys.to_vec();
        let keys_len = keys.len();

        info!(
            "Deleting {} keys from cf: {}",
            keys_len, cf_name_clone
        );

        task::spawn_blocking(move || -> Result<()> {
            let cf = db
                .cf_handle(&cf_name)
                .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;

            let mut batch = WriteBatch::default();

            for key in keys {
                let key_bytes = Self::key_to_bytes(&key);
                batch.delete_cf(&cf, &key_bytes);
            }

            db.write(batch)?;
            Ok(())
        })
            .await??;

        info!(
            "Successfully deleted {} keys from cf: {}",
            keys_len, cf_name_clone
        );

        Ok(())
    }
}

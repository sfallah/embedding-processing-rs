use crate::db::column_families::ColumnFamilyType;
use crate::db::db_record::{DbRecordKey, DbRecordValue};
use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use rocksdb::{
    ColumnFamilyDescriptor, DBCompressionType, IteratorMode, MultiThreaded,
    OptimisticTransactionDB, Options, WriteBatchWithTransaction,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;
use tracing::{debug, info, trace};

#[derive(Debug)]
pub struct RocksDB {
    pub db: Arc<OptimisticTransactionDB<MultiThreaded>>,
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
        cfs: &Vec<String>,
    ) -> Result<OptimisticTransactionDB<MultiThreaded>> {
        let opts = Self::configure_options();
        let cf_descriptors: Vec<_> = cfs
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect();

        let db = OptimisticTransactionDB::<MultiThreaded>::open_cf_descriptors(
            &opts,
            path,
            cf_descriptors,
        )?;
        Ok(db)
    }

    /// Opens the RocksDB database synchronously.
    pub fn open(path: &str) -> Result<Self> {
        let cfs = ColumnFamilyType::all_column_families();

        let path = path.to_string();
        let path_clone = path.clone();

        info!("Opening RocksDB at path: {}", path);

        let db = Self::open_column_families(&path, &cfs)?;

        info!("Successfully opened RocksDB at path: {}", path_clone);

        Ok(RocksDB { db: Arc::new(db) })
    }

    /// hronously puts a key-value pair into the specified cf.
    pub fn put(&self, cf: ColumnFamilyType, key: &u64, value: &[u8]) -> Result<()> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let value = value.to_vec();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        trace!("Putting key: {} into cf: {}", key, cf_name_clone);

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
        db.put_cf(&cf, &key_bytes, &value)?;
        Ok(())
    }

    /// hronously gets the value associated with a key from the specified cf.
    pub fn get(&self, cf: ColumnFamilyType, key: &u64) -> Result<Option<Vec<u8>>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        //trace!("Getting key: {} from cf: {}", key, cf_name_clone);

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
        let result = db.get_cf(&cf, &key_bytes)?;

        match &result {
            Some(value) => debug!(
                "Successfully got key: {} ({} bytes) from cf: {}",
                key,
                value.len(),
                cf_name_clone
            ),
            None => debug!("Key: {} not found in cf: {}", key, cf_name_clone),
        }

        Ok(result)
    }

    pub fn get_sync(&self, cf: ColumnFamilyType, key: &u64) -> Result<Option<Vec<u8>>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let cf_name = cf.name().to_string();

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
        let value = db.get_cf(&cf, &key_bytes)?;
        Ok(value)
    }

    /// hronously retrieves all values from the specified cf.
    pub fn get_all(&self, cf: ColumnFamilyType) -> Result<Vec<Vec<u8>>> {
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        debug!("Getting all values from cf: {}", cf_name_clone);

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;

        let iter = db.iterator_cf(&cf, IteratorMode::Start);
        let mut values = Vec::new();

        for item in iter {
            let (_key, value) = item?;
            values.push(value.to_vec());
        }

        debug!(
            "Successfully retrieved {} values from cf: {}",
            values.len(),
            cf_name_clone
        );

        Ok(values)
    }

    /// hronously gets multiple values associated with the provided keys from the specified cf.
    pub fn multi_get(&self, cf: ColumnFamilyType, keys: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let db = self.db.clone();
        let keys = keys.to_vec();
        let cf_name = cf.name().to_string();

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
    }

    /// hronously deletes a key-value pair from the specified cf.
    pub fn delete(&self, cf: ColumnFamilyType, key: &u64) -> Result<()> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();

        debug!("Deleting key: {} from cf: {}", key, cf_name_clone);

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
        db.delete_cf(&cf, &key_bytes)?;

        debug!(
            "Successfully deleted key: {} from cf: {}",
            key, cf_name_clone
        );

        Ok(())
    }

    /// hronously deletes multiple key-value pairs from the specified cf.
    pub fn multi_delete(&self, cf: ColumnFamilyType, keys: &[u64]) -> Result<()> {
        let db = self.db.clone();
        let cf_name = cf.name().to_string();
        let cf_name_clone = cf_name.clone();
        let keys = keys.to_vec();
        let keys_len = keys.len();
        debug!("Deleting {} keys from cf: {}", keys_len, cf_name_clone);

        let cf = db
            .cf_handle(&cf_name)
            .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;

        let mut batch = WriteBatchWithTransaction::default();

        for key in keys {
            let key_bytes = Self::key_to_bytes(&key);
            batch.delete_cf(&cf, &key_bytes);
        }

        db.write(batch)?;

        debug!(
            "Successfully deleted {} keys from cf: {}",
            keys_len, cf_name_clone
        );

        Ok(())
    }

    pub fn save_records(&self, records: Arc<Vec<DbRecordValue>>) -> Result<()> {
        let db = self.db.clone();
        let records = records.clone();
        let mut cf_map = HashMap::new();
        let mut batch = WriteBatchWithTransaction::default();
        for record in records.iter() {
            let cf = if !cf_map.contains_key(&record.cf_type) {
                let cf = db.cf_handle(&record.cf_type.name()).ok_or_else(|| {
                    anyhow!("Column family '{}' not found", record.cf_type.name())
                })?;
                cf_map.insert(record.cf_type, cf.clone());
                cf
            } else {
                cf_map.get(&record.cf_type).unwrap().clone()
            };
            let key_bytes = Self::key_to_bytes(&record.key);
            batch.put_cf(&cf, key_bytes, &record.value);
        }
        db.write(batch)?;

        Ok(())
    }

    pub(crate) fn delete_records(&self, records: Arc<Vec<DbRecordKey>>) -> Result<()> {
        let db = self.db.clone();
        let records = records.clone();
        let mut cf_map = HashMap::new();
        let mut batch = WriteBatchWithTransaction::default();
        for record in records.iter() {
            let cf = if !cf_map.contains_key(&record.cf_type) {
                let cf = db.cf_handle(&record.cf_type.name()).ok_or_else(|| {
                    anyhow!("Column family '{}' not found", record.cf_type.name())
                })?;
                cf_map.insert(record.cf_type, cf.clone());
                cf
            } else {
                cf_map.get(&record.cf_type).unwrap().clone()
            };
            let key_bytes = Self::key_to_bytes(&record.key);
            batch.delete_cf(&cf, key_bytes);
        }
        db.write(batch)?;

        Ok(())
    }
}

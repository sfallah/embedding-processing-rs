use crate::db::interface::{Database, Table};
use byteorder::{ByteOrder, LittleEndian};
use rocksdb::{
    ColumnFamilyDescriptor, DBCompressionType, DBWithThreadMode, IteratorMode, MultiThreaded,
    Options, WriteBatch,
};
use std::error::Error;
use std::sync::{Arc, Mutex};

pub struct RocksDB {
    pub db: Arc<Mutex<DBWithThreadMode<MultiThreaded>>>,
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
    ) -> Result<DBWithThreadMode<MultiThreaded>, Box<dyn Error>> {
        let opts = Self::configure_options();
        let cf_descriptors: Vec<_> = cfs
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect();

        let db =
            DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(&opts, path, cf_descriptors)?;
        Ok(db)
    }
}

impl Database for RocksDB {
    fn open(path: &str) -> Self {
        let cfs = vec!["default", "documents", "splits", "summaries", "embeddings"];
        let db = Self::open_column_families(path, cfs).expect("Failed to open database");
        RocksDB {
            db: Arc::new(Mutex::new(db)),
        }
    }

    fn put(&self, table: Table, key: &u64, value: &[u8]) -> Result<(), Box<dyn Error>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;
        db.put_cf(&cf, &key_bytes, value).map_err(Into::into)
    }

    fn get(&self, table: Table, key: &u64) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;
        db.get_cf(&cf, &key_bytes).map_err(Into::into)
    }

    fn get_all(&self, table: Table) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;

        let iter = db.iterator_cf(&cf, IteratorMode::Start);
        let mut values = Vec::new();

        for item in iter {
            let (_key, value) = item?;
            values.push(value.to_vec());
        }

        Ok(values)
    }

    fn multi_get(
        &self,
        table: Table,
        keys: &[u64],
    ) -> Result<Vec<Option<Vec<u8>>>, Box<dyn Error>> {
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;

        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            let key_bytes = Self::key_to_bytes(key);
            let value = db.get_cf(&cf, &key_bytes)?;
            results.push(value);
        }
        Ok(results)
    }

    fn delete(&self, table: Table, key: &u64) -> Result<(), Box<dyn Error>> {
        let key_bytes = Self::key_to_bytes(key);
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;
        db.delete_cf(&cf, &key_bytes).map_err(Into::into)
    }

    fn delete_many(&self, table: Table, keys: &[u64]) -> Result<(), Box<dyn Error>> {
        let db = self.db.lock().unwrap();
        let cf = db
            .cf_handle(table.name())
            .ok_or("Column family not found")?;

        let mut batch = WriteBatch::default();

        for key in keys {
            let key_bytes = Self::key_to_bytes(key);
            batch.delete_cf(&cf, &key_bytes);
        }

        db.write(batch).map_err(Into::into)
    }
}

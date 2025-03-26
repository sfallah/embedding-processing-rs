#[cfg(test)]
mod tests {
    use anyhow::Result;
    use embedding_database::prelude::{ColumnFamilyType, RocksDB};
    use fake::faker::lorem::en::*;
    use fake::{Fake, Faker};
    use tempdir::TempDir;

    #[test]
    fn test_put() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_put")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let key: u64 = Faker.fake();
        let value: String = Sentence(1..3).fake();

        // Act
        rocksdb.put(ColumnFamilyType::Default, &key, value.as_bytes())?;

        // Assert
        let retrieved_value = rocksdb.get(ColumnFamilyType::Default, &key)?;
        assert_eq!(retrieved_value, Some(value.into_bytes()));

        Ok(())
    }

    #[test]
    fn test_get() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_get")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let key: u64 = Faker.fake();
        let value: String = Sentence(1..3).fake();

        rocksdb.put(ColumnFamilyType::Default, &key, value.as_bytes())?;

        // Act
        let retrieved_value = rocksdb.get(ColumnFamilyType::Default, &key)?;

        // Assert
        assert_eq!(retrieved_value, Some(value.into_bytes()));

        Ok(())
    }

    #[test]
    fn test_multi_put() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_multi_put")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let keys: Vec<u64> = (0..10).map(|_| Faker.fake()).collect();
        let values: Vec<String> = (0..10).map(|_| Sentence(5..10).fake()).collect();

        // Act
        for (key, value) in keys.iter().zip(values.iter()) {
            rocksdb.put(ColumnFamilyType::Default, key, value.as_bytes())?;
        }

        // Assert
        for (key, value) in keys.iter().zip(values.iter()) {
            let retrieved_value = rocksdb.get(ColumnFamilyType::Default, key)?;
            assert_eq!(retrieved_value, Some(value.as_bytes().to_vec()));
        }

        Ok(())
    }

    #[test]
    fn test_multi_get() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_multi_get")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let keys: Vec<u64> = (0..10).map(|_| Faker.fake()).collect();
        let values: Vec<String> = (0..10).map(|_| Sentence(5..10).fake()).collect();

        for (key, value) in keys.iter().zip(values.iter()) {
            rocksdb.put(ColumnFamilyType::Default, key, value.as_bytes())?;
        }

        // Act
        let retrieved_values = rocksdb.multi_get(ColumnFamilyType::Default, &keys)?;

        // Assert
        for (retrieved, original) in retrieved_values.iter().zip(values.iter()) {
            assert_eq!(
                retrieved.as_ref().map(|v| v.as_slice()),
                Some(original.as_bytes())
            );
        }

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_delete")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let key: u64 = Faker.fake();
        let value: String = Sentence(3..5).fake();

        rocksdb.put(ColumnFamilyType::Default, &key, value.as_bytes())?;

        // Act
        rocksdb.delete(ColumnFamilyType::Default, &key)?;

        // Assert
        let retrieved_value = rocksdb.get(ColumnFamilyType::Default, &key)?;
        assert!(retrieved_value.is_none());

        Ok(())
    }

    #[test]
    fn test_delete_many() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_delete_many")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;
        let keys: Vec<u64> = (0..5).map(|_| Faker.fake()).collect();

        for key in &keys {
            let value: String = Sentence(3..6).fake();
            rocksdb.put(ColumnFamilyType::Default, key, value.as_bytes())?;
        }

        // Act
        rocksdb.multi_delete(ColumnFamilyType::Default, &keys)?;

        // Assert
        for key in &keys {
            let value = rocksdb.get(ColumnFamilyType::Default, key)?;
            assert!(value.is_none());
        }

        Ok(())
    }

    #[test]
    fn test_get_all() -> Result<()> {
        // Arrange
        let temp_dir = TempDir::new("test_rocksdb_get_all")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path)?;

        for _ in 0..10 {
            let key: u64 = Faker.fake();
            let value: String = Sentence(3..6).fake();
            rocksdb.put(ColumnFamilyType::Default, &key, value.as_bytes())?;
        }

        // Act
        let all_values = rocksdb.get_all(ColumnFamilyType::Default)?;

        // Assert
        assert_eq!(all_values.len(), 10);

        Ok(())
    }
}

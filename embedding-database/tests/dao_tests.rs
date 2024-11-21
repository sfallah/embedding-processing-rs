#[cfg(test)]
mod tests {
    use embedding_common::models::embedding::EmbeddingUser;
    use embedding_database::dao::dao_impl::{async_get_embedding_user, has_embedding_user, put_embedding_user};
    use embedding_database::db::rocksdb_impl::RocksDB;
    use fake::{rand, Rng};
    use std::sync::Arc;
    use std::vec;
    use tempdir::TempDir;
    use uuid::Uuid;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_put() -> anyhow::Result<()> {
        // Arrange
        let temp_dir = TempDir::new("dao_test_rocksdb")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let user_uuid = Uuid::new_v4();
        let embed_id: u64 = rand::thread_rng().gen();
        let embedding_user = EmbeddingUser {
            embed_id,
            user_uuid,
        };
       let put_res = put_embedding_user(&rocksdb, &embedding_user).await?;
         assert_eq!(put_res, ());
        let get_res = async_get_embedding_user(&rocksdb, embed_id).await?;
        assert_eq!(get_res, Some(embedding_user));

        let is_user = has_embedding_user(rocksdb.clone(), embed_id, vec![user_uuid])?;
        assert_eq!(is_user, true);

        let not_user = has_embedding_user(rocksdb.clone(), embed_id, vec![Uuid::new_v4()])?;
        assert_eq!(not_user, false);
        Ok(())
    }
}
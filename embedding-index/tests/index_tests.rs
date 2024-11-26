#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::config::AppConfig;
    use embedding_common::models::embedding::EmbeddingUser;
    use embedding_database::dao::embedding_user_dao::put_embedding_user;
    use embedding_database::db::rocksdb_impl::RocksDB;
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::utils::{generate_random_vectors, index_file};
    use rand::{thread_rng, Rng};
    use std::sync::Arc;
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;
    use uuid::Uuid;

    async fn read_config() -> anyhow::Result<AppConfig> {
        let config_file = "tests/test_config/index_config_test.toml".to_string();
        AppConfig::from_file_async(config_file).await
    }

    async fn generate_test_data(
        num_users: usize,
        num_user_embeds: usize,
    ) -> anyhow::Result<Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>> {
        let mut user_ids = vec![];
        for _ in 0..num_users {
            user_ids.push(Uuid::new_v4());
        }

        let num_embeds = num_users * num_user_embeds;

        let mut embed_ids = vec![0u64; num_embeds];
        for embed_id in embed_ids.iter_mut() {
            *embed_id = thread_rng().gen()
        }

        let embeddings = generate_random_vectors(num_embeds, 384);

        let records: Vec<_> = user_ids
            .iter()
            .enumerate()
            .map(|(i, user_id)| {
                let embed_ids = <Vec<u64> as Clone>::clone(&embed_ids)
                    .into_iter()
                    .skip(i * num_user_embeds)
                    .take(num_user_embeds)
                    .collect::<Vec<_>>();
                let embeddings = <Vec<Vec<f32>> as Clone>::clone(&embeddings)
                    .into_iter()
                    .skip(i * num_user_embeds)
                    .take(num_user_embeds)
                    .collect::<Vec<_>>();
                (*user_id, embed_ids, embeddings)
            })
            .collect();

        Ok(records)
    }

    async fn add_records(
        rocksdb: &Arc<RocksDB>,
        index: &HnswIndex,
        records: &Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>,
    ) -> anyhow::Result<()> {
        for (user_id, embed_ids, embeddings) in records.iter() {
            add_embedding_users(&rocksdb, user_id, embed_ids, embeddings).await?;
            index.add_batch(embeddings, embed_ids).await?;
        }
        Ok(())
    }

    async fn add_embedding_users(
        rocksdb: &&Arc<RocksDB>,
        user_id: &Uuid,
        embed_ids: &Vec<u64>,
        embeddings: &Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        for (embed_id, _) in embed_ids.iter().zip(embeddings.iter()) {
            let embedding_user = EmbeddingUser {
                user_uuid: *user_id,
                embed_id: *embed_id,
            };
            put_embedding_user(&rocksdb, &embedding_user)
                .await
                .map_err(|e| anyhow!("Failed to put embedding user: {}", e))?;
        }
        Ok(())
    }

    async fn upsert_records(
        rocksdb: &Arc<RocksDB>,
        index: &HnswIndex,
        records: &Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>,
    ) -> anyhow::Result<usize> {
        let mut num_overwrittens = 0;
        for (user_id, embed_ids, embeddings) in records.iter() {
            add_embedding_users(&rocksdb, user_id, embed_ids, embeddings).await?;
            let overwrittens = index.upsert_batch(embeddings, embed_ids).await?;
            num_overwrittens += overwrittens.len();
        }
        Ok(num_overwrittens)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_read_config() -> anyhow::Result<()> {
        let app_config = read_config().await?;
        println!("app_config: {:?}", app_config);
        Ok(())
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn index_add_test() -> anyhow::Result<()> {
        let config = read_config().await?;
        let index = HnswIndex::load_index("test_index".to_string(), config.index_config).await?;

        let embeddings = generate_random_vectors(2, 384);
        let embedding1 = embeddings.get(0).ok_or(anyhow!("No embeddings"))?;
        let embed_id = thread_rng().gen();
        index.add(embedding1, embed_id).await?;
        let embedding2 = embeddings.get(1).ok_or(anyhow!("No embeddings"))?;
        index.upsert(embedding2, embed_id).await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn config_load() -> anyhow::Result<()> {
        let app_config = read_config().await?;
        let index_config = app_config.index_config;
        println!("index_config: {:?}", index_config);
        let index_file = index_file("summaries".to_string(), &index_config);
        assert_eq!(
            index_file,
            "index_dir/summaries__L2sq_F32_384_16_32_32.usearch"
        );
        println!("index_file: {:?}", index_file);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn index_filter_test() -> anyhow::Result<()> {
        setup_tracing(Level::INFO);
        let db_temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = db_temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let mut app_config = read_config().await?;
        let index_tmp_dir = tempdir::TempDir::new("test_index")?;
        let index_path = index_tmp_dir.path().to_str().unwrap();
        app_config.index_config.index_dir = index_path.to_string();

        let index =
            HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone()).await?;

        let num_users = 2;
        let num_user_embeds = 10;

        let records = generate_test_data(num_users, num_user_embeds).await?;

        assert_eq!(records.len(), num_users);
        assert_eq!(records[0].1.len(), num_user_embeds);

        add_records(&rocksdb, &index, &records).await?;

        check_contained_embeddings(&rocksdb, &index, &records).await?;

        let size = index.size().await?;
        assert_eq!(size, num_users * num_user_embeds);
        println!("Size: {:?}", size);
        let capacity = index.capacity().await?;
        assert!(capacity >= num_users * num_user_embeds);
        println!("Capacity: {:?}", capacity);

        // reinsert records
        let num_overwritten = upsert_records(&rocksdb, &index, &records).await?;
        assert_eq!(num_overwritten, num_users * num_user_embeds);
        let size = index.size().await?;
        assert_eq!(size, num_users * num_user_embeds);
        println!("After reinsert Size: {:?}", size);
        let capacity = index.capacity().await?;
        assert!(capacity >= num_users * num_user_embeds);
        println!("After reinsert Capacity: {:?}", capacity);

        index.save().await?;

        let index2 =
            HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone()).await?;
        let size2 = index2.size().await?;
        assert_eq!(size2, num_users * num_user_embeds);
        println!("Old Size: {:?}", size2);
        println!("Old Capacity: {:?}", index2.capacity().await?);

        check_contained_embeddings(&rocksdb, &index2, &records).await?;

        let records2 = generate_test_data(num_users, num_user_embeds).await?;

        add_records(&rocksdb, &index2, &records2).await?;
        let size3 = index2.size().await?;
        assert_eq!(size3, num_users * num_user_embeds * 2);
        println!("Cur Size: {:?}", size3);
        println!("Cur Capacity: {:?}", index2.capacity().await?);
        index2.save().await?;

        let index3 =
            HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone()).await?;
        let size4 = index3.size().await?;
        assert_eq!(size4, num_users * num_user_embeds * 2);
        println!("Reload Size: {:?}", size4);
        println!("Reload Capacity: {:?}", index3.capacity().await?);

        Ok(())
    }

    async fn check_contained_embeddings(
        rocksdb: &Arc<RocksDB>,
        index: &HnswIndex,
        records: &Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>,
    ) -> anyhow::Result<()> {
        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                let res = index
                    .query_filter(&rocksdb, &vec![user_id.clone()], embedding, 4)
                    .await?;
                assert_eq!(res.0.len(), 4);
                assert_eq!(res.1.len(), 4);
                let first_label = res.0[0];
                let first_dist = res.1[0];
                assert_eq!(first_label, *embed_id);
                assert_eq!(first_dist, 0.0);
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_index() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let app_config = read_config().await?;
        let index = HnswIndex::load_index("summaries".to_string(), app_config.index_config).await?;

        let mut user_ids = vec![];
        for _ in 0..4 {
            user_ids.push(Uuid::new_v4());
        }

        let num_embeds = 400;

        let mut embed_ids = vec![0u64; num_embeds];
        for embed_id in embed_ids.iter_mut() {
            *embed_id = thread_rng().gen()
        }

        let embeddings = generate_random_vectors(num_embeds, 384);

        let each_embeds = num_embeds / user_ids.len();

        let records: Vec<_> = user_ids
            .iter()
            .enumerate()
            .map(|(i, user_id)| {
                let embed_ids = <Vec<u64> as Clone>::clone(&embed_ids)
                    .into_iter()
                    .skip(i * each_embeds)
                    .take(each_embeds)
                    .collect::<Vec<_>>();
                let embeddings = <Vec<Vec<f32>> as Clone>::clone(&embeddings)
                    .into_iter()
                    .skip(i * each_embeds)
                    .take(each_embeds)
                    .collect::<Vec<_>>();
                (*user_id, embed_ids, embeddings)
            })
            .collect();

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, _) in embed_ids.iter().zip(embeddings.iter()) {
                let embedding_user = EmbeddingUser {
                    user_uuid: *user_id,
                    embed_id: *embed_id,
                };
                put_embedding_user(&rocksdb, &embedding_user)
                    .await
                    .map_err(|e| anyhow!("Failed to put embedding user: {}", e))?;
            }
            index.add_batch(embeddings, embed_ids).await?;
        }

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.into_iter().zip(embeddings.into_iter()) {
                let res = index
                    .query_filter(&rocksdb, &vec![user_id.clone()], embedding, 10)
                    .await?;
                assert_eq!(res.0.len(), 10);
                assert_eq!(res.1.len(), 10);
                let first_label = res.0[0];
                let first_dist = res.1[0];
                println!("embed_id: {}, res: {:?}", embed_id, res);
                let orig_embd = index.get_by_label(*embed_id).await?;
                assert_eq!(orig_embd, *embedding);

                assert_eq!(
                    first_label, *embed_id,
                    "embed_id: {}, res: {:?}",
                    embed_id, res
                );
                assert_eq!(first_dist, 0.0);
            }
        }

        let size = index.size().await?;
        assert_eq!(size, num_embeds);
        println!("Size: {:?}", size);

        let capacity = index.capacity().await?;
        assert!(capacity >= num_embeds);
        println!("Capacity: {:?}", capacity);
        Ok(())
    }

    fn setup_tracing(level: Level) {
        let subscriber = FmtSubscriber::builder()
            // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
            // will be written to stdout.
            .with_max_level(level)
            // completes the builder.
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed")
    }
}

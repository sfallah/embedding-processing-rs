#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::models::embedding::EmbeddingUser;
    use embedding_database::dao::dao_impl::put_embedding_user;
    use embedding_database::db::rocksdb_impl::RocksDB;
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::utils::{generate_random_vectors, index_file};
    use rand::{thread_rng, Rng};
    use std::sync::Arc;
    use uuid::Uuid;
    use embedding_common::config::AppConfig;

    fn read_config() -> anyhow::Result<AppConfig> {
        let config_file = "tests/test_config/index_config_test.toml".to_string();
        let app_config =AppConfig::from_file(config_file)?;
        Ok(app_config)
    }
    #[test]
    fn index_add_test() -> anyhow::Result<()> {
        let config = read_config()?;
        let index = HnswIndex::load_index("test_index".to_string(), config.index_config)?;

        let embeddings = generate_random_vectors(2, 384);
        let embedding1 = embeddings.get(0).ok_or(anyhow!("No embeddings"))?;
        let embed_id = thread_rng().gen();
        index.add(embedding1, embed_id)?;
        let embedding2 = embeddings.get(1).ok_or(anyhow!("No embeddings"))?;
        index.upsert(embedding2, embed_id)?;
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread")]
    async fn config_load() -> anyhow::Result<()> {
        let app_config = read_config()?;
        let index_config = app_config.index_config;
        println!("index_config: {:?}", index_config);
        let index_file = index_file("summaries".to_string(), &index_config);
        assert_eq!(index_file, "index_dir/summaries__L2sq_F16_384_16_32_32.usearch");
        println!("index_file: {:?}", index_file);
        Ok(())
    }



    #[tokio::test(flavor = "multi_thread")]
    async fn index_filter_test() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let app_config = read_config()?;
        let index = HnswIndex::load_index("summaries".to_string(), app_config.index_config)?;

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
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                let embedding_user = EmbeddingUser {
                    user_uuid: *user_id,
                    embed_id: *embed_id,
                };
                put_embedding_user(&rocksdb, &embedding_user).await.map_err(|e| anyhow!("Failed to put embedding user: {}", e))?;
                let res = tokio::task::block_in_place(|| {
                    index
                        .add(embedding, *embed_id)
                        .map_err(|e| anyhow!("Failed to add item to index: {}", e))
                });
                match res {
                    Ok(_) => (),
                    Err(e) => {
                        eprintln!("Failed to add item to index: {}", e);
                    }
                }
            }
        }

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                let res = index.query_filter(rocksdb.clone(), user_id, embedding, 4)?;
                assert_eq!(res.0.len(), 4);
                assert_eq!(res.1.len(), 4);
                let first_label = res.0[0];
                let first_dist = res.1[0];
                assert_eq!(first_label, *embed_id);
                assert_eq!(first_dist, 0.0);
            }
        }

        let size = index.size();
        assert_eq!(size, num_embeds);
        println!("Size: {:?}", size);

        let capacity = index.capacity();
        assert!(capacity >= num_embeds);
        println!("Capacity: {:?}", capacity);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_index() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let app_config = read_config()?;
        let index = HnswIndex::load_index("summaries".to_string(), app_config.index_config)?;

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
                put_embedding_user(&rocksdb, &embedding_user).await.map_err(|e| anyhow!("Failed to put embedding user: {}", e))?;
            }
            let res = tokio::task::block_in_place(|| {
                index.batch_add(embeddings, embed_ids)
                    .map_err(|e| anyhow!("Failed to add item to index: {}", e))
            });
            match res {
                Ok(_) => (),
                Err(e) => {
                    eprintln!("Failed to add item to index: {}", e);
                }
            }
        }

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                let res = index.query_filter(rocksdb.clone(), user_id, embedding, 4)?;
                assert_eq!(res.0.len(), 4);
                assert_eq!(res.1.len(), 4);
                let first_label = res.0[0];
                let first_dist = res.1[0];
                assert_eq!(first_label, *embed_id);
                assert_eq!(first_dist, 0.0);
                println!("embed_id: {}, res: {:?}", embed_id, res);
            }
        }

        let size = index.size();
        assert_eq!(size, num_embeds);
        println!("Size: {:?}", size);

        let capacity = index.capacity();
        assert!(capacity >= num_embeds);
        println!("Capacity: {:?}", capacity);
        Ok(())
    }
}

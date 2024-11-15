#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::models::embedding::EmbeddingUser;
    use embedding_database::dao::dao_impl::put_embedding_user;
    use embedding_database::db::rocksdb_impl::RocksDB;
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::utils::generate_random_vectors;
    use rand::{thread_rng, Rng};
    use std::sync::Arc;
    use usearch::{new_index, IndexOptions, MetricKind, ScalarKind};
    use uuid::Uuid;

    #[test]
    fn index_add_test() -> anyhow::Result<()> {
        let options = IndexOptions {
            dimensions: 384, // necessary for most metric kinds
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

        let binding = generate_random_vectors(1, 384);
        let embedding = binding.get(0).ok_or(anyhow!("No embeddings"))?;
        let embed_id = thread_rng().gen();
        index.add(embed_id, embedding)?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn index_filter_test() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path).await?);

        let index = HnswIndex::new(384)?;

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

        let index = HnswIndex::new(384)?;

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

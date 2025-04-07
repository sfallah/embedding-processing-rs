#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::config::config_file::ConfigFromFile;
    use embedding_common::config::AppConfig;
    use embedding_common::prelude::EmbeddingFilterInfo;
    use embedding_database::prelude::{to_embedding_filter_info, RocksDB};
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::index_record::IndexRecord;
    use embedding_index::utils::{generate_random_vectors, index_file};
    use rand::{thread_rng, Rng};
    use simsimd::SpatialSimilarity;
    use std::f32;
    use std::sync::Arc;
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;
    use uuid::Uuid;

    fn read_config() -> anyhow::Result<AppConfig> {
        let config_file = "tests/test_index_config.toml".to_string();
        AppConfig::from_file(config_file)
    }

    fn generate_test_data(
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

    fn add_records(
        rocksdb: &Arc<RocksDB>,
        index: &HnswIndex,
        records: &Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>,
    ) -> anyhow::Result<()> {
        let mut db_records = Vec::new();
        let mut index_records = Vec::new();
        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                {
                    let embedding_user = EmbeddingFilterInfo {
                        user_uuid: *user_id,
                        embed_id: *embed_id,
                        doc_id: 0,
                    };
                    let db_record = to_embedding_filter_info(&embedding_user)?;
                    db_records.push(db_record);

                    let index_record = IndexRecord::new(*embed_id, embedding.clone());
                    index_records.push(index_record);
                }
            }
        }
        rocksdb.save_records(Arc::new(db_records.clone()))?;
        index.upsert_batch_records(Arc::new(index_records.clone()))?;
        Ok(())
    }

    #[test]
    fn index_add_test() -> anyhow::Result<()> {
        let config = read_config()?;
        let index = HnswIndex::create_index("test_index".to_string(), config.index_config)?;

        let embeddings = generate_random_vectors(2, 384);
        let embedding1 = embeddings.get(0).ok_or(anyhow!("No embeddings"))?;
        let embed_id = thread_rng().gen();
        index.add(embedding1, embed_id)?;
        let embedding2 = embeddings.get(1).ok_or(anyhow!("No embeddings"))?;
        index.upsert(embedding2, embed_id)?;
        Ok(())
    }

    #[test]
    fn config_load() -> anyhow::Result<()> {
        let app_config = read_config()?;
        let index_config = app_config.index_config;
        println!("index_config: {:?}", index_config);
        let index_file = index_file("summaries".to_string(), &index_config);
        let scalar_kind = index_config.scalar_kind;
        assert_eq!(
            index_file,
            format!(
                "index_dir/summaries__L2sq_{:?}_384_16_32_32.usearch",
                scalar_kind
            )
        );
        println!("index_file: {:?}", index_file);
        Ok(())
    }

    #[test]
    #[tracing::instrument]
    fn index_filter_test() -> anyhow::Result<()> {
        setup_tracing(Level::INFO);
        let db_temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = db_temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path)?);

        let mut app_config = read_config()?;
        let index_tmp_dir = tempdir::TempDir::new("test_index")?;
        let index_path = index_tmp_dir.path().to_str().unwrap();
        app_config.index_config.index_dir = index_path.to_string();

        let index =
            HnswIndex::create_index("summaries".to_string(), app_config.index_config.clone())?;

        let num_users = 2;
        let num_user_embeds = 10;

        let records = generate_test_data(num_users, num_user_embeds)?;

        assert_eq!(records.len(), num_users);
        assert_eq!(records[0].1.len(), num_user_embeds);

        add_records(&rocksdb, &index, &records)?;

        check_contained_embeddings(&rocksdb, &index, &records)?;

        let size = index.size()?;
        assert_eq!(size, num_users * num_user_embeds);
        println!("Size: {:?}", size);
        let capacity = index.capacity()?;
        assert!(capacity >= num_users * num_user_embeds);
        println!("Capacity: {:?}", capacity);

        // reinsert records
        add_records(&rocksdb, &index, &records)?;
        let size = index.size()?;
        assert_eq!(size, num_users * num_user_embeds);
        println!("After reinsert Size: {:?}", size);
        let capacity = index.capacity()?;
        assert!(capacity >= num_users * num_user_embeds);
        println!("After reinsert Capacity: {:?}", capacity);

        index.save()?;

        let index2 =
            HnswIndex::create_load_index("summaries".to_string(), app_config.index_config.clone())?;
        let size2 = index2.size()?;
        assert_eq!(size2, num_users * num_user_embeds);
        println!("Old Size: {:?}", size2);
        println!("Old Capacity: {:?}", index2.capacity()?);

        check_contained_embeddings(&rocksdb, &index2, &records)?;

        let records2 = generate_test_data(num_users, num_user_embeds)?;

        add_records(&rocksdb, &index2, &records2)?;
        let size3 = index2.size()?;
        assert_eq!(size3, num_users * num_user_embeds * 2);
        println!("Cur Size: {:?}", size3);
        println!("Cur Capacity: {:?}", index2.capacity()?);
        index2.save()?;

        let index3 =
            HnswIndex::create_load_index("summaries".to_string(), app_config.index_config.clone())?;
        let size4 = index3.size()?;
        assert_eq!(size4, num_users * num_user_embeds * 2);
        println!("Reload Size: {:?}", size4);
        println!("Reload Capacity: {:?}", index3.capacity()?);

        Ok(())
    }

    fn check_contained_embeddings(
        rocksdb: &Arc<RocksDB>,
        index: &HnswIndex,
        records: &Vec<(Uuid, Vec<u64>, Vec<Vec<f32>>)>,
    ) -> anyhow::Result<()> {
        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                let res =
                    index.query_filter(&rocksdb, &vec![user_id.clone()], &vec![], embedding, 4)?;
                println!("embed_id: {}, res: {:?}", embed_id, res);
                assert_eq!(res.len(), 4);
                let first_label = res.keys()[0];
                let first_dist = *res.first().unwrap().1;
                assert_eq!(first_label, *embed_id);
                assert_eq!(first_dist, 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn load_index() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = Arc::new(RocksDB::open(db_path)?);

        let app_config = read_config()?;
        let index = HnswIndex::create_index("summaries".to_string(), app_config.index_config)?;

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

        add_records(&rocksdb, &index, &records)?;

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.into_iter().zip(embeddings.into_iter()) {
                let res = index.query_filter(&rocksdb, &vec![user_id.clone()], &vec![], embedding, 10)?;
                assert_eq!(res.len(), 10);
                let first_label = res.keys()[0];
                let first_dist = *res.first().unwrap().1;
                println!("embed_id: {}, res: {:?}", embed_id, res);
                let orig_embd = index.get_by_label(*embed_id)?;
                // convert to f16 and back again to f32
                let embedding: Vec<f32> = embedding
                    .iter()
                    .map(|f| half::f16::from_f32(*f).to_f32())
                    .collect();
                let sim = f32::l2sq(orig_embd.as_slice(), embedding.as_slice());
                println!("sim: {:?}", sim);
                assert_eq!(orig_embd, *embedding);

                assert_eq!(
                    first_label, *embed_id,
                    "embed_id: {}, res: {:?}",
                    embed_id, res
                );
                assert_eq!(first_dist, 0.0);
            }
        }

        let size = index.size()?;
        assert_eq!(size, num_embeds);
        println!("Size: {:?}", size);

        let capacity = index.capacity()?;
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

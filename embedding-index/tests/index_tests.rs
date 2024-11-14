#[cfg(test)]
mod tests {
    use fake::{Fake, Faker};
    use rand::{thread_rng, Rng, SeedableRng};
    use uuid::Uuid;
    use embedding_database::dao::dao_impl::get_embedding_user;
    use embedding_database::db::column_families::ColumnFamilyType;
    use embedding_database::db::rocksdb_impl::RocksDB;
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::utils::generate_random_vectors;

    const SEED: u64 = 0xc651_4843_1995_363f;


    #[tokio::test(flavor = "multi_thread")]
    async fn index_filter_test() -> anyhow::Result<()> {
        let temp_dir = tempdir::TempDir::new("test_embedding_users")?;
        let db_path = temp_dir.path().to_str().unwrap();
        let rocksdb = RocksDB::open(db_path).await?;

        let index = HnswIndex::new(384)?;

        let rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut user_ids = vec![];
        for _ in 0..4 {
            user_ids.push(Uuid::new_v4());
        }

        let mut embed_ids = vec![0u64; 40];
        for embed_id in embed_ids.iter_mut() {
            *embed_id = thread_rng().gen()
        }

        let embeddings = generate_random_vectors(40, 384);

        let records: Vec<_> = user_ids.iter().enumerate().map(|(i, user_id)| {
            let embed_ids = embed_ids.into_iter().skip(i * 10).take(10).collect::<Vec<_>>();
            let embeddings = embeddings.into_iter().skip(i * 10).take(10).collect::<Vec<_>>();
            (user_id, embed_ids, embeddings)
        }).collect();

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                rocksdb.put(ColumnFamilyType::EmbeddingUsers, embed_id, user_id.as_bytes()).await?;
                tokio::task::block_in_place(|| {
                    index.add(embedding, *embed_id).unwrap();
                });
            }
        }


        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_database::db::column_families::ColumnFamilyType;
    use embedding_database::db::rocksdb_impl::RocksDB;
    use embedding_index::hnsw_index::HnswIndex;
    use embedding_index::utils::generate_random_vectors;
    use rand::{thread_rng, Rng};
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
        let rocksdb = RocksDB::open(db_path).await?;

        let index = HnswIndex::new(384)?;

        let mut user_ids = vec![];
        for _ in 0..4 {
            user_ids.push(Uuid::new_v4());
        }

        let mut embed_ids = vec![0u64; 40];
        for embed_id in embed_ids.iter_mut() {
            *embed_id = thread_rng().gen()
        }

        let embeddings = generate_random_vectors(40, 384);

        let records: Vec<_> = user_ids
            .iter()
            .enumerate()
            .map(|(i, user_id)| {
                let embed_ids = <Vec<u64> as Clone>::clone(&embed_ids)
                    .into_iter()
                    .skip(i * 10)
                    .take(10)
                    .collect::<Vec<_>>();
                let embeddings = <Vec<Vec<f32>> as Clone>::clone(&embeddings)
                    .into_iter()
                    .skip(i * 10)
                    .take(10)
                    .collect::<Vec<_>>();
                (user_id, embed_ids, embeddings)
            })
            .collect();

        for (user_id, embed_ids, embeddings) in records.iter() {
            for (embed_id, embedding) in embed_ids.iter().zip(embeddings.iter()) {
                rocksdb
                    .put(
                        ColumnFamilyType::EmbeddingUsers,
                        embed_id,
                        user_id.as_bytes(),
                    )
                    .await?;
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::prelude::*;
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[test]
    fn embedding_user_serde() -> anyhow::Result<()> {
        let user_uuid = Uuid::new_v4();
        let doc_id: u64 = rand::random();
        let embed_id: u64 = rand::random();
        let embedding_user = EmbeddingFilterInfo {
            embed_id,
            doc_id,
            user_uuid,
        };
        println!("embedding_user: {:?}", embedding_user);

        let eu_packed = embedding_user.pack()?;
        let eu_unpacked = EmbeddingFilterInfo::unpack(&eu_packed)?;

        assert_eq!(embedding_user, eu_unpacked);
        assert_eq!(embedding_user.embed_id, eu_unpacked.embed_id);
        assert_eq!(embedding_user.doc_id, eu_unpacked.doc_id);
        assert_eq!(embedding_user.user_uuid, eu_unpacked.user_uuid);
        println!("eu_unpacked: {:?}", eu_unpacked);
        Ok(())
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
    pub struct ModelPrevVersion {
        pub model_id: u64,
        pub path: String,
        pub n_ctx: i32,
        pub n_embd: i32,
    }
    impl ModelPrevVersion {
        pub fn new(model_id: u64, path: String, n_ctx: i32, n_embd: i32) -> Self {
            Self {
                model_id,
                path,
                n_ctx,
                n_embd,
            }
        }
    }

    impl Serde for ModelPrevVersion {}

    #[test]
    #[should_panic]
    fn model_backward_compatibility() {
        let model_prev = ModelPrevVersion::new(1, "path/to/model".to_string(), 1024, 512);
        let prev_packed = model_prev.pack().expect("Failed to pack");
        let prev_unpacked = ModelPrevVersion::unpack(&prev_packed).expect("Failed to unpack");
        assert_eq!(model_prev, prev_unpacked);
        Model::unpack(&prev_packed).expect("Failed to unpack")
    }

    #[test]
    fn msq_decode_value() -> anyhow::Result<()> {
        let user_uuid = Uuid::new_v4();
        let doc_id: u64 = rand::random();
        let embed_id: u64 = rand::random();

        let embedding_filter = EmbeddingFilterInfo {
            embed_id,
            doc_id,
            user_uuid,
        };
        println!("embedding_user: {:?}", embedding_filter);

        let eu_packed = embedding_filter.pack()?;
        let map_value = rmpv::decode::read_value(&mut eu_packed.as_slice())?;
        println!("map_value: {:?}", map_value);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn embedding_user_serde_fail() {
        let user_uuid = Uuid::new_v4();
        let doc_id: u64 = rand::random();
        let embed_id: u64 = rand::random();

        let embedding_user = EmbeddingFilterInfo {
            embed_id,
            doc_id,
            user_uuid,
        };
        println!("embedding_user: {:?}", embedding_user);

        let eu_packed = embedding_user.pack().unwrap();
        EmbeddingFilterInfo::unpack::<EmbeddingFilterInfo>(&eu_packed).unwrap();

        let uuid_bytes = user_uuid.as_bytes();
        EmbeddingFilterInfo::unpack::<EmbeddingFilterInfo>(&uuid_bytes.as_slice())
            .map_err(|e| anyhow!("Failed to upack e: {e}"))
            .expect("Failed to unpack");
    }
}

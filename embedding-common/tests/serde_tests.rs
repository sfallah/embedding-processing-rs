#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use embedding_common::models::embedding::EmbeddingUser;
    use embedding_common::types::Serde;
    use rand::Rng;
    use uuid::Uuid;
    #[test]
    fn embedding_user_serde() -> anyhow::Result<()> {
        let user_uuid = Uuid::new_v4();
        let embed_id: u64 = rand::thread_rng().gen();
        let embedding_user = EmbeddingUser {
            embed_id,
            user_uuid,
        };
        println!("embedding_user: {:?}", embedding_user);

        let eu_packed = embedding_user.pack()?;
        let eu_unpacked = EmbeddingUser::unpack(&eu_packed)?;

        assert_eq!(embedding_user, eu_unpacked);
        assert_eq!(embedding_user.embed_id, eu_unpacked.embed_id);
        assert_eq!(embedding_user.user_uuid, eu_unpacked.user_uuid);
        println!("eu_unpacked: {:?}", eu_unpacked);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn embedding_user_serde_fail() {
        let user_uuid = Uuid::new_v4();
        let embed_id: u64 = rand::thread_rng().gen();
        let embedding_user = EmbeddingUser {
            embed_id,
            user_uuid,
        };
        println!("embedding_user: {:?}", embedding_user);

        let eu_packed = embedding_user.pack().unwrap();
        EmbeddingUser::unpack::<EmbeddingUser>(&eu_packed).unwrap();

        let uuid_bytes = user_uuid.as_bytes();
        EmbeddingUser::unpack::<EmbeddingUser>(&uuid_bytes.as_slice())
            .map_err(|e| anyhow!("Failed to upack e: {e}"))
            .expect("Failed to unpack");
    }
}

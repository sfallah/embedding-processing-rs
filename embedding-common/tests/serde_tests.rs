#[cfg(test)]
mod tests {
    use uuid::Uuid;
    use embedding_common::types::Serde;
    use embedding_common::models::embedding::EmbeddingUser;
    use rand::Rng;
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
}
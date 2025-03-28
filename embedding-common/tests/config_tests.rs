#[cfg(test)]
mod tests {
    use embedding_common::config::config_file::ConfigFromFile;
    use embedding_common::config::{AppConfig, MetricKind, ScalarKind};

    #[test]
    fn test_index_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let index_config = app_config.index_config;
        println!("index config: {:?}", index_config);
        assert_eq!(index_config.index_dir, "test_index_dir".to_string());
        assert_eq!(index_config.metric_kind, MetricKind::Cos);
        assert_eq!(index_config.scalar_kind, ScalarKind::F32);
        assert_eq!(index_config.dimensions, 384);
        assert_eq!(index_config.connectivity, 0);
        assert_eq!(index_config.expansion_add, 0);
        assert_eq!(index_config.expansion_search, 32);
        Ok(())
    }

    #[test]
    fn test_splitter_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let splitter_config = app_config.splitter_config;
        println!("index config: {:?}", splitter_config);
        assert_eq!(splitter_config.max_tokens, 512);
        assert!(splitter_config.hf_model_id.is_some());
        assert_eq!(
            splitter_config.hf_model_id.unwrap(),
            "sentence-transformers/all-MiniLM-L6-v2".to_string()
        );
        assert!(splitter_config.patterns.is_some());
        assert_eq!(splitter_config.patterns.clone().unwrap().len(), 3);
        assert_eq!(splitter_config.patterns.clone().unwrap()[0].len(), 1);
        assert_eq!(splitter_config.patterns.clone().unwrap()[1].len(), 1);
        assert_eq!(splitter_config.patterns.clone().unwrap()[2].len(), 3);

        let text = "This is a test sentence.\n\n This is another test sentence.".to_string();
        let text_splits: Vec<_> = text
            .split(&splitter_config.patterns.clone().unwrap()[0][0])
            .collect();
        assert_eq!(text_splits.len(), 2);
        assert_eq!(text_splits[0], "This is a test sentence.");
        assert_eq!(text_splits[1], " This is another test sentence.");
        Ok(())
    }

    #[test]
    fn test_database_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let database_config = app_config.database_config;
        println!("database config: {:?}", database_config);
        assert_eq!(database_config.rocksdb_dir, "test_rocksdb_dir".to_string());
        Ok(())
    }

    #[test]
    fn test_zmq_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let zmq_config = app_config.zmq_config;
        println!("zmq_server config: {:?}", zmq_config);
        assert_eq!(zmq_config.zmq_host, "127.0.0.1".to_string());
        assert_eq!(zmq_config.zmq_frontend_port, 5556);
        assert_eq!(zmq_config.zmq_backend_port, 5560);
        assert_eq!(zmq_config.zmq_num_workers, 2);
        Ok(())
    }
    #[test]
    fn test_endpoints_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let endpoints_config = app_config.endpoints;
        println!("endpoints config: {:?}", endpoints_config);
        assert_eq!(
            endpoints_config.embedding_endpoint,
            "tcp://localhost:5559".to_string()
        );
        assert!(endpoints_config.reranking_endpoint.is_some());
        assert_eq!(
            endpoints_config.reranking_endpoint.unwrap(),
            "tcp://localhost:5557".to_string()
        );
        Ok(())
    }
}

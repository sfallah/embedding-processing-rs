#[cfg(test)]
mod tests {
    use embedding_common::config::config_file::ConfigFromFile;
    use embedding_common::config::{AppConfig, MetricKind, ScalarKind, StorageConfig};

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
    fn test_storage_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let storage_config = app_config.storage_config;
        println!("storage config: {:?}", storage_config);
        assert_eq!(storage_config.dir, "test_storage_dir".to_string());
        assert_eq!(storage_config.memory_budget_mb, 512);
        assert_eq!(storage_config.fsync_interval_ms, 250);
        assert_eq!(storage_config.snapshot_interval_s, 60);
        assert_eq!(storage_config.segment_max_mb, 8);
        assert_eq!(storage_config.compact_ratio, 0.5);
        storage_config.validate()?;

        // The units the store and the pool are built from, rather than the ones an operator writes.
        assert_eq!(storage_config.memory_budget_bytes(), 512 * 1024 * 1024);
        assert_eq!(storage_config.segment_max_bytes(), 8 * 1024 * 1024);
        assert_eq!(
            storage_config.fsync_interval(),
            Some(std::time::Duration::from_millis(250))
        );
        assert_eq!(
            storage_config.snapshot_interval(),
            std::time::Duration::from_secs(60)
        );
        assert!(storage_config.storage_dir()?.ends_with("test_storage_dir"));
        Ok(())
    }

    #[test]
    fn test_storage_config_defaults_and_validation() {
        // An omitted key falls back to the plan's default rather than to zero.
        let defaults = StorageConfig::default();
        assert_eq!(defaults.dir, "storage");
        assert_eq!(defaults.fsync_interval_ms, 1000); // a one-second fsync timer by default
        assert_eq!(defaults.segment_max_mb, 256); // the store's own default
        defaults.validate().expect("the defaults are usable");

        // A zero fsync interval is the per-request policy: sync every write, not spin on a timer.
        let per_request = StorageConfig {
            fsync_interval_ms: 0,
            ..StorageConfig::default()
        };
        assert_eq!(per_request.fsync_interval(), None);
        per_request
            .validate()
            .expect("per-request fsync is allowed");

        // Settings that would misbehave rather than merely perform badly are refused.
        for bad in [
            StorageConfig {
                memory_budget_mb: 0,
                ..StorageConfig::default()
            },
            StorageConfig {
                segment_max_mb: 0,
                ..StorageConfig::default()
            },
            StorageConfig {
                snapshot_interval_s: 0,
                ..StorageConfig::default()
            },
            StorageConfig {
                compact_ratio: 0.0,
                ..StorageConfig::default()
            },
            StorageConfig {
                compact_ratio: 1.5,
                ..StorageConfig::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{:?} should be refused", bad);
        }
    }

    #[test]
    fn test_config() -> anyhow::Result<()> {
        let conf_file = "tests/test_config.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let config = app_config.zmq_config;
        println!("server config: {:?}", config);
        assert_eq!(config.host, "127.0.0.1".to_string());
        assert_eq!(config.frontend_port, 5556);
        assert_eq!(config.backend_port, 5560);
        assert_eq!(config.num_workers, 2);
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

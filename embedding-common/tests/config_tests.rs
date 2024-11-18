#[cfg(test)]
mod tests{
    use embedding_common::config::{AppConfig, MetricKind, ScalarKind};

    #[test]
    fn test_index_config()-> anyhow::Result<()>{
        let conf_file = "tests/test_config/index_config_default.toml".to_string();
        let app_config = AppConfig::from_file(conf_file)?;
        let index_config = app_config.index_config;
        println!("index_config: {:?}", index_config);
        assert_eq!(index_config.index_dir, "test_index_dir".to_string());
        assert_eq!(index_config.metric_kind, MetricKind::Cos);
        assert_eq!(index_config.scalar_kind, ScalarKind::F32);
        assert_eq!(index_config.dimensions, 384);
        assert_eq!(index_config.connectivity, 0);
        assert_eq!(index_config.expansion_add, 0);
        assert_eq!(index_config.expansion_search, 32);
        Ok(())
    }
}
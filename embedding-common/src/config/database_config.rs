use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    #[serde(default = "default_rocksdb_dir")]
    pub rocksdb_dir: String,
}

fn default_rocksdb_dir() -> String {
    "rocksdb_dir".to_string()
}

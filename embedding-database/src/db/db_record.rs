use crate::db::column_families::ColumnFamilyType;

#[derive(Debug, Clone)]
pub struct DbRecordValue {
    pub cf_type: ColumnFamilyType,
    pub key: u64,
    pub value: Vec<u8>,
}

impl DbRecordValue {
    pub fn new(cf_type: ColumnFamilyType, key: u64, value: Vec<u8>) -> Self {
        Self {
            cf_type,
            key,
            value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DbRecordKey {
    pub cf_type: ColumnFamilyType,
    pub key: u64,
}

impl DbRecordKey {
    pub fn new(cf_type: ColumnFamilyType, key: u64) -> Self {
        Self { cf_type, key }
    }
}

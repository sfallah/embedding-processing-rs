use crate::db::column_families::ColumnFamilyType;

pub(crate) struct DbRecordValue {
    pub(crate) cf_type: ColumnFamilyType,
    pub(crate) key: u64,
    pub(crate) value: Vec<u8>,
}

impl DbRecordValue {
    pub(crate) fn new(cf_type: ColumnFamilyType, key: u64, value: Vec<u8>) -> Self {
        Self {
            cf_type,
            key,
            value,
        }
    }
}

pub(crate) struct DbRecordKey {
    pub(crate) cf_type: ColumnFamilyType,
    pub(crate) key: u64,
}

impl DbRecordKey {
    pub(crate) fn new(cf_type: ColumnFamilyType, key: u64) -> Self {
        Self {
            cf_type,
            key,
        }
    }
}
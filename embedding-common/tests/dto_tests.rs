#[cfg(test)]
mod tests {
    use embedding_common::prelude::*;
    use indexmap::IndexSet;

    #[test]
    fn test_split_dto_set() {
        let mut index_set = IndexSet::new();
        let split1 = SplitDto::new(1, 1, 1, "test", 1, vec![], None, None);
        let split2 = SplitDto::new(2, 2, 2, "test", 1, vec![], None, None);
        index_set.insert(split1);
        index_set.insert(split2);
        assert_eq!(index_set.len(), 2);
    }
}

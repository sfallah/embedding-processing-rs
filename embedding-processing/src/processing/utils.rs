use fast_text_splitter::splitter::split_node::utils::SplitResultLite;

pub fn splits_texts(splits: &Vec<SplitResultLite>) -> Vec<String> {
    splits
        .iter()
        .map(|sentence_split| sentence_split.split_string.clone())
        .collect()
}
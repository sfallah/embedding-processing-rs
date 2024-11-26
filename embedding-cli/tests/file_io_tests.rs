#[cfg(test)]
mod tests {
    use embedding_cli::file_io::list_files;
    use std::path::Path;
    #[tokio::test(flavor = "multi_thread")]
    async fn test_list_txt_files() -> anyhow::Result<()> {
        let dir_path = Path::new("tests/test_data");
        let extensions = vec!["txt".to_string()];
        let res = list_files(dir_path, &extensions).await?;
        assert_eq!(res.len(), 10);
        for file in res.iter() {
            println!("{}", file.display());
        }
        Ok(())
    }
}

use std::path::{Path, PathBuf};
use tokio::fs;

pub async fn list_files(
    dir_path: &Path,
    file_extensions: &Vec<String>,
) -> anyhow::Result<Vec<PathBuf>> {
    if !dir_path.is_dir() {
        if dir_path.is_file() {
            return Ok(vec![dir_path.to_path_buf()]);
        } else {
            return Err(anyhow::anyhow!("{} is not a directory", dir_path.display()));
        }
    }
    let mut entries = fs::read_dir(dir_path).await?;
    let mut file_paths: Vec<PathBuf> = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        match path.extension() {
            Some(ext) => {
                let ext = ext.to_str().unwrap().to_string();
                if file_extensions.contains(&ext) {
                    file_paths.push(path);
                }
            }
            None => continue,
        }
    }
    Ok(file_paths)
}

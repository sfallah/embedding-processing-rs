use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Creates a directory at the specified path if it does not already exist.
///
/// # Arguments
///
/// * `path` - A string slice that holds the path of the directory to create.
///
/// # Returns
///
/// This function returns `Result<()>`, which will be `Ok` if the directory was created successfully
/// or if it already exists, and `Err` if there was an error creating the directory.
pub fn create_directory(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory at {:?}", path))?;
        println!("Directory created at {:?}", path);
    } else {
        println!("Directory already exists at {:?}", path);
    }
    Ok(())
}

/// Returns the current working directory as a `PathBuf`.
///
/// # Returns
///
/// This function returns `Result<PathBuf>`, which will be `Ok` containing the path if the current
/// working directory could be retrieved successfully, and `Err` if there was an error.
fn cur_dir() -> Result<PathBuf, std::io::Error> {
    env::current_dir()
}

/// Returns a directory under the current working directory as a `PathBuf`.
///
/// # Arguments
///
/// * `sub_dir` - A string slice that holds the name of the subdirectory to get.
///
/// # Returns
///
/// This function returns `PathBuf` that points to the specified subdirectory under the current
/// working directory.
pub fn get_directory(sub_dir: &str) -> Result<PathBuf> {
    let mut path = cur_dir().with_context(|| "Failed to get current directory")?;
    path.push(sub_dir);
    Ok(path)
}

/// Returns the default directory for RocksDB as a `PathBuf`.
///
/// # Returns
///
/// This function returns `Result<PathBuf>` that points to the default RocksDB directory under the current
/// working directory.
pub fn get_db_dir(directory: Option<&str>) -> Result<PathBuf> {
    match directory {
        Some(dir) if !dir.is_empty() => get_directory(dir),
        _ => get_directory("rocks_db_dir"),
    }
}

/// Returns the default directory for models as a `PathBuf`.
///
/// # Returns
///
/// This function returns `Result<PathBuf>` that points to the default models directory under the current
/// working directory.
pub fn get_default_models_dir() -> Result<PathBuf> {
    get_directory("models")
}
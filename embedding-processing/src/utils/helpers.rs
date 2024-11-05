use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
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
/// * `sub_dir` - A string slice that holds the name of the sub-directory to get.
///
/// # Returns
///
/// This function returns `PathBuf` that points to the specified sub-directory under the current
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

enum UnitOfTime {
    Nanos,
    Micros,
    Millis,
    Secs,
}

pub fn time_nanos(start: &Instant, msg: &str) -> Instant {
    trace_time_aux(start, msg, UnitOfTime::Nanos)
}
pub fn time_micros(start: &Instant, msg: &str) -> Instant {
    trace_time_aux(start, msg, UnitOfTime::Micros)
}
pub fn time_millis(start: &Instant, msg: &str) -> Instant {
    trace_time_aux(start, msg, UnitOfTime::Millis)
}
pub fn time_secs(start: &Instant, msg: &str) -> Instant {
    trace_time_aux(start, msg, UnitOfTime::Secs)
}

fn trace_time_aux(start: &Instant, msg: &str, uot: UnitOfTime) -> Instant {
    let elapsed = start.elapsed();
    match uot {
        UnitOfTime::Nanos => {
            let nanos = elapsed.as_nanos();
            println!("{}: {} ns", msg, nanos);
        }
        UnitOfTime::Micros => {
            let micros = elapsed.as_micros();
            println!("{}: {} µs", msg, micros);
        }
        UnitOfTime::Millis => {
            let millis = elapsed.as_micros();
            let millis = millis as f64 / 1000.0;
            println!("{}: {} ms", msg, millis);
        }
        UnitOfTime::Secs => {
            let secs = elapsed.as_secs_f64();
            println!("{}: {} s", msg, secs);
        }
    };
    Instant::now()
}

pub fn from_epoch_micros(epoch_millis: u64) -> Instant {
    // Convert the epoch millis to a `DateTime<Utc>`
    let req_ts = Utc.timestamp_nanos(epoch_millis as i64);
    println!("Request timestamp: {:?}", req_ts);

    // Get the current time as `DateTime<Utc>`
    let now = Utc::now();

    // Calculate the difference between now and the provided time
    let duration_since_epoch = now.signed_duration_since(req_ts);
    println!("Duration since epoch: {:?}", duration_since_epoch);

    // Subtract the duration from `Instant::now()` to get the corresponding `Instant`
    Instant::now() - Duration::from_nanos(duration_since_epoch.num_nanoseconds().unwrap() as u64)
}

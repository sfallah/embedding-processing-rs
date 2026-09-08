//! The usearch side of a shard: one index per searchable entity kind.
//!
//! Both indexes live in the shard's own directory as `splits.usearch` and `summaries.usearch`.
//! They are built with `Index::new` and then `load`ed — never `Index::restore`, which resets
//! `expansion_add` and `expansion_search` to the library defaults (128 and 64) instead of the
//! configured values (verified on usearch 2.26.2).

use crate::record::{decode_vector, VectorDtype};
use anyhow::{anyhow, Context, Result};
use embedding_common::config::IndexConfig;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use usearch::{f16, Index, IndexOptions, MetricKind, ScalarKind};

/// The usearch options a shard's indexes are built with. Every shard of one deployment uses the
/// same ones, so an index file only ever meets the parameters it was written with.
pub fn index_options(config: &IndexConfig) -> IndexOptions {
    IndexOptions {
        dimensions: config.dimensions,
        metric: metric_kind(config.metric_kind),
        quantization: scalar_kind(config.scalar_kind),
        connectivity: config.connectivity,
        expansion_add: config.expansion_add,
        expansion_search: config.expansion_search,
        multi: false,
    }
}

fn metric_kind(kind: embedding_common::config::MetricKind) -> MetricKind {
    match kind {
        embedding_common::config::MetricKind::IP => MetricKind::IP,
        embedding_common::config::MetricKind::L2sq => MetricKind::L2sq,
        embedding_common::config::MetricKind::Cos => MetricKind::Cos,
    }
}

fn scalar_kind(kind: embedding_common::config::ScalarKind) -> ScalarKind {
    match kind {
        embedding_common::config::ScalarKind::F64 => ScalarKind::F64,
        embedding_common::config::ScalarKind::F32 => ScalarKind::F32,
        embedding_common::config::ScalarKind::F16 => ScalarKind::F16,
        embedding_common::config::ScalarKind::BF16 => ScalarKind::BF16,
        embedding_common::config::ScalarKind::I8 => ScalarKind::I8,
        embedding_common::config::ScalarKind::B1 => ScalarKind::B1,
    }
}

/// One HNSW index, its file, and the parameters it was created with.
pub struct VectorIndex {
    index: Index,
    /// "splits" or "summaries"; only used in messages.
    name: &'static str,
    path: PathBuf,
    metric: MetricKind,
    dimensions: usize,
    /// True while the graph is mapped from the file rather than held in memory. usearch refuses
    /// `add` on such an index, so a writer has to `load` it first.
    viewed: AtomicBool,
}

impl VectorIndex {
    /// An empty index. Nothing is read from disk; the caller decides between `load` and a rebuild.
    pub fn create(dir: &Path, name: &'static str, config: &IndexConfig) -> Result<Self> {
        let options = index_options(config);
        let index = Index::new(&options)
            .map_err(|e| anyhow!("could not create the {} index: {}", name, e))?;
        Ok(VectorIndex {
            index,
            name,
            path: dir.join(format!("{}.usearch", name)),
            metric: options.metric,
            dimensions: options.dimensions,
            viewed: AtomicBool::new(false),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn exists(&self) -> bool {
        self.path.is_file()
    }

    pub fn size(&self) -> usize {
        self.index.size()
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn metric(&self) -> MetricKind {
        self.metric
    }

    // Read off the live index rather than the options it was built from, so a load that quietly
    // took the library's defaults shows up here. `Index::restore` does exactly that on 2.26.2.
    pub fn connectivity(&self) -> usize {
        self.index.connectivity()
    }

    pub fn expansion_add(&self) -> usize {
        self.index.expansion_add()
    }

    pub fn expansion_search(&self) -> usize {
        self.index.expansion_search()
    }

    /// Read the saved index back. The graph, not the vectors alone, so the result is mutable.
    pub fn load(&self) -> Result<()> {
        let path = self.path_str()?;
        self.index
            .load(path)
            .map_err(|e| anyhow!("could not load {}: {}", self.path.display(), e))?;
        self.viewed.store(false, Ordering::Release);
        Ok(())
    }

    /// Map the saved index instead of reading it in. Searches answer from the mapping, so a shard
    /// nobody is writing to costs its file's page cache rather than its graph: about 43 KB
    /// resident for an 8 MB index, against the megabytes `load` would hold.
    ///
    /// The index is immutable while viewed; `load` promotes it back.
    pub fn view(&self) -> Result<()> {
        let path = self.path_str()?;
        self.index
            .view(path)
            .map_err(|e| anyhow!("could not view {}: {}", self.path.display(), e))?;
        self.viewed.store(true, Ordering::Release);
        Ok(())
    }

    /// Whether the graph is mapped rather than resident. A viewed index refuses `add`.
    pub fn is_viewed(&self) -> bool {
        self.viewed.load(Ordering::Acquire)
    }

    fn path_str(&self) -> Result<&str> {
        self.path
            .to_str()
            .ok_or_else(|| anyhow!("{} is not valid utf-8", self.path.display()))
    }

    /// Room for `additional` more vectors. usearch grows in place, but only when asked.
    pub fn reserve_for(&self, additional: usize) -> Result<()> {
        let wanted = self.index.size() + additional.max(64);
        if self.index.capacity() > wanted {
            return Ok(());
        }
        self.index.reserve(wanted).map_err(|e| {
            anyhow!(
                "could not reserve {} slots in the {} index: {}",
                wanted,
                self.name,
                e
            )
        })
    }

    /// Add a vector held as `f32`, replacing whatever the key held before.
    ///
    /// `multi` is false, so a key means one vector; usearch is asked to drop the old one first
    /// rather than being left to decide.
    pub fn upsert(&self, key: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(anyhow!(
                "{} index takes {} dimensions, got {} for key {}",
                self.name,
                self.dimensions,
                vector.len(),
                key
            ));
        }
        self.remove(key)?;
        self.reserve_for(1)?;
        self.index
            .add(key, vector)
            .map_err(|e| anyhow!("could not add {} to the {} index: {}", key, self.name, e))
    }

    /// Add a vector as it is stored in the log, without going through `f32` when the log and the
    /// index agree on `f16`. This is the rebuild path, where the conversion would be pure waste.
    pub fn add_raw(&self, key: u64, bytes: &[u8], dtype: VectorDtype) -> Result<()> {
        let expected = dtype.vector_bytes(self.dimensions);
        if bytes.len() != expected {
            return Err(anyhow!(
                "{} index takes {} bytes per vector, got {} for key {}",
                self.name,
                expected,
                bytes.len(),
                key
            ));
        }
        let result = match dtype {
            VectorDtype::F16 => {
                // The mmap gives no alignment guarantee, so the halves are copied into an aligned
                // buffer rather than reinterpreted in place. It is a memcpy, not a conversion.
                let halves: Vec<i16> = bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                self.index.add(key, f16::from_i16s(&halves))
            }
            VectorDtype::F32 => self.index.add(key, &decode_vector(bytes, dtype)),
            VectorDtype::None => {
                return Err(anyhow!("key {} has no vector to index", key));
            }
        };
        result.map_err(|e| anyhow!("could not add {} to the {} index: {}", key, self.name, e))
    }

    /// Drop a key. Removing one that is not there is not an error, so no lookup precedes it.
    pub fn remove(&self, key: u64) -> Result<()> {
        self.index.remove(key).map(|_| ()).map_err(|e| {
            anyhow!(
                "could not remove {} from the {} index: {}",
                key,
                self.name,
                e
            )
        })
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        self.check_query(query)?;
        let matches = self
            .index
            .search(query, k)
            .map_err(|e| anyhow!("could not search the {} index: {}", self.name, e))?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    pub fn filtered_search<F>(&self, query: &[f32], k: usize, filter: F) -> Result<Vec<(u64, f32)>>
    where
        F: Fn(u64) -> bool,
    {
        self.check_query(query)?;
        let matches = self
            .index
            .filtered_search(query, k, filter)
            .map_err(|e| anyhow!("could not search the {} index: {}", self.name, e))?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    fn check_query(&self, query: &[f32]) -> Result<()> {
        if query.len() != self.dimensions {
            return Err(anyhow!(
                "{} index takes {} dimensions, the query has {}",
                self.name,
                self.dimensions,
                query.len()
            ));
        }
        Ok(())
    }

    /// Write the index out, temp file and rename, so a crash leaves the previous one intact.
    pub fn save_atomically(&self) -> Result<()> {
        let tmp = self.path.with_extension("usearch.tmp");
        let tmp_str = tmp
            .to_str()
            .ok_or_else(|| anyhow!("{} is not valid utf-8", tmp.display()))?;
        self.index
            .save(tmp_str)
            .map_err(|e| anyhow!("could not save the {} index: {}", self.name, e))?;
        // usearch writes through its own file handle, so the bytes have to be pushed to the
        // device here rather than relying on the rename to carry them.
        fs::File::open(&tmp)
            .with_context(|| format!("reopening {}", tmp.display()))?
            .sync_all()
            .with_context(|| format!("syncing {}", tmp.display()))?;
        fs::rename(&tmp, &self.path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), self.path.display()))?;
        Ok(())
    }

    pub fn remove_file(&self) -> Result<()> {
        if self.path.exists() {
            fs::remove_file(&self.path)
                .with_context(|| format!("removing {}", self.path.display()))?;
        }
        Ok(())
    }

    /// Resident bytes: what the graph and the vectors actually hold, not what their allocators
    /// have reserved. `memory_usage()` counts the reservation and over-reports a small index by
    /// about a factor of two.
    pub fn memory_bytes(&self) -> usize {
        let stats = self.index.memory_stats();
        stats.graph_allocated.saturating_sub(stats.graph_reserved)
            + stats
                .vectors_allocated
                .saturating_sub(stats.vectors_reserved)
    }
}

/// The distance usearch would report between two vectors under `metric`.
///
/// Used by the brute-force path for a small document filter, where walking the graph costs more
/// than scoring the candidates directly. It has to agree with the index: a mode that mixes the
/// two would order hits by two different numbers.
pub fn distance(metric: MetricKind, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        MetricKind::L2sq => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>(),
        MetricKind::IP => 1.0 - dot(a, b),
        MetricKind::Cos => {
            let denom = (norm_squared(a) * norm_squared(b)).sqrt();
            if denom <= f32::EPSILON {
                1.0
            } else {
                1.0 - dot(a, b) / denom
            }
        }
        // The config exposes only the three above; anything else would be a silent mis-ordering,
        // so it is reported as the largest possible distance rather than guessed at.
        _ => f32::INFINITY,
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_squared(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

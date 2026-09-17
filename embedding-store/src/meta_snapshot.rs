//! `meta.snap`: the three maps in a flat file, so a restart is a snapshot load plus a short tail
//! replay instead of a full scan of every segment.
//!
//! Without it the maps could only be rebuilt from the log, which means reading every record —
//! including every embedding — before the shard can answer anything. At a million splits that is
//! on the order of ten gigabytes of reads per restart.

use crate::maps::{DocEntry, Loc, Maps, SplitEntry, SummaryEntry};
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub const META_SNAPSHOT_FILE: &str = "meta.snap";
const MAGIC: u32 = 0x4D45_5441; // "META"
const VERSION: u32 = 1;

pub fn path(dir: &Path) -> PathBuf {
    dir.join(META_SNAPSHOT_FILE)
}

// ---------------------------------------------------------------------------
// Little-endian writer and reader
// ---------------------------------------------------------------------------

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn loc(&mut self, loc: &Loc) {
        self.u32(loc.segment);
        self.u64(loc.offset);
        self.u32(loc.len);
    }
    fn ids(&mut self, ids: &[u64]) {
        self.u32(ids.len() as u32);
        for id in ids {
            self.u64(*id);
        }
    }
    fn ids_opt(&mut self, ids: &Option<Vec<u64>>) {
        match ids {
            None => self.u32(u32::MAX),
            Some(ids) => self.ids(ids),
        }
    }
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(anyhow!(
                "meta snapshot is truncated: wanted {} bytes at {}, file is {}",
                n,
                self.pos,
                self.buf.len()
            ));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn loc(&mut self) -> Result<Loc> {
        Ok(Loc {
            segment: self.u32()?,
            offset: self.u64()?,
            len: self.u32()?,
        })
    }
    fn ids(&mut self) -> Result<Vec<u64>> {
        let n = self.u32()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.u64()?);
        }
        Ok(out)
    }
    fn ids_opt(&mut self) -> Result<Option<Vec<u64>>> {
        let n = self.u32()?;
        if n == u32::MAX {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(self.u64()?);
        }
        Ok(Some(out))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
}

// ---------------------------------------------------------------------------

/// Write the maps, temp-and-rename so a crash never leaves a half-written snapshot in place.
pub fn store(dir: &Path, maps: &Maps, snapshot_seq: u64) -> Result<()> {
    let mut w = Writer::new();
    w.u32(MAGIC);
    w.u32(VERSION);
    w.u64(snapshot_seq);
    w.u64(maps.docs.len() as u64);
    w.u64(maps.splits.len() as u64);
    w.u64(maps.summaries.len() as u64);

    for (doc_id, entry) in &maps.docs {
        w.u64(*doc_id);
        w.u64(entry.seq);
        w.loc(&entry.loc);
        w.string(&entry.url);
        w.ids(&entry.split_ids);
        w.ids_opt(&entry.summary_ids);
        w.ids(&entry.extra_summary_ids);
    }
    for (split_id, entry) in &maps.splits {
        w.u64(*split_id);
        w.u64(entry.doc_id);
        w.loc(&entry.loc);
    }
    for (summary_id, entry) in &maps.summaries {
        w.u64(*summary_id);
        w.u64(entry.doc_id);
        w.u64(entry.split_id);
        w.loc(&entry.loc);
    }

    let crc = crc32fast::hash(&w.buf);
    w.u32(crc);

    let final_path = path(dir);
    let tmp = dir.join(format!("{}.tmp", META_SNAPSHOT_FILE));
    fs::write(&tmp, &w.buf).with_context(|| format!("writing {}", tmp.display()))?;
    {
        let f = fs::File::open(&tmp)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &final_path).with_context(|| format!("renaming {}", tmp.display()))?;
    crate::fsync_dir(dir)?;
    Ok(())
}

/// Load the maps. Returns `None` when there is no snapshot; a snapshot that fails its checks is
/// an error the caller may choose to treat as "no snapshot" and replay the whole log instead.
pub fn load(dir: &Path) -> Result<Option<(Maps, u64)>> {
    let file = path(dir);
    if !file.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&file).with_context(|| format!("reading {}", file.display()))?;
    if bytes.len() < 4 {
        return Err(anyhow!("meta snapshot {} is too short", file.display()));
    }
    let body = &bytes[..bytes.len() - 4];
    let expected = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    let found = crc32fast::hash(body);
    if expected != found {
        return Err(anyhow!(
            "meta snapshot {} failed its checksum ({:#x} != {:#x})",
            file.display(),
            expected,
            found
        ));
    }

    let mut r = Reader::new(body);
    if r.u32()? != MAGIC {
        return Err(anyhow!("{} is not a meta snapshot", file.display()));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(anyhow!(
            "meta snapshot {} has version {}, expected {}",
            file.display(),
            version,
            VERSION
        ));
    }
    let snapshot_seq = r.u64()?;
    let n_docs = r.u64()? as usize;
    let n_splits = r.u64()? as usize;
    let n_summaries = r.u64()? as usize;

    let mut maps = Maps::default();
    maps.docs.reserve(n_docs);
    maps.splits.reserve(n_splits);
    maps.summaries.reserve(n_summaries);

    for _ in 0..n_docs {
        let doc_id = r.u64()?;
        let seq = r.u64()?;
        let loc = r.loc()?;
        let url = r.string()?;
        let split_ids = r.ids()?;
        let summary_ids = r.ids_opt()?;
        let extra_summary_ids = r.ids()?;
        maps.docs.insert(
            doc_id,
            DocEntry {
                url,
                split_ids,
                summary_ids,
                extra_summary_ids,
                seq,
                loc,
            },
        );
    }
    for _ in 0..n_splits {
        let split_id = r.u64()?;
        let doc_id = r.u64()?;
        let loc = r.loc()?;
        maps.splits.insert(split_id, SplitEntry { doc_id, loc });
    }
    for _ in 0..n_summaries {
        let summary_id = r.u64()?;
        let doc_id = r.u64()?;
        let split_id = r.u64()?;
        let loc = r.loc()?;
        maps.summaries.insert(
            summary_id,
            SummaryEntry {
                doc_id,
                split_id,
                loc,
            },
        );
    }

    Ok(Some((maps, snapshot_seq)))
}

/// Remove the snapshot, used when it no longer matches the manifest.
pub fn remove(dir: &Path) -> Result<()> {
    let file = path(dir);
    if file.exists() {
        fs::remove_file(&file)?;
    }
    Ok(())
}

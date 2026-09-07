//! Segments: one append-only file being written, any number of immutable memory-mapped ones.
//!
//! A segment keeps its id for life. The active segment is created with its final id and its file
//! is never renamed, so a `Loc` recorded while the segment was active stays correct after it is
//! sealed; only compaction rewrites locations.

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub fn segment_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("records-{:06}.seg", id))
}

/// The segment currently being appended to.
pub struct ActiveSegment {
    pub id: u32,
    pub path: PathBuf,
    writer: BufWriter<File>,
    reader: File,
    /// Bytes appended, buffered ones included.
    pub len: u64,
    pub first_seq: Option<u64>,
    pub last_seq: u64,
    unflushed: bool,
}

impl ActiveSegment {
    pub fn open(dir: &Path, id: u32) -> Result<Self> {
        let path = segment_path(dir, id);
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening segment {}", path.display()))?;
        let len = file.metadata()?.len();
        file.seek(SeekFrom::End(0))?;
        let reader = file.try_clone()?;
        Ok(ActiveSegment {
            id,
            path,
            writer: BufWriter::new(file),
            reader,
            len,
            first_seq: None,
            last_seq: 0,
            unflushed: false,
        })
    }

    /// Append one encoded record, returning the offset it landed at.
    pub fn append(&mut self, bytes: &[u8], seq: u64) -> Result<u64> {
        let offset = self.len;
        self.writer.write_all(bytes)?;
        self.len += bytes.len() as u64;
        self.unflushed = true;
        if self.first_seq.is_none() {
            self.first_seq = Some(seq);
        }
        self.last_seq = seq;
        Ok(offset)
    }

    /// Push buffered bytes to the file, so `read_at` can see them. Cheap; not a durability point.
    pub fn flush(&mut self) -> Result<()> {
        if self.unflushed {
            self.writer.flush()?;
            self.unflushed = false;
        }
        Ok(())
    }

    /// Durability point: flush, then ask the OS to put the data on the device.
    pub fn sync(&mut self) -> Result<()> {
        self.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    pub fn read_at(&self, offset: u64, len: u32) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        self.reader
            .read_exact_at(&mut buf, offset)
            .with_context(|| {
                format!(
                    "reading {} bytes at {} of {}",
                    len,
                    offset,
                    self.path.display()
                )
            })?;
        Ok(buf)
    }

    /// Read the whole segment, for replay.
    pub fn read_all(&self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.len as usize];
        if self.len > 0 {
            self.reader.read_exact_at(&mut buf, 0)?;
        }
        Ok(buf)
    }

    /// Cut a torn tail off. Used by replay when the last record is incomplete.
    pub fn truncate_to(&mut self, len: u64) -> Result<()> {
        self.flush()?;
        self.writer.get_ref().set_len(len)?;
        self.writer.get_mut().seek(SeekFrom::Start(len))?;
        self.len = len;
        Ok(())
    }

    /// Flush, fsync and hand back the file so it can be mapped read-only.
    pub fn seal(mut self) -> Result<SealedSegment> {
        self.sync()?;
        let ActiveSegment { id, path, .. } = self;
        SealedSegment::open(&path, id)
    }
}

/// An immutable, memory-mapped segment.
pub struct SealedSegment {
    pub id: u32,
    pub path: PathBuf,
    map: Mmap,
}

impl SealedSegment {
    pub fn open(path: &Path, id: u32) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Safety: the file is never written again once sealed, so the mapping cannot see a torn
        // update. Compaction writes a new file and swaps it in rather than editing this one.
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mapping {}", path.display()))?;
        Ok(SealedSegment {
            id,
            path: path.to_path_buf(),
            map,
        })
    }

    pub fn len(&self) -> u64 {
        self.map.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn slice(&self, offset: u64, len: u32) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start + len as usize;
        if end > self.map.len() {
            return Err(anyhow!(
                "location {}..{} is past the end of segment {} ({} bytes)",
                start,
                end,
                self.id,
                self.map.len()
            ));
        }
        Ok(&self.map[start..end])
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

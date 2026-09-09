//! The record frame and the four record kinds.
//!
//! Frame layout, all integers little-endian:
//!
//! ```text
//! len u32 | crc32 u32 | seq u64 | kind u8 | dtype u8 | meta_len u32 | meta (msgpack) | vector bytes
//! ```
//!
//! `len` is the whole record including the `len` field, so the next record starts at
//! `offset + len`. The CRC covers everything after the CRC field, that is `seq` onward.

use half::f16;
use serde::{Deserialize, Serialize};

/// len(4) + crc(4) + seq(8) + kind(1) + dtype(1) + meta_len(4)
pub const RECORD_HEADER_LEN: usize = 22;

/// The CRC is computed over the record from this offset to its end.
const CRC_COVERED_FROM: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    Doc = 1,
    Split = 2,
    Summary = 3,
    DeleteDoc = 4,
}

impl RecordKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(RecordKind::Doc),
            2 => Some(RecordKind::Split),
            3 => Some(RecordKind::Summary),
            4 => Some(RecordKind::DeleteDoc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VectorDtype {
    None = 0,
    F16 = 1,
    F32 = 2,
}

impl VectorDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(VectorDtype::None),
            1 => Some(VectorDtype::F16),
            2 => Some(VectorDtype::F32),
            _ => None,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            VectorDtype::None => 0,
            VectorDtype::F16 => 2,
            VectorDtype::F32 => 4,
        }
    }

    /// Byte length of a vector of `n_embd` elements in this dtype.
    pub fn vector_bytes(self, n_embd: usize) -> usize {
        self.bytes_per_element() * n_embd
    }
}

// ---------------------------------------------------------------------------
// Record metadata. One struct per kind; the vector never lives in the msgpack.
// ---------------------------------------------------------------------------

/// Document-level metadata. `summary_ids` is the document's own extractive summary list, in the
/// order the pipeline recorded it; `split_ids` is in split order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocMeta {
    pub doc_id: u64,
    pub url: String,
    pub split_ids: Vec<u64>,
    pub summary_ids: Option<Vec<u64>>,
}

/// A split. `summary_ids` is this split's own summary list; the same id may also appear in the
/// document's list, in which case the summary is still stored exactly once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitMeta {
    pub split_id: u64,
    pub seq_id: i32,
    pub doc_id: u64,
    pub token_len: usize,
    pub text: String,
    pub summary_ids: Vec<u64>,
    pub model_id: u64,
}

/// A summary. `split_id` records the split the sentence came from, `split_seq_id` its position
/// among that split's sentences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryMeta {
    pub summary_id: u64,
    pub doc_id: u64,
    pub split_id: u64,
    pub split_seq_id: i32,
    pub token_len: usize,
    pub centrality: f32,
    pub text: String,
    pub model_id: u64,
}

/// Tombstone for a whole document, written before a replacing insert and on delete.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteMeta {
    pub doc_id: u64,
}

// ---------------------------------------------------------------------------
// Encoding and decoding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes available than the frame claims: a torn tail, not corruption.
    Truncated {
        need: usize,
        have: usize,
    },
    /// A length that cannot be right for any record.
    BadLength(u32),
    BadCrc {
        expected: u32,
        found: u32,
    },
    BadKind(u8),
    BadDtype(u8),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Truncated { need, have } => {
                write!(f, "record truncated: need {} bytes, have {}", need, have)
            }
            DecodeError::BadLength(len) => write!(f, "impossible record length {}", len),
            DecodeError::BadCrc { expected, found } => {
                write!(
                    f,
                    "crc mismatch: expected {:#x}, found {:#x}",
                    expected, found
                )
            }
            DecodeError::BadKind(k) => write!(f, "unknown record kind {}", k),
            DecodeError::BadDtype(d) => write!(f, "unknown vector dtype {}", d),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug)]
pub struct RecordView<'a> {
    pub len: u32,
    pub seq: u64,
    pub kind: RecordKind,
    pub dtype: VectorDtype,
    pub meta: &'a [u8],
    pub vector: &'a [u8],
}

/// Build one record frame.
pub fn encode(
    seq: u64,
    kind: RecordKind,
    dtype: VectorDtype,
    meta: &[u8],
    vector: &[u8],
) -> Vec<u8> {
    let len = RECORD_HEADER_LEN + meta.len() + vector.len();
    let mut buf = Vec::with_capacity(len);
    buf.extend_from_slice(&(len as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // crc, filled in below
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.push(kind as u8);
    buf.push(dtype as u8);
    buf.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    buf.extend_from_slice(meta);
    buf.extend_from_slice(vector);
    let crc = crc32fast::hash(&buf[CRC_COVERED_FROM..]);
    buf[4..8].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Read the frame length without validating anything else.
pub fn peek_len(buf: &[u8]) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    Some(u32::from_le_bytes(buf[0..4].try_into().unwrap()))
}

/// Decode one record from the front of `buf`, validating the CRC.
pub fn decode(buf: &[u8]) -> Result<RecordView<'_>, DecodeError> {
    if buf.len() < RECORD_HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: RECORD_HEADER_LEN,
            have: buf.len(),
        });
    }
    let len = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if (len as usize) < RECORD_HEADER_LEN {
        return Err(DecodeError::BadLength(len));
    }
    if buf.len() < len as usize {
        return Err(DecodeError::Truncated {
            need: len as usize,
            have: buf.len(),
        });
    }
    let record = &buf[..len as usize];
    let expected = u32::from_le_bytes(record[4..8].try_into().unwrap());
    let found = crc32fast::hash(&record[CRC_COVERED_FROM..]);
    if expected != found {
        return Err(DecodeError::BadCrc { expected, found });
    }
    let seq = u64::from_le_bytes(record[8..16].try_into().unwrap());
    let kind = RecordKind::from_u8(record[16]).ok_or(DecodeError::BadKind(record[16]))?;
    let dtype = VectorDtype::from_u8(record[17]).ok_or(DecodeError::BadDtype(record[17]))?;
    let meta_len = u32::from_le_bytes(record[18..22].try_into().unwrap()) as usize;
    if RECORD_HEADER_LEN + meta_len > len as usize {
        return Err(DecodeError::BadLength(len));
    }
    let meta_end = RECORD_HEADER_LEN + meta_len;
    Ok(RecordView {
        len,
        seq,
        kind,
        dtype,
        meta: &record[RECORD_HEADER_LEN..meta_end],
        vector: &record[meta_end..],
    })
}

// ---------------------------------------------------------------------------
// Vector encoding
// ---------------------------------------------------------------------------

pub fn encode_vector(v: &[f32], dtype: VectorDtype) -> Vec<u8> {
    match dtype {
        VectorDtype::None => Vec::new(),
        VectorDtype::F16 => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&f16::from_f32(*x).to_bits().to_le_bytes());
            }
            out
        }
        VectorDtype::F32 => {
            let mut out = Vec::with_capacity(v.len() * 4);
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
    }
}

pub fn decode_vector(bytes: &[u8], dtype: VectorDtype) -> Vec<f32> {
    match dtype {
        VectorDtype::None => Vec::new(),
        VectorDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        VectorDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_frame() {
        let meta = rmp_serde::to_vec_named(&DeleteMeta { doc_id: 42 }).unwrap();
        let vector = encode_vector(&[1.0, -0.5, 0.25], VectorDtype::F16);
        let buf = encode(7, RecordKind::DeleteDoc, VectorDtype::F16, &meta, &vector);

        let view = decode(&buf).unwrap();
        assert_eq!(view.len as usize, buf.len());
        assert_eq!(view.seq, 7);
        assert_eq!(view.kind, RecordKind::DeleteDoc);
        assert_eq!(view.dtype, VectorDtype::F16);
        assert_eq!(view.meta, meta.as_slice());
        assert_eq!(
            decode_vector(view.vector, view.dtype),
            vec![1.0, -0.5, 0.25]
        );
    }

    #[test]
    fn detects_corruption() {
        let meta = rmp_serde::to_vec_named(&DeleteMeta { doc_id: 1 }).unwrap();
        let mut buf = encode(1, RecordKind::DeleteDoc, VectorDtype::None, &meta, &[]);
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        assert!(matches!(decode(&buf), Err(DecodeError::BadCrc { .. })));
    }

    #[test]
    fn detects_truncation() {
        let meta = rmp_serde::to_vec_named(&DeleteMeta { doc_id: 1 }).unwrap();
        let buf = encode(1, RecordKind::DeleteDoc, VectorDtype::None, &meta, &[]);
        let short = &buf[..buf.len() - 1];
        assert!(matches!(decode(short), Err(DecodeError::Truncated { .. })));
    }

    #[test]
    fn f16_round_trip_is_within_half_precision() {
        let original: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0 - 0.5).collect();
        let decoded = decode_vector(
            &encode_vector(&original, VectorDtype::F16),
            VectorDtype::F16,
        );
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-3, "{} vs {}", a, b);
        }
    }
}

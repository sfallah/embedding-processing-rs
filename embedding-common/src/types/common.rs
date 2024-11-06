use rmp_serde;
use rmp_serde::to_vec_named;
use serde::{Deserialize, Serialize};
use serde_json;

/// A trait to provide JSON and MsgPack serialization and deserialization for Rust structs.
pub trait Serde {
    /// Serializes the invoking object into a JSON string.
    fn to_json(&self) -> serde_json::Result<String>
    where
        Self: Sized + Serialize,
    {
        serde_json::to_string(self)
    }

    /// Deserializes a JSON string into an object of invoking type.
    fn from_json<'a, T>(s: &'a str) -> serde_json::Result<T>
    where
        T: Deserialize<'a> + Sized,
    {
        serde_json::from_str::<T>(s)
    }

    /// Packs the invoking object into a MsgPack byte vector.
    fn pack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
    where
        Self: Sized + Serialize,
    {
        Ok(to_vec_named(&self).unwrap())
    }

    /// Unpacks a MsgPack byte slice into an object of invoking type.
    fn unpack<'a, T>(b: &'a [u8]) -> Result<T, rmp_serde::decode::Error>
    where
        T: Deserialize<'a> + Sized,
    {
        rmp_serde::from_slice(b)
    }
}

use ahash::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

/// A deterministic hasher that uses the AHash algorithm with fixed seeds.
///
/// This struct allows for the creation of deterministic hashers using the AHash algorithm,
/// ensuring that the hash values remain consistent across different runs of the program
/// when the same seeds are used.
pub struct DeterministicAHasher {
    state: RandomState,
}

impl DeterministicAHasher {
    /// Creates a new `DeterministicAHasher` with the specified seeds.
    /// If no seeds are provided, default seeds are used.
    ///
    /// # Arguments
    ///
    /// * `seed1` - Optional first seed for the hasher.
    /// * `seed2` - Optional second seed for the hasher.
    ///
    pub fn new(seed1: Option<u64>, seed2: Option<u64>) -> Self {
        let seed1 = seed1.unwrap_or(0xDEADBEEFDEADBEEF);
        let seed2 = seed2.unwrap_or(0xCAFEBABECAFEBABE);
        DeterministicAHasher {
            state: RandomState::with_seeds(seed1, seed2, 0, 0),
        }
    }

    /// Generates a 64-bit hash for the given item.
    ///
    /// # Arguments
    ///
    /// * `t` - A reference to the item to be hashed. The item must implement the `Hash` trait.
    ///
    /// # Returns
    ///
    /// A 64-bit hash value.
    ///
    pub fn hash<T: Hash>(&self, t: &T) -> u64 {
        let mut hasher = self.state.build_hasher();
        t.hash(&mut hasher);
        hasher.finish()
    }
}

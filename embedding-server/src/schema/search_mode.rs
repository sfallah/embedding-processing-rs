use serde_repr::{Deserialize_repr, Serialize_repr};
use std::fmt;

/// Represents the types of search modes that can be used to query the vector DB.
#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum SearchModeType {
    /// Searching for similarity with splits only.
    SplitOnly = 1,
    /// Searching for similarity with summaries only.
    SummaryOnly = 2,
    /// Searching for similarity with both splits and summaries.
    SplitAndSummary = 3,
}

/// Converts a `SearchModeType` enum to its corresponding string representation.
impl fmt::Display for SearchModeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            SearchModeType::SplitOnly => "SplitOnly",
            SearchModeType::SummaryOnly => "SummaryOnly",
            SearchModeType::SplitAndSummary => "SplitAndSummary",
        };
        write!(f, "{}", s)
    }
}

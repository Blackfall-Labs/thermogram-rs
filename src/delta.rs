//! Delta - Represents a change to the Thermogram state
//!
//! Deltas are append-only records of state changes. They form the "dirty" state
//! before consolidation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A delta represents a single change to the Thermogram state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Unique ID for this delta
    pub id: String,

    /// Type of operation
    pub delta_type: DeltaType,

    /// Key being modified
    pub key: String,

    /// New value (serialized as bytes)
    pub value: Vec<u8>,

    /// Metadata about the change
    pub metadata: DeltaMetadata,

    /// Hash of previous delta (forms chain)
    pub prev_hash: Option<String>,

    /// Hash of this delta
    pub hash: String,
}

/// Types of delta operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaType {
    /// Create new key
    Create,

    /// Update existing key
    Update,

    /// Delete key (tombstone)
    Delete,

    /// Merge with existing value (for incremental updates)
    Merge,
}

/// Metadata attached to each delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMetadata {
    /// When this delta was created
    pub timestamp: DateTime<Utc>,

    /// Source of the delta (e.g., "llm_mining", "user_edit")
    pub source: String,

    /// Confidence or strength of this update (0.0 - 1.0)
    pub strength: f32,

    /// Optional: Number of observations contributing to this update
    pub observation_count: Option<usize>,

    /// Custom metadata
    pub custom: serde_json::Value,
}

impl Delta {
    /// Create a new delta
    pub fn new(
        delta_type: DeltaType,
        key: String,
        value: Vec<u8>,
        source: String,
        strength: f32,
        prev_hash: Option<String>,
    ) -> Self {
        let metadata = DeltaMetadata {
            timestamp: Utc::now(),
            source,
            strength,
            observation_count: None,
            custom: serde_json::Value::Null,
        };

        let mut delta = Self {
            id: uuid::Uuid::new_v4().to_string(),
            delta_type,
            key,
            value,
            metadata,
            prev_hash,
            hash: String::new(), // Computed below
        };

        delta.hash = delta.compute_hash();
        delta
    }

    /// Create a CREATE delta
    pub fn create(key: impl Into<String>, value: Vec<u8>, source: impl Into<String>) -> Self {
        Self::new(
            DeltaType::Create,
            key.into(),
            value,
            source.into(),
            1.0,
            None,
        )
    }

    /// Create an UPDATE delta
    pub fn update(
        key: impl Into<String>,
        value: Vec<u8>,
        source: impl Into<String>,
        strength: f32,
        prev_hash: Option<String>,
    ) -> Self {
        Self::new(
            DeltaType::Update,
            key.into(),
            value,
            source.into(),
            strength,
            prev_hash,
        )
    }

    /// Create a DELETE delta
    pub fn delete(key: impl Into<String>, source: impl Into<String>, prev_hash: Option<String>) -> Self {
        Self::new(
            DeltaType::Delete,
            key.into(),
            vec![],
            source.into(),
            1.0,
            prev_hash,
        )
    }

    /// Create a MERGE delta (incremental update)
    pub fn merge(
        key: impl Into<String>,
        value: Vec<u8>,
        source: impl Into<String>,
        strength: f32,
        prev_hash: Option<String>,
    ) -> Self {
        Self::new(
            DeltaType::Merge,
            key.into(),
            value,
            source.into(),
            strength,
            prev_hash,
        )
    }

    /// Compute hash of this delta
    pub fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash all fields except the hash itself
        hasher.update(self.id.as_bytes());
        hasher.update(&[self.delta_type as u8]);
        hasher.update(self.key.as_bytes());
        hasher.update(&self.value);
        hasher.update(self.metadata.timestamp.to_rfc3339().as_bytes());
        hasher.update(self.metadata.source.as_bytes());
        hasher.update(&self.metadata.strength.to_le_bytes());

        if let Some(ref prev) = self.prev_hash {
            hasher.update(prev.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Verify this delta's hash
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Verify this delta follows from the previous one
    pub fn verify_chain(&self, prev: &Delta) -> bool {
        match &self.prev_hash {
            Some(prev_hash) => prev_hash == &prev.hash,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_creation() {
        let delta = Delta::create("test_key", b"test_value".to_vec(), "test_source");

        assert_eq!(delta.delta_type, DeltaType::Create);
        assert_eq!(delta.key, "test_key");
        assert_eq!(delta.value, b"test_value");
        assert!(delta.verify_hash());
    }

    #[test]
    fn test_delta_chain() {
        let delta1 = Delta::create("key", b"value1".to_vec(), "source");
        let delta2 = Delta::update(
            "key",
            b"value2".to_vec(),
            "source",
            0.8,
            Some(delta1.hash.clone()),
        );

        assert!(delta2.verify_chain(&delta1));
    }

    #[test]
    fn test_hash_stability() {
        let delta = Delta::create("key", b"value".to_vec(), "source");
        let hash1 = delta.hash.clone();
        let hash2 = delta.compute_hash();

        assert_eq!(hash1, hash2);
    }
}

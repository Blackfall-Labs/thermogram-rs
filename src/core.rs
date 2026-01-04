//! Core Thermogram implementation
//!
//! The main Thermogram type combining dirty/clean states with hash-chained
//! deltas and plasticity rules.

use crate::consolidation::{consolidate, ConsolidatedEntry, ConsolidationPolicy, ConsolidationResult};
use crate::delta::{Delta, DeltaType};
use crate::error::{Error, Result};
use crate::hash_chain::HashChain;
use crate::plasticity::PlasticityRule;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A Thermogram - plastic memory capsule with dirty/clean states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thermogram {
    /// Unique ID for this thermogram
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Clean state (consolidated snapshot)
    pub clean_state: HashMap<String, ConsolidatedEntry>,

    /// Dirty state (pending deltas)
    pub dirty_chain: HashChain,

    /// Plasticity rule governing updates
    pub plasticity_rule: PlasticityRule,

    /// Consolidation policy
    pub consolidation_policy: ConsolidationPolicy,

    /// Metadata
    pub metadata: ThermogramMetadata,
}

/// Thermogram metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermogramMetadata {
    /// When this thermogram was created
    pub created_at: DateTime<Utc>,

    /// When last consolidated
    pub last_consolidation: DateTime<Utc>,

    /// Total deltas applied (lifetime)
    pub total_deltas: usize,

    /// Total consolidations performed
    pub total_consolidations: usize,

    /// Custom metadata
    pub custom: serde_json::Value,
}

impl Thermogram {
    /// Create a new Thermogram
    pub fn new(name: impl Into<String>, plasticity_rule: PlasticityRule) -> Self {
        let now = Utc::now();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            clean_state: HashMap::new(),
            dirty_chain: HashChain::new(),
            plasticity_rule,
            consolidation_policy: ConsolidationPolicy::default(),
            metadata: ThermogramMetadata {
                created_at: now,
                last_consolidation: now,
                total_deltas: 0,
                total_consolidations: 0,
                custom: serde_json::Value::Null,
            },
        }
    }

    /// Apply a delta to the dirty state
    pub fn apply_delta(&mut self, delta: Delta) -> Result<()> {
        // Append to hash chain
        self.dirty_chain.append(delta)?;
        self.metadata.total_deltas += 1;

        // Check if we should auto-consolidate
        if self.should_consolidate() {
            self.consolidate()?;
        }

        Ok(())
    }

    /// Read current value for a key (merges dirty + clean)
    pub fn read(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Check dirty state first
        if let Some(delta) = self.dirty_chain.get_latest(key) {
            match delta.delta_type {
                DeltaType::Create | DeltaType::Update | DeltaType::Merge => {
                    return Ok(Some(delta.value.clone()));
                }
                DeltaType::Delete => {
                    return Ok(None);
                }
            }
        }

        // Fall back to clean state
        Ok(self.clean_state.get(key).map(|entry| entry.value.clone()))
    }

    /// Read with strength information
    pub fn read_with_strength(&self, key: &str) -> Result<Option<(Vec<u8>, f32)>> {
        // Check dirty state first
        if let Some(delta) = self.dirty_chain.get_latest(key) {
            match delta.delta_type {
                DeltaType::Create | DeltaType::Update | DeltaType::Merge => {
                    return Ok(Some((delta.value.clone(), delta.metadata.strength)));
                }
                DeltaType::Delete => {
                    return Ok(None);
                }
            }
        }

        // Fall back to clean state
        Ok(self
            .clean_state
            .get(key)
            .map(|entry| (entry.value.clone(), entry.strength)))
    }

    /// List all keys in current state
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.clean_state.keys().cloned().collect();

        // Add dirty keys not in clean
        for delta in &self.dirty_chain.deltas {
            if !keys.contains(&delta.key) && delta.delta_type != DeltaType::Delete {
                keys.push(delta.key.clone());
            }
        }

        keys
    }

    /// Get history for a key
    pub fn history(&self, key: &str) -> Vec<&Delta> {
        self.dirty_chain.get_history(key)
    }

    /// Check if consolidation should happen
    pub fn should_consolidate(&self) -> bool {
        let dirty_size = self.dirty_chain.len();
        let dirty_bytes = self.estimate_dirty_size();

        self.consolidation_policy.should_consolidate(
            dirty_size,
            &self.metadata.last_consolidation,
            dirty_bytes,
        )
    }

    /// Manually trigger consolidation
    pub fn consolidate(&mut self) -> Result<ConsolidationResult> {
        let (new_clean_state, result) = consolidate(
            &self.dirty_chain.deltas,
            &self.clean_state,
            &self.plasticity_rule,
            &self.consolidation_policy,
        )?;

        // Update state
        self.clean_state = new_clean_state;
        self.dirty_chain = HashChain::new();
        self.metadata.last_consolidation = Utc::now();
        self.metadata.total_consolidations += 1;

        Ok(result)
    }

    /// Estimate size of dirty state in bytes
    fn estimate_dirty_size(&self) -> usize {
        self.dirty_chain
            .deltas
            .iter()
            .map(|d| d.value.len() + d.key.len() + 200) // Rough estimate
            .sum()
    }

    /// Save to disk
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Serialize to JSON
        let json = serde_json::to_string_pretty(self)?;

        // Write to file
        std::fs::write(path, json)?;

        Ok(())
    }

    /// Load from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Read file
        let json = std::fs::read_to_string(path)?;

        // Deserialize
        let thermo: Thermogram = serde_json::from_str(&json)?;

        // Verify hash chain
        thermo.dirty_chain.verify()?;

        Ok(thermo)
    }

    /// Get statistics
    pub fn stats(&self) -> ThermogramStats {
        ThermogramStats {
            total_keys: self.keys().len(),
            clean_entries: self.clean_state.len(),
            dirty_deltas: self.dirty_chain.len(),
            total_deltas_lifetime: self.metadata.total_deltas,
            total_consolidations: self.metadata.total_consolidations,
            created_at: self.metadata.created_at,
            last_consolidation: self.metadata.last_consolidation,
            estimated_size_bytes: self.estimate_size(),
        }
    }

    /// Estimate total size in bytes
    fn estimate_size(&self) -> usize {
        let clean_size: usize = self
            .clean_state
            .values()
            .map(|e| e.value.len() + e.key.len() + 100)
            .sum();

        clean_size + self.estimate_dirty_size()
    }
}

/// Statistics about a Thermogram
#[derive(Debug, Clone)]
pub struct ThermogramStats {
    pub total_keys: usize,
    pub clean_entries: usize,
    pub dirty_deltas: usize,
    pub total_deltas_lifetime: usize,
    pub total_consolidations: usize,
    pub created_at: DateTime<Utc>,
    pub last_consolidation: DateTime<Utc>,
    pub estimated_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_thermogram() {
        let thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        assert_eq!(thermo.name, "test");
        assert!(thermo.clean_state.is_empty());
        assert!(thermo.dirty_chain.is_empty());
    }

    #[test]
    fn test_apply_and_read() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta = Delta::create("key1", b"value1".to_vec(), "source");
        thermo.apply_delta(delta).unwrap();

        let value = thermo.read("key1").unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_update() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta1 = Delta::create("key1", b"value1".to_vec(), "source");
        thermo.apply_delta(delta1).unwrap();

        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let delta2 = Delta::update("key1", b"value2".to_vec(), "source", 0.8, prev_hash);
        thermo.apply_delta(delta2).unwrap();

        let value = thermo.read("key1").unwrap();
        assert_eq!(value, Some(b"value2".to_vec()));
    }

    #[test]
    fn test_delete() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta1 = Delta::create("key1", b"value1".to_vec(), "source");
        thermo.apply_delta(delta1).unwrap();

        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let delta2 = Delta::delete("key1", "source", prev_hash);
        thermo.apply_delta(delta2).unwrap();

        let value = thermo.read("key1").unwrap();
        assert_eq!(value, None);
    }

    #[test]
    fn test_manual_consolidation() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta = Delta::create("key1", b"value1".to_vec(), "source");
        thermo.apply_delta(delta).unwrap();

        // Manually consolidate
        let result = thermo.consolidate().unwrap();

        assert_eq!(result.deltas_processed, 1);
        assert_eq!(thermo.clean_state.len(), 1);
        assert_eq!(thermo.dirty_chain.len(), 0);
    }

    #[test]
    fn test_save_load() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.thermo");

        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        let delta = Delta::create("key1", b"value1".to_vec(), "source");
        thermo.apply_delta(delta).unwrap();

        thermo.save(&path).unwrap();

        let loaded = Thermogram::load(&path).unwrap();
        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.read("key1").unwrap(), Some(b"value1".to_vec()));
    }
}

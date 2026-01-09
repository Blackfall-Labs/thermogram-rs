//! Core Thermogram implementation
//!
//! The main Thermogram type combining hot/cold tensor states with hash-chained
//! deltas and plasticity rules.
//!
//! ## Thermal States
//!
//! - **Hot tensors**: High plasticity, volatile, session-local. Fast to update.
//! - **Cold tensors**: Crystallized, stable, personality backbone. Slow to change.
//!
//! Consolidation crystallizes hot → cold. Warming moves cold → hot when needed.
//! Single file, two internal states, bidirectional transitions.

use crate::consolidation::{ConsolidatedEntry, ConsolidationPolicy, ConsolidationResult};
use crate::delta::{Delta, DeltaType};
use crate::error::Result;
use crate::hash_chain::HashChain;
use crate::plasticity::PlasticityRule;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Thermal state of a tensor entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThermalState {
    /// Hot: High plasticity, volatile, session-local
    #[default]
    Hot,
    /// Cold: Crystallized, stable, personality backbone
    Cold,
}

/// Configuration for thermal transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Minimum strength to crystallize (hot → cold)
    pub crystallization_threshold: f32,
    /// Minimum observations before crystallization eligible
    pub min_observations: usize,
    /// Strength below which hot entries are pruned
    pub prune_threshold: f32,
    /// Whether cold entries can be warmed back to hot
    pub allow_warming: bool,
    /// Strength increase required to warm a cold entry
    pub warming_delta: f32,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            crystallization_threshold: 0.75,
            min_observations: 3,
            prune_threshold: 0.1,
            allow_warming: true,
            warming_delta: 0.3,
        }
    }
}

/// A Thermogram - plastic memory capsule with hot/cold tensor states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thermogram {
    /// Unique ID for this thermogram
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Hot tensors (high plasticity, volatile)
    pub hot_entries: HashMap<String, ConsolidatedEntry>,

    /// Cold tensors (crystallized, stable)
    pub cold_entries: HashMap<String, ConsolidatedEntry>,

    /// Pending deltas (audit trail)
    pub dirty_chain: HashChain,

    /// Plasticity rule governing updates
    pub plasticity_rule: PlasticityRule,

    /// Consolidation policy
    pub consolidation_policy: ConsolidationPolicy,

    /// Thermal transition config
    pub thermal_config: ThermalConfig,

    /// Metadata
    pub metadata: ThermogramMetadata,
}

/// Result of crystallization operation
#[derive(Debug, Clone, Default)]
pub struct CrystallizationResult {
    /// New entries crystallized to cold layer
    pub crystallized: usize,
    /// Existing cold entries that were merged/updated
    pub merged: usize,
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
            hot_entries: HashMap::new(),
            cold_entries: HashMap::new(),
            dirty_chain: HashChain::new(),
            plasticity_rule,
            consolidation_policy: ConsolidationPolicy::default(),
            thermal_config: ThermalConfig::default(),
            metadata: ThermogramMetadata {
                created_at: now,
                last_consolidation: now,
                total_deltas: 0,
                total_consolidations: 0,
                custom: serde_json::Value::Null,
            },
        }
    }

    /// Create with custom thermal config
    pub fn with_thermal_config(
        name: impl Into<String>,
        plasticity_rule: PlasticityRule,
        thermal_config: ThermalConfig,
    ) -> Self {
        let mut thermo = Self::new(name, plasticity_rule);
        thermo.thermal_config = thermal_config;
        thermo
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

    /// Read current value for a key (dirty → hot → cold priority)
    pub fn read(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Check dirty state first (uncommitted changes)
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

        // Check hot tensors (session-local)
        if let Some(entry) = self.hot_entries.get(key) {
            return Ok(Some(entry.value.clone()));
        }

        // Fall back to cold tensors (crystallized)
        Ok(self.cold_entries.get(key).map(|entry| entry.value.clone()))
    }

    /// Read with strength and thermal state information
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

        // Check hot tensors
        if let Some(entry) = self.hot_entries.get(key) {
            return Ok(Some((entry.value.clone(), entry.strength)));
        }

        // Fall back to cold tensors
        Ok(self
            .cold_entries
            .get(key)
            .map(|entry| (entry.value.clone(), entry.strength)))
    }

    /// Read with full thermal state information
    pub fn read_with_state(&self, key: &str) -> Result<Option<(Vec<u8>, f32, ThermalState)>> {
        // Check dirty state first
        if let Some(delta) = self.dirty_chain.get_latest(key) {
            match delta.delta_type {
                DeltaType::Create | DeltaType::Update | DeltaType::Merge => {
                    return Ok(Some((delta.value.clone(), delta.metadata.strength, ThermalState::Hot)));
                }
                DeltaType::Delete => {
                    return Ok(None);
                }
            }
        }

        // Check hot tensors
        if let Some(entry) = self.hot_entries.get(key) {
            return Ok(Some((entry.value.clone(), entry.strength, ThermalState::Hot)));
        }

        // Fall back to cold tensors
        Ok(self
            .cold_entries
            .get(key)
            .map(|entry| (entry.value.clone(), entry.strength, ThermalState::Cold)))
    }

    /// List all keys in current state (hot + cold + dirty)
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.cold_entries.keys().cloned().collect();

        // Add hot keys not in cold
        for key in self.hot_entries.keys() {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        }

        // Add dirty keys not in hot or cold
        for delta in &self.dirty_chain.deltas {
            if !keys.contains(&delta.key) && delta.delta_type != DeltaType::Delete {
                keys.push(delta.key.clone());
            }
        }

        keys
    }

    /// List only hot tensor keys
    pub fn hot_keys(&self) -> Vec<String> {
        self.hot_entries.keys().cloned().collect()
    }

    /// List only cold tensor keys
    pub fn cold_keys(&self) -> Vec<String> {
        self.cold_entries.keys().cloned().collect()
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

    /// Manually trigger consolidation (dirty → hot, then crystallize eligible hot → cold)
    pub fn consolidate(&mut self) -> Result<ConsolidationResult> {
        // First: apply dirty deltas to hot layer
        let (new_hot_state, mut result) = crate::consolidation::consolidate(
            &self.dirty_chain.deltas,
            &self.hot_entries,
            &self.plasticity_rule,
            &self.consolidation_policy,
        )?;

        // Update hot state
        self.hot_entries = new_hot_state;
        self.dirty_chain = HashChain::new();
        self.metadata.last_consolidation = Utc::now();
        self.metadata.total_consolidations += 1;

        // Second: crystallize eligible hot entries to cold
        let crystal_result = self.crystallize()?;
        result.entries_merged += crystal_result.crystallized;

        Ok(result)
    }

    /// Crystallize high-confidence hot entries to cold layer
    ///
    /// Entries are crystallized when:
    /// - Strength >= crystallization_threshold
    /// - Update count >= min_observations
    pub fn crystallize(&mut self) -> Result<CrystallizationResult> {
        let mut result = CrystallizationResult::default();
        let mut keys_to_crystallize = Vec::new();

        // Find eligible entries
        for (key, entry) in &self.hot_entries {
            if entry.strength >= self.thermal_config.crystallization_threshold
                && entry.update_count >= self.thermal_config.min_observations
            {
                keys_to_crystallize.push(key.clone());
            }
        }

        // Move to cold layer
        for key in keys_to_crystallize {
            if let Some(entry) = self.hot_entries.remove(&key) {
                // Merge with existing cold entry if present
                if let Some(cold_entry) = self.cold_entries.get_mut(&key) {
                    // Average the strengths, weighted toward the new one
                    cold_entry.strength = cold_entry.strength * 0.3 + entry.strength * 0.7;
                    cold_entry.value = entry.value;
                    cold_entry.updated_at = entry.updated_at;
                    cold_entry.update_count += entry.update_count;
                    result.merged += 1;
                } else {
                    // New cold entry
                    self.cold_entries.insert(key, entry);
                    result.crystallized += 1;
                }
            }
        }

        Ok(result)
    }

    /// Warm a cold entry back to hot (reactivate for updates)
    ///
    /// Used when a cold entry needs modification. The entry moves
    /// from cold → hot where it can be updated with high plasticity.
    pub fn warm(&mut self, key: &str) -> Result<bool> {
        if !self.thermal_config.allow_warming {
            return Ok(false);
        }

        if let Some(mut entry) = self.cold_entries.remove(key) {
            // Reduce strength slightly (warming cost)
            entry.strength = (entry.strength - self.thermal_config.warming_delta * 0.1).max(0.1);
            entry.updated_at = Utc::now();
            self.hot_entries.insert(key.to_string(), entry);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Warm all cold entries matching a predicate
    pub fn warm_matching<F>(&mut self, predicate: F) -> Result<usize>
    where
        F: Fn(&str, &ConsolidatedEntry) -> bool,
    {
        if !self.thermal_config.allow_warming {
            return Ok(0);
        }

        let keys_to_warm: Vec<String> = self
            .cold_entries
            .iter()
            .filter(|(k, v)| predicate(k, v))
            .map(|(k, _)| k.clone())
            .collect();

        let mut warmed = 0;
        for key in keys_to_warm {
            if self.warm(&key)? {
                warmed += 1;
            }
        }

        Ok(warmed)
    }

    /// Prune weak hot entries below threshold
    pub fn prune_hot(&mut self) -> usize {
        let before = self.hot_entries.len();
        self.hot_entries.retain(|_, entry| {
            entry.strength >= self.thermal_config.prune_threshold
        });
        before - self.hot_entries.len()
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
            hot_entries: self.hot_entries.len(),
            cold_entries: self.cold_entries.len(),
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
        let hot_size: usize = self
            .hot_entries
            .values()
            .map(|e| e.value.len() + e.key.len() + 100)
            .sum();

        let cold_size: usize = self
            .cold_entries
            .values()
            .map(|e| e.value.len() + e.key.len() + 100)
            .sum();

        hot_size + cold_size + self.estimate_dirty_size()
    }
}

/// Statistics about a Thermogram
#[derive(Debug, Clone)]
pub struct ThermogramStats {
    pub total_keys: usize,
    pub hot_entries: usize,
    pub cold_entries: usize,
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
        assert!(thermo.hot_entries.is_empty());
        assert!(thermo.cold_entries.is_empty());
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

        // Manually consolidate (dirty → hot)
        let result = thermo.consolidate().unwrap();

        assert_eq!(result.deltas_processed, 1);
        assert_eq!(thermo.hot_entries.len(), 1);
        assert_eq!(thermo.dirty_chain.len(), 0);
    }

    #[test]
    fn test_crystallization() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        thermo.thermal_config.crystallization_threshold = 0.7;
        thermo.thermal_config.min_observations = 2;

        // Create high-strength entry with enough observations
        let mut delta = Delta::create("key1", b"value1".to_vec(), "source");
        delta.metadata.strength = 0.9;
        thermo.apply_delta(delta).unwrap();
        thermo.consolidate().unwrap();

        // Add more updates to reach min_observations
        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let mut delta2 = Delta::update("key1", b"value1".to_vec(), "source", 0.9, prev_hash);
        delta2.metadata.strength = 0.9;
        thermo.apply_delta(delta2).unwrap();
        thermo.consolidate().unwrap();

        // Should have crystallized to cold
        assert_eq!(thermo.cold_entries.len(), 1);
        assert!(thermo.hot_entries.is_empty());
    }

    #[test]
    fn test_warming() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        // Manually add to cold layer
        thermo.cold_entries.insert(
            "cold_key".to_string(),
            ConsolidatedEntry {
                key: "cold_key".to_string(),
                value: b"cold_value".to_vec(),
                strength: 0.8,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 5,
            },
        );

        // Warm it back to hot
        let warmed = thermo.warm("cold_key").unwrap();
        assert!(warmed);
        assert!(thermo.cold_entries.is_empty());
        assert_eq!(thermo.hot_entries.len(), 1);
    }

    #[test]
    fn test_thermal_state_read() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        // Add to hot
        thermo.hot_entries.insert(
            "hot_key".to_string(),
            ConsolidatedEntry {
                key: "hot_key".to_string(),
                value: b"hot".to_vec(),
                strength: 0.5,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 1,
            },
        );

        // Add to cold
        thermo.cold_entries.insert(
            "cold_key".to_string(),
            ConsolidatedEntry {
                key: "cold_key".to_string(),
                value: b"cold".to_vec(),
                strength: 0.9,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 10,
            },
        );

        // Verify thermal states
        let (_, _, state) = thermo.read_with_state("hot_key").unwrap().unwrap();
        assert_eq!(state, ThermalState::Hot);

        let (_, _, state) = thermo.read_with_state("cold_key").unwrap().unwrap();
        assert_eq!(state, ThermalState::Cold);
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

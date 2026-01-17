//! # Thermogram Colony
//!
//! A colony is a collection of thermograms that act as a unit.
//! Colonies grow, split, merge, and balance load automatically.
//!
//! ## Biological Rationale
//!
//! Brain regions don't have fixed capacity - they grow and reorganize.
//! Thermogram colonies mimic this by:
//! - Starting with 1 thermogram
//! - Splitting when capacity exceeded (maintaining connectome locality)
//! - Merging when under-utilized
//! - Balancing load during consolidation
//!
//! ## Colony Rules
//!
//! 1. Each mesh starts with 1 thermogram
//! 2. When capacity exceeded, split into 2 (locality-preserving)
//! 3. Consolidation balances load across members
//! 4. Related synapses stay in same capsule
//! 5. Colony can grow to N thermograms as expertise deepens

use crate::{
    consolidation::ConsolidatedEntry,
    core::{Thermogram, ThermalConfig, ThermalState},
    delta::Delta,
    error::Result,
    plasticity::PlasticityRule,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Colony metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonyMetadata {
    /// Unique colony identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Creation timestamp
    pub created: DateTime<Utc>,

    /// Last modification timestamp
    pub modified: DateTime<Utc>,

    /// Total entry count across all members
    pub total_entries: usize,

    /// Number of splits performed
    pub split_count: usize,

    /// Number of merges performed
    pub merge_count: usize,
}

impl ColonyMetadata {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            name: name.into(),
            created: now,
            modified: now,
            total_entries: 0,
            split_count: 0,
            merge_count: 0,
        }
    }
}

/// Result of a colony consolidation operation
#[derive(Debug, Clone, Default)]
pub struct ColonyConsolidationResult {
    /// Entries promoted (across all members)
    pub promoted: usize,

    /// Entries demoted (across all members)
    pub demoted: usize,

    /// Entries pruned (across all members)
    pub pruned: usize,

    /// Entries moved between members for balancing
    pub rebalanced: usize,

    /// Splits performed
    pub splits: usize,

    /// Merges performed
    pub merges: usize,
}

/// Configuration for colony behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonyConfig {
    /// Maximum entries per thermogram before split
    pub split_threshold: usize,

    /// Minimum entries per thermogram before merge candidate
    pub merge_threshold: usize,

    /// Target balance ratio (0.5 = perfect balance)
    pub balance_target: f32,

    /// Maximum deviation from balance before rebalancing
    pub balance_tolerance: f32,

    /// Maximum number of thermograms in colony
    pub max_members: usize,

    /// Thermal config for new members
    pub thermal_config: ThermalConfig,
}

impl Default for ColonyConfig {
    fn default() -> Self {
        Self {
            split_threshold: 10000,
            merge_threshold: 1000,
            balance_target: 0.5,
            balance_tolerance: 0.3,
            max_members: 16,
            thermal_config: ThermalConfig::default(),
        }
    }
}

impl ColonyConfig {
    /// Config optimized for fast-learning agents (smaller, faster)
    pub fn fast_learner() -> Self {
        Self {
            split_threshold: 5000,
            merge_threshold: 500,
            balance_target: 0.5,
            balance_tolerance: 0.25,
            max_members: 8,
            thermal_config: ThermalConfig::fast_learner(),
        }
    }
}

/// A colony of thermograms for a single mesh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermogramColony {
    /// The thermograms in this colony
    pub members: Vec<Thermogram>,

    /// Colony configuration
    pub config: ColonyConfig,

    /// Colony metadata
    pub metadata: ColonyMetadata,

    /// Key to member index mapping (for locality)
    #[serde(default)]
    key_locality: HashMap<String, usize>,
}

impl ThermogramColony {
    /// Create a new colony with a single thermogram
    pub fn new(id: impl Into<String>, name: impl Into<String>, rule: PlasticityRule) -> Self {
        let id = id.into();
        let name = name.into();
        let config = ColonyConfig::default();

        let first_member = Thermogram::with_thermal_config(
            format!("{}_0", id),
            rule,
            config.thermal_config.clone(),
        );

        Self {
            members: vec![first_member],
            config,
            metadata: ColonyMetadata::new(&id, &name),
            key_locality: HashMap::new(),
        }
    }

    /// Create colony with custom config
    pub fn with_config(
        id: impl Into<String>,
        name: impl Into<String>,
        rule: PlasticityRule,
        config: ColonyConfig,
    ) -> Self {
        let id = id.into();
        let name = name.into();

        let first_member = Thermogram::with_thermal_config(
            format!("{}_0", id),
            rule,
            config.thermal_config.clone(),
        );

        Self {
            members: vec![first_member],
            config,
            metadata: ColonyMetadata::new(&id, &name),
            key_locality: HashMap::new(),
        }
    }

    /// Get total entry count across all members and temperatures
    pub fn total_entries(&self) -> usize {
        self.members.iter().map(|m| m.stats().total_keys).sum()
    }

    /// Get number of members in colony
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Read from any thermogram in colony (checks hot→warm→cool→cold order)
    /// Returns the ConsolidatedEntry if found
    pub fn read(&self, key: &str) -> Option<&ConsolidatedEntry> {
        // First check locality hint
        if let Some(&idx) = self.key_locality.get(key) {
            if let Some(member) = self.members.get(idx) {
                if let Some(entry) = Self::find_entry_in_member(member, key) {
                    return Some(entry);
                }
            }
        }

        // Fall back to linear search
        for member in &self.members {
            if let Some(entry) = Self::find_entry_in_member(member, key) {
                return Some(entry);
            }
        }
        None
    }

    /// Find entry in a single member (hot→warm→cool→cold priority)
    fn find_entry_in_member<'a>(member: &'a Thermogram, key: &str) -> Option<&'a ConsolidatedEntry> {
        for state in ThermalState::all() {
            if let Some(entry) = member.entries(state).get(key) {
                return Some(entry);
            }
        }
        None
    }

    /// Check if key exists in any member
    fn key_exists_in_member(member: &Thermogram, key: &str) -> bool {
        ThermalState::all().iter().any(|state| member.entries(*state).contains_key(key))
    }

    /// Read from specific temperature layer across all members
    pub fn read_layer(&self, key: &str, state: ThermalState) -> Option<&ConsolidatedEntry> {
        // Check locality hint first
        if let Some(&idx) = self.key_locality.get(key) {
            if let Some(member) = self.members.get(idx) {
                if let Some(entry) = member.entries(state).get(key) {
                    return Some(entry);
                }
            }
        }

        // Fall back to linear search
        for member in &self.members {
            if let Some(entry) = member.entries(state).get(key) {
                return Some(entry);
            }
        }
        None
    }

    /// Write to appropriate thermogram (based on key locality)
    pub fn write(&mut self, key: &str, entry: ConsolidatedEntry, state: ThermalState) {
        // Determine target member
        let target_idx = self.select_member_for_write(key);

        // Update locality mapping
        self.key_locality.insert(key.to_string(), target_idx);

        // Write to target
        if let Some(member) = self.members.get_mut(target_idx) {
            member.entries_mut(state).insert(key.to_string(), entry);
        }

        self.metadata.modified = Utc::now();
    }

    /// Apply delta to colony (routes to appropriate member)
    pub fn apply_delta(&mut self, delta: Delta) -> Result<()> {
        let key = delta.key.clone();
        let target_idx = self.select_member_for_write(&key);

        // Update locality mapping
        self.key_locality.insert(key, target_idx);

        // Apply to target
        if let Some(member) = self.members.get_mut(target_idx) {
            member.apply_delta(delta)?;
        }

        self.metadata.modified = Utc::now();
        Ok(())
    }

    /// Select member for writing (locality-aware)
    fn select_member_for_write(&self, key: &str) -> usize {
        // Check existing locality
        if let Some(&idx) = self.key_locality.get(key) {
            if idx < self.members.len() {
                return idx;
            }
        }

        // Find member with most related keys (simple prefix matching)
        let prefix = key.split('_').next().unwrap_or(key);
        let mut best_idx = 0;
        let mut best_score = 0usize;

        for (idx, member) in self.members.iter().enumerate() {
            let score = member
                .hot_entries
                .keys()
                .chain(member.warm_entries.keys())
                .chain(member.cool_entries.keys())
                .chain(member.cold_entries.keys())
                .filter(|k| k.starts_with(prefix))
                .count();

            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        // Fall back to least loaded member
        if best_score == 0 {
            let mut min_entries = usize::MAX;
            for (idx, member) in self.members.iter().enumerate() {
                let entries = member.stats().total_keys;
                if entries < min_entries {
                    min_entries = entries;
                    best_idx = idx;
                }
            }
        }

        best_idx
    }

    /// Consolidate all members and balance load
    pub fn consolidate(&mut self) -> Result<ColonyConsolidationResult> {
        let mut result = ColonyConsolidationResult::default();

        // Consolidate each member
        for member in &mut self.members {
            member.consolidate()?;
        }

        // Run thermal transitions on each member
        for member in &mut self.members {
            member.run_thermal_transitions()?;
            // Note: individual promotion/demotion counts not tracked at colony level
        }

        // Check for splits needed
        result.splits = self.check_and_split()?;

        // Check for merges needed
        result.merges = self.check_and_merge()?;

        // Rebalance if needed
        result.rebalanced = self.rebalance()?;

        // Update metadata
        self.metadata.total_entries = self.total_entries();
        self.metadata.modified = Utc::now();

        Ok(result)
    }

    /// Check and perform splits where needed
    fn check_and_split(&mut self) -> Result<usize> {
        if self.members.len() >= self.config.max_members {
            return Ok(0);
        }

        let mut splits = 0;
        let mut indices_to_split = Vec::new();

        // Find members that need splitting
        for (idx, member) in self.members.iter().enumerate() {
            if member.stats().total_keys > self.config.split_threshold {
                indices_to_split.push(idx);
            }
        }

        // Perform splits (in reverse to maintain indices)
        for idx in indices_to_split.into_iter().rev() {
            if self.members.len() >= self.config.max_members {
                break;
            }
            self.split_member(idx)?;
            splits += 1;
        }

        self.metadata.split_count += splits;
        Ok(splits)
    }

    /// Split a member into two (locality-preserving)
    fn split_member(&mut self, idx: usize) -> Result<()> {
        let member = &self.members[idx];
        let new_id = format!("{}_{}", self.metadata.id, self.members.len());

        // Create new member
        let mut new_member = Thermogram::with_thermal_config(
            new_id,
            member.plasticity_rule.clone(),
            self.config.thermal_config.clone(),
        );

        // Partition entries by key prefix locality
        let entries_to_move = self.select_entries_for_split(idx);

        // Move entries to new member
        for (key, state) in entries_to_move {
            if let Some(entry) = self.members[idx].entries_mut(state).remove(&key) {
                new_member.entries_mut(state).insert(key.clone(), entry);

                // Update locality mapping
                self.key_locality
                    .insert(key, self.members.len());
            }
        }

        self.members.push(new_member);
        Ok(())
    }

    /// Select entries to move during split (locality-aware)
    fn select_entries_for_split(&self, idx: usize) -> Vec<(String, ThermalState)> {
        let member = &self.members[idx];
        let mut entries = Vec::new();

        // Collect all entries with their state
        for (key, _) in &member.hot_entries {
            entries.push((key.clone(), ThermalState::Hot));
        }
        for (key, _) in &member.warm_entries {
            entries.push((key.clone(), ThermalState::Warm));
        }
        for (key, _) in &member.cool_entries {
            entries.push((key.clone(), ThermalState::Cool));
        }
        for (key, _) in &member.cold_entries {
            entries.push((key.clone(), ThermalState::Cold));
        }

        // Sort by key for locality grouping
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Take second half (roughly)
        let split_point = entries.len() / 2;
        entries.into_iter().skip(split_point).collect()
    }

    /// Check and perform merges where needed
    fn check_and_merge(&mut self) -> Result<usize> {
        if self.members.len() <= 1 {
            return Ok(0);
        }

        let mut merges = 0;

        // Find small members
        loop {
            let small_indices: Vec<usize> = self
                .members
                .iter()
                .enumerate()
                .filter(|(_, m)| m.stats().total_keys < self.config.merge_threshold)
                .map(|(i, _)| i)
                .collect();

            if small_indices.len() < 2 {
                break;
            }

            // Merge first two small members
            let idx_b = small_indices[1];
            let idx_a = small_indices[0];

            self.merge_members(idx_a, idx_b)?;
            merges += 1;
        }

        self.metadata.merge_count += merges;
        Ok(merges)
    }

    /// Merge two members
    fn merge_members(&mut self, keep_idx: usize, remove_idx: usize) -> Result<()> {
        // Remove the member to be merged
        let removed = self.members.remove(remove_idx);

        // Move all entries to the kept member
        for (key, entry) in removed.hot_entries {
            self.members[keep_idx]
                .hot_entries
                .insert(key.clone(), entry);
            self.key_locality.insert(key, keep_idx);
        }
        for (key, entry) in removed.warm_entries {
            self.members[keep_idx]
                .warm_entries
                .insert(key.clone(), entry);
            self.key_locality.insert(key, keep_idx);
        }
        for (key, entry) in removed.cool_entries {
            self.members[keep_idx]
                .cool_entries
                .insert(key.clone(), entry);
            self.key_locality.insert(key, keep_idx);
        }
        for (key, entry) in removed.cold_entries {
            self.members[keep_idx]
                .cold_entries
                .insert(key.clone(), entry);
            self.key_locality.insert(key, keep_idx);
        }

        // Update locality indices for remaining members
        for (_, idx) in self.key_locality.iter_mut() {
            if *idx > remove_idx {
                *idx -= 1;
            }
        }

        Ok(())
    }

    /// Rebalance entries across members
    fn rebalance(&mut self) -> Result<usize> {
        if self.members.len() <= 1 {
            return Ok(0);
        }

        let total = self.total_entries();
        if total == 0 {
            return Ok(0);
        }

        let target_per_member = total / self.members.len();
        let tolerance = (target_per_member as f32 * self.config.balance_tolerance) as usize;

        let mut moved = 0;

        // Find overloaded and underloaded members
        let mut overloaded: Vec<usize> = Vec::new();
        let mut underloaded: Vec<usize> = Vec::new();

        for (idx, member) in self.members.iter().enumerate() {
            let count = member.stats().total_keys;
            if count > target_per_member + tolerance {
                overloaded.push(idx);
            } else if count < target_per_member.saturating_sub(tolerance) {
                underloaded.push(idx);
            }
        }

        // Move entries from overloaded to underloaded
        for &over_idx in &overloaded {
            for &under_idx in &underloaded {
                let over_count = self.members[over_idx].stats().total_keys;
                let under_count = self.members[under_idx].stats().total_keys;

                if over_count <= target_per_member + tolerance {
                    break;
                }

                let to_move = (over_count - target_per_member).min(target_per_member - under_count);

                moved += self.move_entries(over_idx, under_idx, to_move);
            }
        }

        Ok(moved)
    }

    /// Move entries between members
    fn move_entries(&mut self, from_idx: usize, to_idx: usize, count: usize) -> usize {
        let mut moved = 0;

        // Move from hot first (most volatile)
        let keys: Vec<String> = self.members[from_idx]
            .hot_entries
            .keys()
            .take(count)
            .cloned()
            .collect();

        for key in keys {
            if let Some(entry) = self.members[from_idx].hot_entries.remove(&key) {
                self.members[to_idx].hot_entries.insert(key.clone(), entry);
                self.key_locality.insert(key, to_idx);
                moved += 1;
            }
        }

        if moved >= count {
            return moved;
        }

        // Then warm if needed
        let remaining = count - moved;
        let keys: Vec<String> = self.members[from_idx]
            .warm_entries
            .keys()
            .take(remaining)
            .cloned()
            .collect();

        for key in keys {
            if let Some(entry) = self.members[from_idx].warm_entries.remove(&key) {
                self.members[to_idx].warm_entries.insert(key.clone(), entry);
                self.key_locality.insert(key, to_idx);
                moved += 1;
            }
        }

        moved
    }

    /// Reinforce an entry (strengthens, may promote)
    pub fn reinforce(&mut self, key: &str, amount: f32) -> bool {
        // Find which member has the key
        let target_idx = if let Some(&idx) = self.key_locality.get(key) {
            idx
        } else {
            // Search for it
            for (idx, member) in self.members.iter().enumerate() {
                if Self::key_exists_in_member(member, key) {
                    return self.members[idx].reinforce(key, amount).unwrap_or(false);
                }
            }
            return false;
        };

        if let Some(member) = self.members.get_mut(target_idx) {
            member.reinforce(key, amount).unwrap_or(false)
        } else {
            false
        }
    }

    /// Weaken an entry (may demote)
    pub fn weaken(&mut self, key: &str, amount: f32) -> bool {
        // Find which member has the key
        let target_idx = if let Some(&idx) = self.key_locality.get(key) {
            idx
        } else {
            for (idx, member) in self.members.iter().enumerate() {
                if Self::key_exists_in_member(member, key) {
                    return self.members[idx].weaken(key, amount).unwrap_or(false);
                }
            }
            return false;
        };

        if let Some(member) = self.members.get_mut(target_idx) {
            member.weaken(key, amount).unwrap_or(false)
        } else {
            false
        }
    }

    /// Apply decay to all members
    pub fn apply_decay(&mut self) {
        for member in &mut self.members {
            member.apply_decay();
        }
    }

    /// Get statistics for the colony
    pub fn stats(&self) -> ColonyStats {
        let member_stats: Vec<_> = self.members.iter().map(|m| m.stats()).collect();

        ColonyStats {
            member_count: self.members.len(),
            total_entries: member_stats.iter().map(|s| s.total_keys).sum(),
            hot_entries: member_stats.iter().map(|s| s.hot_entries).sum(),
            warm_entries: member_stats.iter().map(|s| s.warm_entries).sum(),
            cool_entries: member_stats.iter().map(|s| s.cool_entries).sum(),
            cold_entries: member_stats.iter().map(|s| s.cold_entries).sum(),
            split_count: self.metadata.split_count,
            merge_count: self.metadata.merge_count,
        }
    }
}

/// Statistics for a thermogram colony
#[derive(Debug, Clone, Default)]
pub struct ColonyStats {
    pub member_count: usize,
    pub total_entries: usize,
    pub hot_entries: usize,
    pub warm_entries: usize,
    pub cool_entries: usize,
    pub cold_entries: usize,
    pub split_count: usize,
    pub merge_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colony_creation() {
        let colony = ThermogramColony::new("test_colony", "Test Colony", PlasticityRule::stdp_like());

        assert_eq!(colony.member_count(), 1);
        assert_eq!(colony.total_entries(), 0);
    }

    #[test]
    fn test_colony_write_read() {
        let mut colony =
            ThermogramColony::new("test_colony", "Test Colony", PlasticityRule::stdp_like());

        let entry = ConsolidatedEntry {
            key: "key1".to_string(),
            value: vec![1, 2, 3],
            strength: 0.8,
            ternary_strength: None,
            updated_at: Utc::now(),
            update_count: 1,
        };

        colony.write("key1", entry.clone(), ThermalState::Hot);

        let read_entry = colony.read("key1");
        assert!(read_entry.is_some());
        assert_eq!(read_entry.unwrap().value, vec![1, 2, 3]);
    }

    #[test]
    fn test_colony_locality() {
        let mut colony =
            ThermogramColony::new("test_colony", "Test Colony", PlasticityRule::stdp_like());

        // Write entries with similar prefixes
        for i in 0..10 {
            let entry = ConsolidatedEntry {
                key: format!("group_a_{}", i),
                value: vec![i],
                strength: 0.8,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 1,
            };
            colony.write(&format!("group_a_{}", i), entry, ThermalState::Hot);
        }

        // All should be in same member due to locality
        let stats = colony.stats();
        assert_eq!(stats.member_count, 1);
        assert_eq!(stats.total_entries, 10);
    }

    #[test]
    fn test_colony_split() {
        let config = ColonyConfig {
            split_threshold: 5,
            merge_threshold: 1,
            max_members: 4,
            ..Default::default()
        };

        let mut colony = ThermogramColony::with_config(
            "test_colony",
            "Test Colony",
            PlasticityRule::stdp_like(),
            config,
        );

        // Add enough entries to trigger split
        for i in 0..10 {
            let entry = ConsolidatedEntry {
                key: format!("key_{}", i),
                value: vec![i],
                strength: 0.8,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 5,
            };
            colony.write(&format!("key_{}", i), entry, ThermalState::Hot);
        }

        // Consolidate should trigger split
        let result = colony.consolidate().unwrap();
        assert!(result.splits > 0 || colony.member_count() > 1);
    }

    #[test]
    fn test_colony_reinforce_weaken() {
        let mut colony =
            ThermogramColony::new("test_colony", "Test Colony", PlasticityRule::stdp_like());

        let entry = ConsolidatedEntry {
            key: "key1".to_string(),
            value: vec![1, 2, 3],
            strength: 0.5,
            ternary_strength: None,
            updated_at: Utc::now(),
            update_count: 1,
        };

        colony.write("key1", entry, ThermalState::Hot);

        // Reinforce
        assert!(colony.reinforce("key1", 0.2));
        let read = colony.read("key1").unwrap();
        assert!(read.strength > 0.5);

        // Weaken
        assert!(colony.weaken("key1", 0.1));
    }
}

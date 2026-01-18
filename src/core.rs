//! Core Thermogram implementation
//!
//! The main Thermogram type combining 4-temperature tensor states with hash-chained
//! deltas and plasticity rules.
//!
//! ## Thermal States (4-Temperature Model)
//!
//! - **Hot**: Working memory, volatile, fast decay (minutes)
//! - **Warm**: Session learning, medium decay (hours), persists to cartridge
//! - **Cool**: Expertise/skill memory, slow decay (days), long-term mastery
//! - **Cold**: Core identity, glacial decay (weeks+), personality backbone
//!
//! Bidirectional flow:
//! - Cement forward: reinforcement strengthens, promotes to colder layer
//! - Degrade backward: lack of use weakens, demotes to hotter layer

use crate::consolidation::{ConsolidatedEntry, ConsolidationPolicy, ConsolidationResult};
use crate::delta::{Delta, DeltaType};
use crate::error::Result;
use crate::hash_chain::HashChain;
use crate::plasticity::PlasticityRule;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Thermal state of a tensor entry (4-temperature model)
///
/// Each temperature has different decay rates and promotion/demotion thresholds.
/// Entries flow bidirectionally: cement forward when reinforced, degrade backward when unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Hash)]
pub enum ThermalState {
    /// Hot: Working memory, volatile, fast decay
    #[default]
    Hot,
    /// Warm: Session learning, medium decay, persists to cartridge
    Warm,
    /// Cool: Expertise/skill memory, slow decay, long-term
    Cool,
    /// Cold: Core identity, glacial decay, personality backbone
    Cold,
}

impl ThermalState {
    /// Get default decay rate for this temperature (per tick)
    pub fn default_decay_rate(&self) -> f32 {
        match self {
            Self::Hot => 0.1,     // 10% per tick (fast)
            Self::Warm => 0.01,   // 1% per tick (medium)
            Self::Cool => 0.001,  // 0.1% per tick (slow)
            Self::Cold => 0.0001, // 0.01% per tick (glacial)
        }
    }

    /// Get promotion threshold to next colder layer
    pub fn promotion_threshold(&self) -> f32 {
        match self {
            Self::Hot => 0.6,   // 60% to promote to Warm
            Self::Warm => 0.75, // 75% to promote to Cool
            Self::Cool => 0.9,  // 90% to promote to Cold
            Self::Cold => 1.0,  // Cannot promote further
        }
    }

    /// Get demotion threshold to next hotter layer
    pub fn demotion_threshold(&self) -> f32 {
        match self {
            Self::Hot => 0.0,  // Cannot demote further
            Self::Warm => 0.3, // Below 30% demotes to Hot
            Self::Cool => 0.4, // Below 40% demotes to Warm
            Self::Cold => 0.5, // Below 50% demotes to Cool
        }
    }

    /// Minimum observations before promotion eligible
    pub fn min_observations_for_promotion(&self) -> usize {
        match self {
            Self::Hot => 3,
            Self::Warm => 10,
            Self::Cool => 50,
            Self::Cold => usize::MAX,
        }
    }

    /// Next colder state (for promotion)
    pub fn colder(&self) -> Option<Self> {
        match self {
            Self::Hot => Some(Self::Warm),
            Self::Warm => Some(Self::Cool),
            Self::Cool => Some(Self::Cold),
            Self::Cold => None,
        }
    }

    /// Next hotter state (for demotion)
    pub fn hotter(&self) -> Option<Self> {
        match self {
            Self::Hot => None,
            Self::Warm => Some(Self::Hot),
            Self::Cool => Some(Self::Warm),
            Self::Cold => Some(Self::Cool),
        }
    }

    /// Get numeric index (for array access)
    pub fn index(&self) -> usize {
        match self {
            Self::Hot => 0,
            Self::Warm => 1,
            Self::Cool => 2,
            Self::Cold => 3,
        }
    }

    /// Create from numeric index
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Hot),
            1 => Some(Self::Warm),
            2 => Some(Self::Cool),
            3 => Some(Self::Cold),
            _ => None,
        }
    }

    /// All states in order from hottest to coldest
    pub fn all() -> [Self; 4] {
        [Self::Hot, Self::Warm, Self::Cool, Self::Cold]
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cool => "cool",
            Self::Cold => "cold",
        }
    }
}

/// Configuration for 4-temperature thermal transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Decay rates per temperature layer [hot, warm, cool, cold]
    pub decay_rates: [f32; 4],

    /// Promotion thresholds (strength needed to promote) [hot, warm, cool, cold]
    pub promotion_thresholds: [f32; 4],

    /// Demotion thresholds (strength below which demotion occurs) [hot, warm, cool, cold]
    pub demotion_thresholds: [f32; 4],

    /// Minimum observations before promotion eligible [hot, warm, cool, cold]
    pub min_observations: [usize; 4],

    /// Whether each layer can be demoted [hot, warm, cool, cold]
    pub allow_demotion: [bool; 4],

    /// Strength below which entries are pruned entirely
    pub prune_threshold: f32,

    // Legacy fields for backward compatibility
    /// Legacy: Minimum strength to crystallize (mapped to cool→cold promotion)
    #[serde(default = "default_crystallization_threshold")]
    pub crystallization_threshold: f32,
    /// Legacy: Whether warming is allowed
    #[serde(default = "default_allow_warming")]
    pub allow_warming: bool,
    /// Legacy: Warming delta
    #[serde(default = "default_warming_delta")]
    pub warming_delta: f32,
}

fn default_crystallization_threshold() -> f32 { 0.75 }
fn default_allow_warming() -> bool { true }
fn default_warming_delta() -> f32 { 0.3 }

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            decay_rates: [0.1, 0.01, 0.001, 0.0001],
            promotion_thresholds: [0.6, 0.75, 0.9, 1.0],
            demotion_thresholds: [0.0, 0.3, 0.4, 0.5],
            min_observations: [3, 10, 50, usize::MAX],
            allow_demotion: [false, true, true, true],
            prune_threshold: 0.05,
            // Legacy
            crystallization_threshold: 0.75,
            allow_warming: true,
            warming_delta: 0.3,
        }
    }
}

impl ThermalConfig {
    /// Create config optimized for fast learners (agents, workers)
    pub fn fast_learner() -> Self {
        Self {
            decay_rates: [0.05, 0.005, 0.0005, 0.00005],
            promotion_thresholds: [0.5, 0.65, 0.85, 1.0],
            demotion_thresholds: [0.0, 0.2, 0.3, 0.4],
            min_observations: [2, 5, 20, usize::MAX],
            allow_demotion: [false, true, true, false], // Cold doesn't demote
            prune_threshold: 0.03,
            crystallization_threshold: 0.85,
            allow_warming: true,
            warming_delta: 0.2,
        }
    }

    /// Create config optimized for organic emergence (gradual learning)
    pub fn organic() -> Self {
        Self {
            decay_rates: [0.1, 0.01, 0.001, 0.0001],
            promotion_thresholds: [0.7, 0.8, 0.95, 1.0],
            demotion_thresholds: [0.0, 0.25, 0.35, 0.45],
            min_observations: [5, 15, 100, usize::MAX],
            allow_demotion: [false, true, true, true],
            prune_threshold: 0.05,
            crystallization_threshold: 0.95,
            allow_warming: true,
            warming_delta: 0.3,
        }
    }

    /// Get decay rate for a thermal state
    pub fn decay_rate(&self, state: ThermalState) -> f32 {
        self.decay_rates[state.index()]
    }

    /// Get promotion threshold for a thermal state
    pub fn promotion_threshold(&self, state: ThermalState) -> f32 {
        self.promotion_thresholds[state.index()]
    }

    /// Get demotion threshold for a thermal state
    pub fn demotion_threshold(&self, state: ThermalState) -> f32 {
        self.demotion_thresholds[state.index()]
    }

    /// Get min observations for a thermal state
    pub fn min_obs(&self, state: ThermalState) -> usize {
        self.min_observations[state.index()]
    }

    /// Check if demotion is allowed for a thermal state
    pub fn can_demote(&self, state: ThermalState) -> bool {
        self.allow_demotion[state.index()]
    }
}

/// A Thermogram - plastic memory capsule with 4-temperature tensor states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thermogram {
    /// Unique ID for this thermogram
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Hot tensors (working memory, volatile, fast decay)
    #[serde(default)]
    pub hot_entries: HashMap<String, ConsolidatedEntry>,

    /// Warm tensors (session learning, medium decay)
    #[serde(default)]
    pub warm_entries: HashMap<String, ConsolidatedEntry>,

    /// Cool tensors (expertise/skill, slow decay)
    #[serde(default)]
    pub cool_entries: HashMap<String, ConsolidatedEntry>,

    /// Cold tensors (core identity, glacial decay)
    #[serde(default)]
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
            warm_entries: HashMap::new(),
            cool_entries: HashMap::new(),
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

    /// Create for fast-learning agents (optimized config)
    pub fn for_fast_learner(name: impl Into<String>, plasticity_rule: PlasticityRule) -> Self {
        Self::with_thermal_config(name, plasticity_rule, ThermalConfig::fast_learner())
    }

    /// Create for organic/gradual learning (emergence-focused config)
    pub fn for_organic(name: impl Into<String>, plasticity_rule: PlasticityRule) -> Self {
        Self::with_thermal_config(name, plasticity_rule, ThermalConfig::organic())
    }

    /// Get entries map for a thermal state (immutable)
    pub fn entries(&self, state: ThermalState) -> &HashMap<String, ConsolidatedEntry> {
        match state {
            ThermalState::Hot => &self.hot_entries,
            ThermalState::Warm => &self.warm_entries,
            ThermalState::Cool => &self.cool_entries,
            ThermalState::Cold => &self.cold_entries,
        }
    }

    /// Get entries map for a thermal state (mutable)
    pub fn entries_mut(&mut self, state: ThermalState) -> &mut HashMap<String, ConsolidatedEntry> {
        match state {
            ThermalState::Hot => &mut self.hot_entries,
            ThermalState::Warm => &mut self.warm_entries,
            ThermalState::Cool => &mut self.cool_entries,
            ThermalState::Cold => &mut self.cold_entries,
        }
    }

    /// Get total entry count across all layers
    pub fn total_entries(&self) -> usize {
        self.hot_entries.len()
            + self.warm_entries.len()
            + self.cool_entries.len()
            + self.cold_entries.len()
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

    /// Read current value for a key (dirty → hot → warm → cool → cold priority)
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

        // Check all thermal layers in order (hot → warm → cool → cold)
        for state in ThermalState::all() {
            if let Some(entry) = self.entries(state).get(key) {
                return Ok(Some(entry.value.clone()));
            }
        }

        Ok(None)
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

        // Check all thermal layers
        for state in ThermalState::all() {
            if let Some(entry) = self.entries(state).get(key) {
                return Ok(Some((entry.value.clone(), entry.strength)));
            }
        }

        Ok(None)
    }

    /// Read with full thermal state information
    pub fn read_with_state(&self, key: &str) -> Result<Option<(Vec<u8>, f32, ThermalState)>> {
        // Check dirty state first (treated as Hot)
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

        // Check all thermal layers
        for state in ThermalState::all() {
            if let Some(entry) = self.entries(state).get(key) {
                return Ok(Some((entry.value.clone(), entry.strength, state)));
            }
        }

        Ok(None)
    }

    /// List all keys in current state (all layers + dirty)
    pub fn keys(&self) -> Vec<String> {
        let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Add from all layers
        for state in ThermalState::all() {
            for key in self.entries(state).keys() {
                keys.insert(key.clone());
            }
        }

        // Add dirty keys (excluding deletes)
        for delta in &self.dirty_chain.deltas {
            if delta.delta_type != DeltaType::Delete {
                keys.insert(delta.key.clone());
            }
        }

        keys.into_iter().collect()
    }

    /// List keys for a specific thermal state
    pub fn keys_for_state(&self, state: ThermalState) -> Vec<String> {
        self.entries(state).keys().cloned().collect()
    }

    /// List only hot tensor keys
    pub fn hot_keys(&self) -> Vec<String> {
        self.keys_for_state(ThermalState::Hot)
    }

    /// List only warm tensor keys
    pub fn warm_keys(&self) -> Vec<String> {
        self.keys_for_state(ThermalState::Warm)
    }

    /// List only cool tensor keys
    pub fn cool_keys(&self) -> Vec<String> {
        self.keys_for_state(ThermalState::Cool)
    }

    /// List only cold tensor keys
    pub fn cold_keys(&self) -> Vec<String> {
        self.keys_for_state(ThermalState::Cold)
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

    /// Crystallize high-confidence hot entries to cold layer (legacy method)
    ///
    /// Note: This is a legacy 2-temperature method. For 4-temperature transitions,
    /// use run_thermal_transitions() instead. This method skips warm/cool layers.
    ///
    /// Entries are crystallized when:
    /// - Strength >= crystallization_threshold
    /// - Update count >= min_observations[0] (hot layer)
    pub fn crystallize(&mut self) -> Result<CrystallizationResult> {
        let mut result = CrystallizationResult::default();
        let mut keys_to_crystallize = Vec::new();

        // Find eligible entries (using legacy crystallization_threshold and hot min_observations)
        let min_obs = self.thermal_config.min_observations[ThermalState::Hot.index()];
        for (key, entry) in &self.hot_entries {
            if entry.strength >= self.thermal_config.crystallization_threshold
                && entry.update_count >= min_obs
            {
                keys_to_crystallize.push(key.clone());
            }
        }

        // Move to cold layer (skipping warm/cool for legacy behavior)
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

    /// Run 4-temperature thermal transitions (promotion and demotion)
    ///
    /// Promotes strong entries to colder layers, demotes weak entries to hotter layers.
    /// This is the preferred method for 4-temperature thermograms.
    pub fn run_thermal_transitions(&mut self) -> Result<()> {
        // Hot → Warm promotions
        self.promote_layer(ThermalState::Hot, ThermalState::Warm)?;

        // Warm → Cool promotions
        self.promote_layer(ThermalState::Warm, ThermalState::Cool)?;

        // Cool → Cold promotions
        self.promote_layer(ThermalState::Cool, ThermalState::Cold)?;

        // Demotions (if allowed)
        if self.thermal_config.can_demote(ThermalState::Cold) {
            self.demote_layer(ThermalState::Cold, ThermalState::Cool)?;
        }
        if self.thermal_config.can_demote(ThermalState::Cool) {
            self.demote_layer(ThermalState::Cool, ThermalState::Warm)?;
        }
        if self.thermal_config.can_demote(ThermalState::Warm) {
            self.demote_layer(ThermalState::Warm, ThermalState::Hot)?;
        }

        // Prune very weak entries
        self.prune_all_layers()?;

        Ok(())
    }

    /// Promote entries from one layer to the next colder layer
    fn promote_layer(&mut self, from: ThermalState, to: ThermalState) -> Result<usize> {
        let threshold = self.thermal_config.promotion_threshold(from);
        let min_obs = self.thermal_config.min_obs(from);

        let keys_to_promote: Vec<String> = self
            .entries(from)
            .iter()
            .filter(|(_, entry)| entry.strength >= threshold && entry.update_count >= min_obs)
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_promote.len();
        for key in keys_to_promote {
            if let Some(entry) = self.entries_mut(from).remove(&key) {
                if let Some(existing) = self.entries_mut(to).get_mut(&key) {
                    existing.strength = existing.strength * 0.3 + entry.strength * 0.7;
                    existing.value = entry.value;
                    existing.updated_at = entry.updated_at;
                    existing.update_count += entry.update_count;
                } else {
                    self.entries_mut(to).insert(key, entry);
                }
            }
        }
        Ok(count)
    }

    /// Demote entries from one layer to the next hotter layer
    fn demote_layer(&mut self, from: ThermalState, to: ThermalState) -> Result<usize> {
        let threshold = self.thermal_config.demotion_threshold(from);

        let keys_to_demote: Vec<String> = self
            .entries(from)
            .iter()
            .filter(|(_, entry)| entry.strength < threshold)
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_demote.len();
        for key in keys_to_demote {
            if let Some(mut entry) = self.entries_mut(from).remove(&key) {
                entry.strength = (entry.strength * 0.95).max(0.01);
                entry.updated_at = Utc::now();
                if let Some(existing) = self.entries_mut(to).get_mut(&key) {
                    existing.strength = existing.strength.max(entry.strength);
                    existing.updated_at = entry.updated_at;
                } else {
                    self.entries_mut(to).insert(key, entry);
                }
            }
        }
        Ok(count)
    }

    /// Prune very weak entries from all layers
    fn prune_all_layers(&mut self) -> Result<usize> {
        let threshold = self.thermal_config.prune_threshold;
        let mut total = 0;
        for state in ThermalState::all() {
            let before = self.entries(state).len();
            self.entries_mut(state).retain(|_, e| e.strength >= threshold);
            total += before - self.entries(state).len();
        }
        Ok(total)
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
            warm_entries: self.warm_entries.len(),
            cool_entries: self.cool_entries.len(),
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
        let mut total = 0;

        for state in ThermalState::all() {
            total += self
                .entries(state)
                .values()
                .map(|e| e.value.len() + e.key.len() + 100)
                .sum::<usize>();
        }

        total + self.estimate_dirty_size()
    }

    /// Apply decay to all entries based on their thermal state
    pub fn apply_decay(&mut self) {
        for state in ThermalState::all() {
            let decay_rate = self.thermal_config.decay_rate(state);
            for entry in self.entries_mut(state).values_mut() {
                entry.strength = (entry.strength * (1.0 - decay_rate)).max(0.0);
            }
        }
    }

    /// Reinforce an entry (increase strength)
    pub fn reinforce(&mut self, key: &str, amount: f32) -> Result<bool> {
        for state in ThermalState::all() {
            if let Some(entry) = self.entries_mut(state).get_mut(key) {
                entry.strength = (entry.strength + amount).min(1.0);
                entry.update_count += 1;
                entry.updated_at = Utc::now();
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Weaken an entry (decrease strength)
    pub fn weaken(&mut self, key: &str, amount: f32) -> Result<bool> {
        for state in ThermalState::all() {
            if let Some(entry) = self.entries_mut(state).get_mut(key) {
                entry.strength = (entry.strength - amount).max(0.0);
                entry.updated_at = Utc::now();
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Statistics about a Thermogram
#[derive(Debug, Clone)]
pub struct ThermogramStats {
    pub total_keys: usize,
    pub hot_entries: usize,
    pub warm_entries: usize,
    pub cool_entries: usize,
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
        thermo.thermal_config.min_observations[0] = 2; // Hot layer min_observations

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

        // Should have crystallized to cold (legacy behavior skips warm/cool)
        assert_eq!(thermo.cold_entries.len(), 1);
        assert!(thermo.hot_entries.is_empty());
    }

    #[test]
    fn test_4temp_promotion() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        // Set low thresholds for easy promotion
        thermo.thermal_config.promotion_thresholds = [0.5, 0.5, 0.5, 1.0];
        thermo.thermal_config.min_observations = [1, 1, 1, usize::MAX];

        // Add high-strength entry to hot
        thermo.hot_entries.insert(
            "key1".to_string(),
            ConsolidatedEntry {
                key: "key1".to_string(),
                value: b"value1".to_vec(),
                strength: 0.8,
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 5,
            },
        );

        // Run transitions
        thermo.run_thermal_transitions().unwrap();

        // Should have promoted through all layers to cold
        assert!(thermo.hot_entries.is_empty());
        assert!(thermo.warm_entries.is_empty());
        assert!(thermo.cool_entries.is_empty());
        assert_eq!(thermo.cold_entries.len(), 1);
    }

    #[test]
    fn test_4temp_demotion() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        // Set thresholds: cold demotes at 0.5, others don't demote (threshold 0.0)
        thermo.thermal_config.demotion_thresholds = [0.0, 0.0, 0.0, 0.5];
        thermo.thermal_config.allow_demotion = [false, true, true, true];

        // Add weak entry to cold
        thermo.cold_entries.insert(
            "key1".to_string(),
            ConsolidatedEntry {
                key: "key1".to_string(),
                value: b"value1".to_vec(),
                strength: 0.3, // Below cold's demotion threshold (0.5)
                ternary_strength: None,
                updated_at: Utc::now(),
                update_count: 1,
            },
        );

        // Run transitions
        thermo.run_thermal_transitions().unwrap();

        // Should have demoted one level to cool (other layers don't demote with threshold 0.0)
        assert!(thermo.cold_entries.is_empty());
        assert_eq!(thermo.cool_entries.len(), 1);
        // Entry should exist with slightly reduced strength
        let entry = thermo.cool_entries.get("key1").unwrap();
        assert!(entry.strength < 0.3);
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

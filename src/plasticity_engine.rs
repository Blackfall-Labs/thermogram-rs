//! Plasticity Engine - SNN-based delta generation
//!
//! The plasticity engine is a small SNN that "stirs" the connections in a Thermogram,
//! generating deltas based on spiking dynamics, STDP, homeostasis, and competition.
//!
//! Each Thermogram has its own embedded SNN with independent neuromodulator state.
//! Applications can coordinate multiple SNNs by passing shared neuromodulator values.

use crate::delta::Delta;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Neuromodulator state (can be synced from central coordinator or run independently)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodState {
    /// Dopamine: reward signal, learning rate modulation (0.0 - 1.0)
    pub dopamine: f32,

    /// Serotonin: mood/confidence, decay rate modulation (0.0 - 1.0)
    pub serotonin: f32,

    /// Norepinephrine: arousal/attention, competition strength (0.0 - 1.0)
    pub norepinephrine: f32,

    /// Acetylcholine: attention/focus, gating modulation (0.0 - 1.0)
    pub acetylcholine: f32,
}

impl Default for NeuromodState {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
        }
    }
}

impl NeuromodState {
    /// Create baseline state
    pub fn baseline() -> Self {
        Self::default()
    }

    /// Reward signal (increase dopamine)
    pub fn reward(&mut self, amount: f32) {
        self.dopamine = (self.dopamine + amount).clamp(0.0, 1.0);
    }

    /// Stress signal (decrease serotonin, increase norepinephrine)
    pub fn stress(&mut self, amount: f32) {
        self.serotonin = (self.serotonin - amount).clamp(0.0, 1.0);
        self.norepinephrine = (self.norepinephrine + amount).clamp(0.0, 1.0);
    }

    /// Focus signal (increase acetylcholine)
    pub fn focus(&mut self, amount: f32) {
        self.acetylcholine = (self.acetylcholine + amount).clamp(0.0, 1.0);
    }

    /// Decay over time (return to baseline)
    pub fn decay(&mut self, rate: f32) {
        let baseline = Self::baseline();
        self.dopamine += (baseline.dopamine - self.dopamine) * rate;
        self.serotonin += (baseline.serotonin - self.serotonin) * rate;
        self.norepinephrine += (baseline.norepinephrine - self.norepinephrine) * rate;
        self.acetylcholine += (baseline.acetylcholine - self.acetylcholine) * rate;
    }
}

/// Trait for plasticity engines that generate deltas
pub trait PlasticityEngine: Send + Sync {
    /// Process an activation vector and generate deltas
    ///
    /// Takes:
    /// - `activation`: Input activation vector from meshes/mining
    /// - `neuromod`: Current neuromodulator state
    ///
    /// Returns: Vector of deltas to apply to the Thermogram
    fn process(&mut self, activation: &[f32], neuromod: &NeuromodState) -> Result<Vec<Delta>>;

    /// Update internal state based on neuromodulation
    fn sync_neuromod(&mut self, neuromod: &NeuromodState);

    /// Get current internal state for serialization
    fn state(&self) -> PlasticityEngineState;

    /// Restore from saved state
    fn restore(&mut self, state: &PlasticityEngineState) -> Result<()>;
}

/// Serializable state of a plasticity engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityEngineState {
    /// Engine type identifier
    pub engine_type: String,

    /// Neuromodulator state
    pub neuromod: NeuromodState,

    /// Engine-specific state (serialized)
    pub custom_state: Vec<u8>,
}

/// Configuration for neuromodulation sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodSyncConfig {
    /// Whether to sync from external source
    pub enabled: bool,

    /// Sync rate (how much external state influences local state)
    pub sync_rate: f32,

    /// Minimum time between syncs (milliseconds)
    pub sync_interval_ms: u64,
}

impl Default for NeuromodSyncConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_rate: 0.1, // 10% influence from external
            sync_interval_ms: 100, // Sync every 100ms
        }
    }
}

impl NeuromodSyncConfig {
    /// No syncing - fully independent
    pub fn independent() -> Self {
        Self {
            enabled: false,
            sync_rate: 0.0,
            sync_interval_ms: 0,
        }
    }

    /// Full sync - completely driven by external state
    pub fn full_sync() -> Self {
        Self {
            enabled: true,
            sync_rate: 1.0,
            sync_interval_ms: 50,
        }
    }

    /// Apply sync between local and external neuromod states
    pub fn apply_sync(&self, local: &mut NeuromodState, external: &NeuromodState) {
        if !self.enabled {
            return;
        }

        let rate = self.sync_rate;
        local.dopamine += (external.dopamine - local.dopamine) * rate;
        local.serotonin += (external.serotonin - local.serotonin) * rate;
        local.norepinephrine += (external.norepinephrine - local.norepinephrine) * rate;
        local.acetylcholine += (external.acetylcholine - local.acetylcholine) * rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromod_reward() {
        let mut state = NeuromodState::baseline();
        state.reward(0.3);

        assert!(state.dopamine > 0.5);
    }

    #[test]
    fn test_neuromod_stress() {
        let mut state = NeuromodState::baseline();
        state.stress(0.2);

        assert!(state.serotonin < 0.5);
        assert!(state.norepinephrine > 0.5);
    }

    #[test]
    fn test_neuromod_decay() {
        let mut state = NeuromodState::baseline();
        state.dopamine = 1.0;

        state.decay(0.1);

        assert!(state.dopamine < 1.0);
        assert!(state.dopamine > 0.5); // Moving toward baseline
    }

    #[test]
    fn test_sync_config() {
        let config = NeuromodSyncConfig::default();
        let mut local = NeuromodState::baseline();
        let mut external = NeuromodState::baseline();
        external.dopamine = 1.0;

        config.apply_sync(&mut local, &external);

        // Should move toward external but not fully
        assert!(local.dopamine > 0.5);
        assert!(local.dopamine < 1.0);
    }

    #[test]
    fn test_independent_no_sync() {
        let config = NeuromodSyncConfig::independent();
        let mut local = NeuromodState::baseline();
        let mut external = NeuromodState::baseline();
        external.dopamine = 1.0;

        config.apply_sync(&mut local, &external);

        // Should not change
        assert_eq!(local.dopamine, 0.5);
    }
}

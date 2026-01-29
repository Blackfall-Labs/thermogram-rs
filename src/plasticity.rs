//! Plasticity Rules - STDP-like policies for memory updates
//!
//! Inspired by Spike-Timing Dependent Plasticity (STDP) in neuroscience,
//! plasticity rules determine when to:
//! - Update existing memory (strengthen/weaken)
//! - Create new memory (novelty threshold exceeded)
//! - Merge memories (similar patterns detected)
//! - Prune memories (decay below threshold)
//!
//! ## Signal-Native Plasticity
//!
//! All thresholds and rates use `Signal` (2-byte polarity + magnitude).
//! Strength updates operate on Signal magnitudes using integer arithmetic.
//!
//! ## Ternary Plasticity
//!
//! For ternary weights (+1, 0, -1), plasticity works via discrete state transitions:
//! - **Strengthen**: Neg→Zero→Pos
//! - **Weaken**: Pos→Zero→Neg
//! - **Prune**: Only Zero weights are prunable
//!
//! This enables BitNet-style quantized neural networks with STDP learning.

use serde::{Deserialize, Serialize};
use ternary_signal::Signal;

use crate::ternary::TernaryWeight;

/// Plasticity rule governing memory updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityRule {
    /// Update policy
    pub policy: UpdatePolicy,

    /// Novelty threshold — magnitude above which a new memory is created
    pub novelty_threshold: Signal,

    /// Merge threshold — similarity above (1.0 - threshold) triggers merge
    pub merge_threshold: Signal,

    /// Decay rate per consolidation cycle
    pub decay_rate: Signal,

    /// Minimum strength to keep (prune below this)
    pub prune_threshold: Signal,

    /// Learning rate for updates
    pub learning_rate: Signal,
}

/// Policy for how updates are applied
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdatePolicy {
    /// STDP-like: Strengthen recent, weaken old
    STDP,

    /// Always replace (no plasticity)
    Replace,

    /// Exponential moving average
    EMA,

    /// Bayesian update (weighted by confidence)
    Bayesian,

    /// Winner-take-all (strongest wins)
    WTA,
}

impl PlasticityRule {
    /// STDP-like plasticity (default for neural memory)
    pub fn stdp_like() -> Self {
        Self {
            policy: UpdatePolicy::STDP,
            novelty_threshold: Signal::positive(153),  // ~0.6
            merge_threshold: Signal::positive(77),     // ~0.3
            decay_rate: Signal::positive(3),           // ~0.01
            prune_threshold: Signal::positive(26),     // ~0.1
            learning_rate: Signal::positive(26),       // ~0.1
        }
    }

    /// Conservative plasticity (high novelty threshold, slow updates)
    pub fn conservative() -> Self {
        Self {
            policy: UpdatePolicy::EMA,
            novelty_threshold: Signal::positive(204),  // ~0.8
            merge_threshold: Signal::positive(102),    // ~0.4
            decay_rate: Signal::positive(1),           // ~0.005
            prune_threshold: Signal::positive(13),     // ~0.05
            learning_rate: Signal::positive(13),       // ~0.05
        }
    }

    /// Aggressive plasticity (low novelty threshold, fast updates)
    pub fn aggressive() -> Self {
        Self {
            policy: UpdatePolicy::STDP,
            novelty_threshold: Signal::positive(102),  // ~0.4
            merge_threshold: Signal::positive(51),     // ~0.2
            decay_rate: Signal::positive(5),           // ~0.02
            prune_threshold: Signal::positive(51),     // ~0.2
            learning_rate: Signal::positive(51),       // ~0.2
        }
    }

    /// No plasticity (simple replacement)
    pub fn replace_only() -> Self {
        Self {
            policy: UpdatePolicy::Replace,
            novelty_threshold: Signal::positive(255),  // 1.0 - never create new
            merge_threshold: Signal::positive(0),      // 0.0 - never merge
            decay_rate: Signal::positive(0),           // 0.0 - no decay
            prune_threshold: Signal::positive(0),      // 0.0 - never prune
            learning_rate: Signal::positive(255),      // 1.0 - full replacement
        }
    }

    /// Bayesian update (confidence-weighted)
    pub fn bayesian() -> Self {
        Self {
            policy: UpdatePolicy::Bayesian,
            novelty_threshold: Signal::positive(179),  // ~0.7
            merge_threshold: Signal::positive(77),     // ~0.3
            decay_rate: Signal::positive(3),           // ~0.01
            prune_threshold: Signal::positive(26),     // ~0.1
            learning_rate: Signal::positive(26),       // ~0.1
        }
    }

    /// Apply this rule to decide how to update
    ///
    /// Returns the new strength for the memory based on:
    /// - Current strength (Signal)
    /// - New observation strength (Signal)
    /// - Time since last update (seconds)
    ///
    /// All arithmetic uses Signal magnitudes. The result preserves
    /// the polarity of the stronger contributor.
    pub fn apply_update(
        &self,
        current_strength: Signal,
        new_strength: Signal,
        time_delta_seconds: f64,
    ) -> Signal {
        let cur_f = current_strength.magnitude_f32();
        let new_f = new_strength.magnitude_f32();
        let decay_f = self.decay_rate.magnitude_f32();
        let lr_f = self.learning_rate.magnitude_f32();

        let result_f = match self.policy {
            UpdatePolicy::STDP => {
                // Decay old, strengthen with new
                let decayed = cur_f * (1.0 - decay_f * time_delta_seconds as f32 / 86400.0);
                let updated = decayed + lr_f * new_f;
                updated.clamp(0.0, 1.0)
            }

            UpdatePolicy::Replace => new_f,

            UpdatePolicy::EMA => {
                // Exponential moving average
                lr_f * new_f + (1.0 - lr_f) * cur_f
            }

            UpdatePolicy::Bayesian => {
                // Weighted average by confidence
                let total_weight = cur_f + new_f;
                if total_weight > 0.0 {
                    (cur_f * cur_f + new_f * new_f) / total_weight
                } else {
                    0.0
                }
            }

            UpdatePolicy::WTA => {
                // Winner takes all
                cur_f.max(new_f)
            }
        };

        // Preserve polarity: use new_strength polarity if it's stronger,
        // otherwise keep current polarity
        let polarity = if new_f >= cur_f {
            new_strength.polarity
        } else {
            current_strength.polarity
        };

        Signal::new(polarity, (result_f * 255.0) as u8)
    }

    /// Should this observation create a new memory?
    pub fn should_create_new(&self, novelty_score: Signal) -> bool {
        novelty_score.magnitude > self.novelty_threshold.magnitude
    }

    /// Should this observation merge with existing?
    /// Takes similarity as a Signal magnitude (higher = more similar)
    pub fn should_merge(&self, similarity: Signal) -> bool {
        // Merge when similarity > (1.0 - threshold)
        let anti_threshold = 255 - self.merge_threshold.magnitude;
        similarity.magnitude > anti_threshold
    }

    /// Should this memory be pruned?
    pub fn should_prune_signal(&self, strength: Signal) -> bool {
        strength.magnitude < self.prune_threshold.magnitude
    }

    // =========================================================================
    // TERNARY PLASTICITY - Discrete state transitions for quantized weights
    // =========================================================================

    /// Apply ternary update based on policy
    ///
    /// For ternary weights, updates happen via discrete state transitions:
    /// - **Strengthen**: Neg→Zero→Pos (when confident observation agrees)
    /// - **Weaken**: Pos→Zero→Neg (when confident observation disagrees)
    /// - **No change**: When confidence is too low
    ///
    /// The `confidence` parameter (0.0-1.0) determines transition probability.
    pub fn apply_ternary_update(
        &self,
        current: TernaryWeight,
        observed: TernaryWeight,
        confidence: f32,
    ) -> TernaryWeight {
        match self.policy {
            UpdatePolicy::STDP => {
                if current == observed {
                    if confidence > 0.7 {
                        current.strengthen()
                    } else {
                        current
                    }
                } else if confidence > 0.5 {
                    self.move_toward(current, observed)
                } else {
                    self.decay_toward_zero(current)
                }
            }

            UpdatePolicy::Replace => {
                if confidence > 0.5 {
                    observed
                } else {
                    current
                }
            }

            UpdatePolicy::EMA => {
                if current == observed {
                    current
                } else if confidence > self.learning_rate.magnitude_f32() {
                    observed
                } else {
                    current
                }
            }

            UpdatePolicy::Bayesian => {
                if confidence > 0.7 {
                    observed
                } else if confidence > 0.3 {
                    self.decay_toward_zero(current)
                } else {
                    current
                }
            }

            UpdatePolicy::WTA => {
                if confidence > 0.5 {
                    observed
                } else {
                    current
                }
            }
        }
    }

    /// Move current weight one step toward target
    fn move_toward(&self, current: TernaryWeight, target: TernaryWeight) -> TernaryWeight {
        use TernaryWeight::*;
        match (current, target) {
            (Pos, Pos) | (Zero, Zero) | (Neg, Neg) => current,
            (Zero, Pos) | (Neg, Pos) => current.strengthen(),
            (Zero, Neg) | (Pos, Neg) => current.weaken(),
            (Pos, Zero) => current.weaken(),
            (Neg, Zero) => current.strengthen(),
        }
    }

    /// Decay weight toward zero (for low-confidence or aging)
    fn decay_toward_zero(&self, current: TernaryWeight) -> TernaryWeight {
        match current {
            TernaryWeight::Pos => TernaryWeight::Zero,
            TernaryWeight::Neg => TernaryWeight::Zero,
            TernaryWeight::Zero => TernaryWeight::Zero,
        }
    }

    /// Public method to decay weight toward zero
    /// Used by embedded_snn for ternary weight aging
    pub fn decay_toward_zero_public(&self, current: TernaryWeight) -> TernaryWeight {
        self.decay_toward_zero(current)
    }

    /// Should this ternary weight be pruned?
    ///
    /// For ternary weights, only Zero weights are prunable (inactive connections).
    pub fn should_prune_ternary(&self, weight: TernaryWeight) -> bool {
        weight == TernaryWeight::Zero
    }

    /// Apply STDP-style ternary update for co-firing neurons
    ///
    /// Pre→Post firing (causal): strengthen connection
    /// Post→Pre firing (anti-causal): weaken connection
    pub fn apply_ternary_stdp(
        &self,
        current: TernaryWeight,
        pre_fired: bool,
        post_fired: bool,
        time_delta_ms: i64,
    ) -> TernaryWeight {
        if pre_fired && post_fired {
            if time_delta_ms > 0 {
                current.strengthen()
            } else if time_delta_ms < 0 {
                current.weaken()
            } else {
                if current == TernaryWeight::Neg {
                    TernaryWeight::Zero
                } else {
                    current
                }
            }
        } else {
            current
        }
    }

    /// Ternary majority voting for merge decisions
    ///
    /// Given multiple ternary weights, return the majority value.
    pub fn ternary_majority_vote(weights: &[TernaryWeight]) -> TernaryWeight {
        let mut pos_count = 0i32;
        let mut neg_count = 0i32;

        for &w in weights {
            match w {
                TernaryWeight::Pos => pos_count += 1,
                TernaryWeight::Neg => neg_count += 1,
                TernaryWeight::Zero => {}
            }
        }

        if pos_count > neg_count {
            TernaryWeight::Pos
        } else if neg_count > pos_count {
            TernaryWeight::Neg
        } else {
            TernaryWeight::Zero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_update() {
        let rule = PlasticityRule::stdp_like();

        let current = Signal::positive(128); // ~0.5
        let new = Signal::positive(204);     // ~0.8
        let time_delta = 0.0; // Immediate

        let updated = rule.apply_update(current, new, time_delta);

        // Should strengthen (learning_rate * new adds to current)
        assert!(updated.magnitude > current.magnitude);
        assert!(updated.magnitude <= 255);
    }

    #[test]
    fn test_novelty_threshold() {
        let rule = PlasticityRule::stdp_like();

        assert!(rule.should_create_new(Signal::positive(179)));  // ~0.7, above 0.6 threshold
        assert!(!rule.should_create_new(Signal::positive(128))); // ~0.5, below 0.6 threshold
    }

    #[test]
    fn test_merge_decision() {
        let rule = PlasticityRule::stdp_like();

        // merge_threshold is ~0.3, so merge when similarity > 0.7 (255 - 77 = 178)
        assert!(rule.should_merge(Signal::positive(204)));  // High similarity
        assert!(!rule.should_merge(Signal::positive(153))); // Low similarity
    }

    #[test]
    fn test_prune_decision() {
        let rule = PlasticityRule::stdp_like();

        assert!(rule.should_prune_signal(Signal::positive(13)));  // Below threshold (~0.1 = 26)
        assert!(!rule.should_prune_signal(Signal::positive(128))); // Above threshold
    }

    #[test]
    fn test_ema_update() {
        let rule = PlasticityRule {
            policy: UpdatePolicy::EMA,
            learning_rate: Signal::positive(26), // ~0.1
            ..PlasticityRule::stdp_like()
        };

        let current = Signal::positive(128); // ~0.5
        let new = Signal::positive(255);     // ~1.0

        let updated = rule.apply_update(current, new, 0.0);

        // Should be weighted average: 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        // magnitude ≈ 140
        let expected_f = 0.1 * 1.0 + 0.9 * (128.0 / 255.0);
        let expected_mag = (expected_f * 255.0) as u8;
        // Allow some rounding tolerance
        assert!((updated.magnitude as i16 - expected_mag as i16).unsigned_abs() <= 3);
    }
}

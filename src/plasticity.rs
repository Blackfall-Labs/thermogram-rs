//! Plasticity Rules - STDP-like policies for memory updates
//!
//! Inspired by Spike-Timing Dependent Plasticity (STDP) in neuroscience,
//! plasticity rules determine when to:
//! - Update existing memory (strengthen/weaken)
//! - Create new memory (novelty threshold exceeded)
//! - Merge memories (similar patterns detected)
//! - Prune memories (decay below threshold)
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

use crate::ternary::TernaryWeight;

/// Plasticity rule governing memory updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityRule {
    /// Update policy
    pub policy: UpdatePolicy,

    /// Novelty threshold (cosine distance) - above this = create new
    pub novelty_threshold: f32,

    /// Merge threshold - below this = merge with existing
    pub merge_threshold: f32,

    /// Decay rate per consolidation cycle
    pub decay_rate: f32,

    /// Minimum strength to keep (prune below this)
    pub prune_threshold: f32,

    /// Learning rate for updates
    pub learning_rate: f32,
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
            novelty_threshold: 0.6,
            merge_threshold: 0.3,
            decay_rate: 0.01,        // 1% decay per cycle
            prune_threshold: 0.1,     // Prune if strength < 10%
            learning_rate: 0.1,       // 10% weight on new observations
        }
    }

    /// Conservative plasticity (high novelty threshold, slow updates)
    pub fn conservative() -> Self {
        Self {
            policy: UpdatePolicy::EMA,
            novelty_threshold: 0.8,   // High threshold = less new memories
            merge_threshold: 0.4,
            decay_rate: 0.005,        // Slow decay
            prune_threshold: 0.05,
            learning_rate: 0.05,      // Slow learning
        }
    }

    /// Aggressive plasticity (low novelty threshold, fast updates)
    pub fn aggressive() -> Self {
        Self {
            policy: UpdatePolicy::STDP,
            novelty_threshold: 0.4,   // Low threshold = more new memories
            merge_threshold: 0.2,
            decay_rate: 0.02,         // Fast decay
            prune_threshold: 0.2,
            learning_rate: 0.2,       // Fast learning
        }
    }

    /// No plasticity (simple replacement)
    pub fn replace_only() -> Self {
        Self {
            policy: UpdatePolicy::Replace,
            novelty_threshold: 1.0,   // Never create new
            merge_threshold: 0.0,     // Never merge
            decay_rate: 0.0,          // No decay
            prune_threshold: 0.0,     // Never prune
            learning_rate: 1.0,       // Full replacement
        }
    }

    /// Bayesian update (confidence-weighted)
    pub fn bayesian() -> Self {
        Self {
            policy: UpdatePolicy::Bayesian,
            novelty_threshold: 0.7,
            merge_threshold: 0.3,
            decay_rate: 0.01,
            prune_threshold: 0.1,
            learning_rate: 0.1,
        }
    }

    /// Apply this rule to decide how to update
    ///
    /// Returns the new strength for the memory based on:
    /// - Current strength
    /// - New observation strength
    /// - Time since last update
    pub fn apply_update(
        &self,
        current_strength: f32,
        new_strength: f32,
        time_delta_seconds: f64,
    ) -> f32 {
        match self.policy {
            UpdatePolicy::STDP => {
                // Decay old, strengthen with new
                let decayed = current_strength * (1.0 - self.decay_rate * time_delta_seconds as f32 / 86400.0);
                let updated = decayed + self.learning_rate * new_strength;
                updated.clamp(0.0, 1.0)
            }

            UpdatePolicy::Replace => new_strength,

            UpdatePolicy::EMA => {
                // Exponential moving average
                let alpha = self.learning_rate;
                alpha * new_strength + (1.0 - alpha) * current_strength
            }

            UpdatePolicy::Bayesian => {
                // Weighted average by confidence
                let total_weight = current_strength + new_strength;
                if total_weight > 0.0 {
                    (current_strength * current_strength + new_strength * new_strength) / total_weight
                } else {
                    0.0
                }
            }

            UpdatePolicy::WTA => {
                // Winner takes all
                current_strength.max(new_strength)
            }
        }
    }

    /// Should this observation create a new memory?
    pub fn should_create_new(&self, novelty_score: f32) -> bool {
        novelty_score > self.novelty_threshold
    }

    /// Should this observation merge with existing?
    pub fn should_merge(&self, similarity: f32) -> bool {
        similarity > (1.0 - self.merge_threshold)
    }

    /// Should this memory be pruned?
    pub fn should_prune(&self, strength: f32) -> bool {
        strength < self.prune_threshold
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
                // STDP-like: strengthen if agree, weaken if disagree
                if current == observed {
                    // Already aligned - reinforce if confident
                    if confidence > 0.7 {
                        current.strengthen()
                    } else {
                        current
                    }
                } else if confidence > 0.5 {
                    // Disagree with confidence - move toward observed
                    self.move_toward(current, observed)
                } else {
                    // Low confidence - decay toward zero
                    self.decay_toward_zero(current)
                }
            }

            UpdatePolicy::Replace => {
                // Simple replacement if confident
                if confidence > 0.5 {
                    observed
                } else {
                    current
                }
            }

            UpdatePolicy::EMA => {
                // EMA doesn't make sense for ternary, use voting instead
                // If observed != current, flip based on confidence
                if current == observed {
                    current
                } else if confidence > self.learning_rate {
                    observed
                } else {
                    current
                }
            }

            UpdatePolicy::Bayesian => {
                // Bayesian-like: weight by confidence
                // Higher confidence = more likely to change
                if confidence > 0.7 {
                    observed
                } else if confidence > 0.3 {
                    // Medium confidence - move toward zero (uncertain)
                    self.decay_toward_zero(current)
                } else {
                    current
                }
            }

            UpdatePolicy::WTA => {
                // Winner-take-all: observed wins if confident
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
            // Already at target
            (Pos, Pos) | (Zero, Zero) | (Neg, Neg) => current,

            // Move toward Pos
            (Zero, Pos) | (Neg, Pos) => current.strengthen(),

            // Move toward Neg
            (Zero, Neg) | (Pos, Neg) => current.weaken(),

            // Move toward Zero
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
                // Pre fired before post (causal) - strengthen
                current.strengthen()
            } else if time_delta_ms < 0 {
                // Post fired before pre (anti-causal) - weaken
                current.weaken()
            } else {
                // Simultaneous - slight strengthen (Hebbian)
                if current == TernaryWeight::Neg {
                    TernaryWeight::Zero
                } else {
                    current
                }
            }
        } else {
            // No co-firing - no change
            current
        }
    }

    /// Ternary majority voting for merge decisions
    ///
    /// Given multiple ternary weights, return the majority value.
    /// Useful for merging multiple observations.
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

        let current = 0.5;
        let new = 0.8;
        let time_delta = 0.0; // Immediate

        let updated = rule.apply_update(current, new, time_delta);

        // Should strengthen
        assert!(updated > current);
        assert!(updated <= 1.0);
    }

    #[test]
    fn test_novelty_threshold() {
        let rule = PlasticityRule::stdp_like();

        assert!(rule.should_create_new(0.7)); // Above threshold
        assert!(!rule.should_create_new(0.5)); // Below threshold
    }

    #[test]
    fn test_merge_decision() {
        let rule = PlasticityRule::stdp_like();

        assert!(rule.should_merge(0.8)); // High similarity
        assert!(!rule.should_merge(0.6)); // Low similarity
    }

    #[test]
    fn test_prune_decision() {
        let rule = PlasticityRule::stdp_like();

        assert!(rule.should_prune(0.05)); // Below threshold
        assert!(!rule.should_prune(0.5)); // Above threshold
    }

    #[test]
    fn test_ema_update() {
        let rule = PlasticityRule {
            policy: UpdatePolicy::EMA,
            learning_rate: 0.1,
            ..PlasticityRule::stdp_like()
        };

        let current = 0.5;
        let new = 1.0;

        let updated = rule.apply_update(current, new, 0.0);

        // Should be weighted average
        let expected = 0.1 * 1.0 + 0.9 * 0.5;
        assert!((updated - expected).abs() < 0.001);
    }
}

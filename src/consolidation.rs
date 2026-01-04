//! Consolidation - Dirty → Clean state transitions
//!
//! Consolidation applies accumulated deltas to create a clean snapshot,
//! pruning weak memories and applying plasticity rules.

use crate::delta::{Delta, DeltaType};
use crate::plasticity::PlasticityRule;
use crate::error::{Error, Result};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Policy for when to consolidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationPolicy {
    /// Triggers that initiate consolidation
    pub triggers: Vec<ConsolidationTrigger>,

    /// Whether to prune weak memories during consolidation
    pub enable_pruning: bool,

    /// Whether to merge similar memories
    pub enable_merging: bool,
}

/// Triggers for consolidation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationTrigger {
    /// Consolidate after N deltas
    DeltaCount(usize),

    /// Consolidate after time period
    TimePeriod { hours: u64 },

    /// Consolidate when dirty state exceeds size
    DirtySize { bytes: usize },

    /// Manual trigger
    Manual,
}

impl ConsolidationPolicy {
    /// Default policy - consolidate every 1000 deltas or 24 hours
    pub fn default() -> Self {
        Self {
            triggers: vec![
                ConsolidationTrigger::DeltaCount(1000),
                ConsolidationTrigger::TimePeriod { hours: 24 },
            ],
            enable_pruning: true,
            enable_merging: true,
        }
    }

    /// Aggressive consolidation - frequent, with pruning
    pub fn aggressive() -> Self {
        Self {
            triggers: vec![
                ConsolidationTrigger::DeltaCount(100),
                ConsolidationTrigger::TimePeriod { hours: 1 },
            ],
            enable_pruning: true,
            enable_merging: true,
        }
    }

    /// Conservative consolidation - infrequent, no pruning
    pub fn conservative() -> Self {
        Self {
            triggers: vec![
                ConsolidationTrigger::DeltaCount(10000),
                ConsolidationTrigger::TimePeriod { hours: 168 }, // 1 week
            ],
            enable_pruning: false,
            enable_merging: false,
        }
    }

    /// Check if any trigger is met
    pub fn should_consolidate(
        &self,
        delta_count: usize,
        last_consolidation: &DateTime<Utc>,
        dirty_size: usize,
    ) -> bool {
        for trigger in &self.triggers {
            match trigger {
                ConsolidationTrigger::DeltaCount(n) => {
                    if delta_count >= *n {
                        return true;
                    }
                }
                ConsolidationTrigger::TimePeriod { hours } => {
                    let elapsed = Utc::now().signed_duration_since(*last_consolidation);
                    let threshold = Duration::hours(*hours as i64);
                    if elapsed >= threshold {
                        return true;
                    }
                }
                ConsolidationTrigger::DirtySize { bytes } => {
                    if dirty_size >= *bytes {
                        return true;
                    }
                }
                ConsolidationTrigger::Manual => {
                    // Manual triggers don't auto-fire
                }
            }
        }
        false
    }
}

/// Consolidated state entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedEntry {
    /// Key
    pub key: String,

    /// Consolidated value
    pub value: Vec<u8>,

    /// Current strength (after plasticity applied)
    pub strength: f32,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// Number of updates contributing to this entry
    pub update_count: usize,
}

/// Result of consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Number of deltas consolidated
    pub deltas_processed: usize,

    /// Number of entries in clean state
    pub entries_after: usize,

    /// Number of entries pruned
    pub entries_pruned: usize,

    /// Number of entries merged
    pub entries_merged: usize,

    /// Size reduction (bytes)
    pub size_reduction: usize,

    /// When consolidation happened
    pub timestamp: DateTime<Utc>,
}

/// Consolidate deltas into clean state
pub fn consolidate(
    dirty_deltas: &[Delta],
    clean_state: &HashMap<String, ConsolidatedEntry>,
    plasticity_rule: &PlasticityRule,
    policy: &ConsolidationPolicy,
) -> Result<(HashMap<String, ConsolidatedEntry>, ConsolidationResult)> {
    let mut new_state = clean_state.clone();
    let mut stats = ConsolidationResult {
        deltas_processed: dirty_deltas.len(),
        entries_after: 0,
        entries_pruned: 0,
        entries_merged: 0,
        size_reduction: 0,
        timestamp: Utc::now(),
    };

    // Apply each delta
    for delta in dirty_deltas {
        match delta.delta_type {
            DeltaType::Create | DeltaType::Update => {
                if let Some(existing) = new_state.get_mut(&delta.key) {
                    // Update existing entry
                    let time_delta = delta
                        .metadata
                        .timestamp
                        .signed_duration_since(existing.updated_at)
                        .num_seconds() as f64;

                    let new_strength = plasticity_rule.apply_update(
                        existing.strength,
                        delta.metadata.strength,
                        time_delta,
                    );

                    existing.value = delta.value.clone();
                    existing.strength = new_strength;
                    existing.updated_at = delta.metadata.timestamp;
                    existing.update_count += 1;
                } else {
                    // Create new entry
                    new_state.insert(
                        delta.key.clone(),
                        ConsolidatedEntry {
                            key: delta.key.clone(),
                            value: delta.value.clone(),
                            strength: delta.metadata.strength,
                            updated_at: delta.metadata.timestamp,
                            update_count: 1,
                        },
                    );
                }
            }

            DeltaType::Delete => {
                new_state.remove(&delta.key);
            }

            DeltaType::Merge => {
                // Merge operation - combine with existing
                if let Some(existing) = new_state.get_mut(&delta.key) {
                    // For now, just update like normal
                    // In future, could implement proper merging logic
                    let time_delta = delta
                        .metadata
                        .timestamp
                        .signed_duration_since(existing.updated_at)
                        .num_seconds() as f64;

                    let new_strength = plasticity_rule.apply_update(
                        existing.strength,
                        delta.metadata.strength,
                        time_delta,
                    );

                    existing.strength = new_strength;
                    existing.updated_at = delta.metadata.timestamp;
                    existing.update_count += 1;
                    stats.entries_merged += 1;
                }
            }
        }
    }

    // Prune weak entries if enabled
    if policy.enable_pruning {
        let before_count = new_state.len();
        new_state.retain(|_, entry| {
            !plasticity_rule.should_prune(entry.strength)
        });
        stats.entries_pruned = before_count - new_state.len();
    }

    stats.entries_after = new_state.len();

    Ok((new_state, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_policy_delta_count() {
        let policy = ConsolidationPolicy::default();
        let last_consolidation = Utc::now();

        assert!(!policy.should_consolidate(500, &last_consolidation, 0));
        assert!(policy.should_consolidate(1000, &last_consolidation, 0));
    }

    #[test]
    fn test_consolidation_policy_time() {
        let policy = ConsolidationPolicy::default();
        let last_consolidation = Utc::now() - Duration::hours(25);

        assert!(policy.should_consolidate(0, &last_consolidation, 0));
    }

    #[test]
    fn test_consolidate_create() {
        let deltas = vec![
            Delta::create("key1", b"value1".to_vec(), "source"),
        ];

        let clean_state = HashMap::new();
        let plasticity = PlasticityRule::stdp_like();
        let policy = ConsolidationPolicy::default();

        let (new_state, stats) = consolidate(&deltas, &clean_state, &plasticity, &policy).unwrap();

        assert_eq!(new_state.len(), 1);
        assert_eq!(stats.deltas_processed, 1);
        assert_eq!(stats.entries_after, 1);
    }

    #[test]
    fn test_consolidate_update() {
        let mut clean_state = HashMap::new();
        clean_state.insert(
            "key1".to_string(),
            ConsolidatedEntry {
                key: "key1".to_string(),
                value: b"old_value".to_vec(),
                strength: 0.5,
                updated_at: Utc::now(),
                update_count: 1,
            },
        );

        let deltas = vec![
            Delta::update("key1", b"new_value".to_vec(), "source", 0.8, None),
        ];

        let plasticity = PlasticityRule::stdp_like();
        let policy = ConsolidationPolicy::default();

        let (new_state, _) = consolidate(&deltas, &clean_state, &plasticity, &policy).unwrap();

        let entry = new_state.get("key1").unwrap();
        assert_eq!(entry.value, b"new_value");
        assert!(entry.strength > 0.5); // Should have strengthened
    }

    #[test]
    fn test_consolidate_with_pruning() {
        let mut clean_state = HashMap::new();
        clean_state.insert(
            "weak_key".to_string(),
            ConsolidatedEntry {
                key: "weak_key".to_string(),
                value: b"value".to_vec(),
                strength: 0.05, // Below prune threshold
                updated_at: Utc::now(),
                update_count: 1,
            },
        );

        let deltas = vec![];
        let plasticity = PlasticityRule::stdp_like();
        let policy = ConsolidationPolicy::default();

        let (new_state, stats) = consolidate(&deltas, &clean_state, &plasticity, &policy).unwrap();

        assert_eq!(new_state.len(), 0);
        assert_eq!(stats.entries_pruned, 1);
    }
}

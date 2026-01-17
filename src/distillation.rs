//! # Distillation - Semantic Delta Sharing
//!
//! When learning systems evolve, they can share distilled knowledge with peers.
//! This module provides types for semantic deltas - the IDEAS, not raw weights.
//!
//! ## Key Principle
//!
//! Share the IDEA, not the full knowledge:
//! - System A learns TypeScript → doesn't send "all of TypeScript"
//! - Sends: "How does TS typing differ from Rust typing"
//! - System B receives semantic delta, consolidates personally
//! - Both have loose shared understanding, not identical copies
//!
//! ## Distillation Process
//!
//! 1. Compute embedding difference (before/after learning)
//! 2. Check if delta is significant enough to share
//! 3. Extract semantic anchor (what was learned ABOUT)
//! 4. Create shareable delta (NOT raw weights)
//!
//! ## Usage
//!
//! The types in this module are transport-agnostic. Your application
//! decides how to transmit deltas (message queues, HTTP, files, etc.).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Semantic delta for cross-instance learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDelta {
    /// Unique delta identifier
    pub id: String,

    /// What concept this delta relates to (anchor point)
    pub concept_anchor: String,

    /// The embedding shift (how understanding changed)
    /// Typically 384-d for mini embedding models
    pub embedding_delta: Vec<f32>,

    /// Strength of the delta (how confident)
    pub strength: f32,

    /// Source of this delta
    pub source: DeltaSource,

    /// When this delta was created
    pub timestamp: DateTime<Utc>,

    /// Optional: Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

impl SemanticDelta {
    /// Create a new semantic delta
    pub fn new(
        concept_anchor: impl Into<String>,
        embedding_delta: Vec<f32>,
        strength: f32,
        source: DeltaSource,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            concept_anchor: concept_anchor.into(),
            embedding_delta,
            strength,
            source,
            timestamp: Utc::now(),
            tags: Vec::new(),
        }
    }

    /// Add tags to the delta
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Compute magnitude of the embedding delta
    pub fn magnitude(&self) -> f32 {
        self.embedding_delta
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }

    /// Check if delta is significant enough to share
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.magnitude() >= threshold && self.strength >= 0.3
    }

    /// Scale the delta by a factor
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            id: self.id.clone(),
            concept_anchor: self.concept_anchor.clone(),
            embedding_delta: self.embedding_delta.iter().map(|x| x * factor).collect(),
            strength: self.strength * factor,
            source: self.source.clone(),
            timestamp: self.timestamp,
            tags: self.tags.clone(),
        }
    }
}

/// Source of a semantic delta
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeltaSource {
    /// Self-learning from experience
    SelfLearning {
        /// Task that triggered learning
        task_context: String,
    },

    /// Peer learning (received from another instance)
    PeerLearning {
        /// Instance that shared this
        instance_id: String,
        /// When it was shared
        shared_at: DateTime<Utc>,
    },

    /// Migration capsule
    Migration {
        /// Version of the migration
        version: String,
    },

    /// Skill box baseline
    SkillBox {
        /// Skill identifier
        skill_id: String,
    },

    /// Document ingestion (seeding)
    DocumentIngestion {
        /// Document/Engram identifier
        document_id: String,
    },
}

/// Configuration for distillation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Minimum magnitude for delta to be shareable
    pub min_share_magnitude: f32,

    /// Minimum strength for delta to be shareable
    pub min_share_strength: f32,

    /// Maximum age of deltas to accept (in seconds)
    pub max_delta_age_secs: u64,

    /// Scale factor when ingesting peer deltas
    pub peer_ingestion_scale: f32,

    /// Whether to auto-share learned deltas
    pub auto_share: bool,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_share_magnitude: 0.05,
            min_share_strength: 0.3,
            max_delta_age_secs: 86400 * 7, // 1 week
            peer_ingestion_scale: 0.5,     // Peers' learning weighted at 50%
            auto_share: false,
        }
    }
}

/// A snapshot of thermogram state for distillation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermogramSnapshot {
    /// Snapshot identifier
    pub id: String,

    /// Centroid embedding (average of all entries)
    pub centroid: Vec<f32>,

    /// Number of entries at snapshot time
    pub entry_count: usize,

    /// When snapshot was taken
    pub timestamp: DateTime<Utc>,

    /// Optional: Entry-level embeddings for fine-grained comparison
    #[serde(default)]
    pub entry_embeddings: Vec<(String, Vec<f32>)>,
}

impl ThermogramSnapshot {
    /// Create a snapshot from centroid
    pub fn from_centroid(id: impl Into<String>, centroid: Vec<f32>, entry_count: usize) -> Self {
        Self {
            id: id.into(),
            centroid,
            entry_count,
            timestamp: Utc::now(),
            entry_embeddings: Vec::new(),
        }
    }

    /// Add entry-level embeddings
    pub fn with_entries(mut self, entries: Vec<(String, Vec<f32>)>) -> Self {
        self.entry_embeddings = entries;
        self
    }
}

/// Distill learning into shareable semantic delta
///
/// Computes the embedding difference between before/after states
/// and creates a shareable delta if significant.
pub fn distill_learning(
    before: &ThermogramSnapshot,
    after: &ThermogramSnapshot,
    concept_anchor: impl Into<String>,
    task_context: impl Into<String>,
    config: &DistillationConfig,
) -> Option<SemanticDelta> {
    // Ensure same dimensionality
    if before.centroid.len() != after.centroid.len() {
        return None;
    }

    // Compute embedding difference
    let embedding_delta: Vec<f32> = before
        .centroid
        .iter()
        .zip(&after.centroid)
        .map(|(b, a)| a - b)
        .collect();

    // Compute magnitude
    let magnitude: f32 = embedding_delta.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Check if significant
    if magnitude < config.min_share_magnitude {
        return None;
    }

    // Compute confidence based on entry count change and magnitude
    let entry_ratio = (after.entry_count as f32) / (before.entry_count.max(1) as f32);
    let strength = (magnitude * entry_ratio).min(1.0);

    if strength < config.min_share_strength {
        return None;
    }

    Some(SemanticDelta::new(
        concept_anchor,
        embedding_delta,
        strength,
        DeltaSource::SelfLearning {
            task_context: task_context.into(),
        },
    ))
}

/// Apply semantic delta to embedding (for ingestion)
pub fn apply_delta_to_embedding(embedding: &[f32], delta: &SemanticDelta, scale: f32) -> Vec<f32> {
    if embedding.len() != delta.embedding_delta.len() {
        return embedding.to_vec();
    }

    embedding
        .iter()
        .zip(&delta.embedding_delta)
        .map(|(e, d)| e + d * scale * delta.strength)
        .collect()
}

/// Compute cosine similarity between embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Batch of deltas for transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaBatch {
    /// Batch identifier
    pub id: String,

    /// Source instance
    pub source_instance: String,

    /// Deltas in this batch
    pub deltas: Vec<SemanticDelta>,

    /// When batch was created
    pub created: DateTime<Utc>,

    /// Optional signature for verification
    #[serde(default)]
    pub signature: Option<String>,
}

impl DeltaBatch {
    /// Create a new delta batch
    pub fn new(source_instance: impl Into<String>, deltas: Vec<SemanticDelta>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source_instance: source_instance.into(),
            deltas,
            created: Utc::now(),
            signature: None,
        }
    }

    /// Add signature to batch
    pub fn with_signature(mut self, sig: impl Into<String>) -> Self {
        self.signature = Some(sig.into());
        self
    }

    /// Filter to only significant deltas
    pub fn filter_significant(&self, threshold: f32) -> Self {
        Self {
            id: self.id.clone(),
            source_instance: self.source_instance.clone(),
            deltas: self
                .deltas
                .iter()
                .filter(|d| d.is_significant(threshold))
                .cloned()
                .collect(),
            created: self.created,
            signature: None, // Signature invalidated by filtering
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_delta_creation() {
        let delta = SemanticDelta::new(
            "rust_types",
            vec![0.1, -0.2, 0.3],
            0.8,
            DeltaSource::SelfLearning {
                task_context: "learning types".to_string(),
            },
        );

        assert_eq!(delta.concept_anchor, "rust_types");
        assert_eq!(delta.strength, 0.8);
    }

    #[test]
    fn test_delta_magnitude() {
        let delta = SemanticDelta::new(
            "test",
            vec![3.0, 4.0], // 3-4-5 triangle
            0.5,
            DeltaSource::SelfLearning {
                task_context: "test".to_string(),
            },
        );

        assert!((delta.magnitude() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_distill_learning() {
        let before = ThermogramSnapshot::from_centroid("snap1", vec![0.0, 0.0, 0.0], 10);

        // Larger delta to produce sufficient strength (magnitude * entry_ratio >= 0.3)
        let after = ThermogramSnapshot::from_centroid("snap2", vec![0.2, 0.2, 0.2], 20);

        let config = DistillationConfig::default();

        let delta = distill_learning(&before, &after, "test_concept", "learning test", &config);

        assert!(delta.is_some());
        let delta = delta.unwrap();
        assert_eq!(delta.concept_anchor, "test_concept");
    }

    #[test]
    fn test_distill_insignificant() {
        let before = ThermogramSnapshot::from_centroid("snap1", vec![0.0, 0.0, 0.0], 10);

        // Very small change
        let after = ThermogramSnapshot::from_centroid("snap2", vec![0.001, 0.001, 0.001], 10);

        let config = DistillationConfig::default();

        let delta = distill_learning(&before, &after, "test_concept", "learning test", &config);

        assert!(delta.is_none()); // Too small to share
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001); // Orthogonal

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001); // Opposite
    }

    #[test]
    fn test_apply_delta() {
        let embedding = vec![1.0, 2.0, 3.0];
        let delta = SemanticDelta::new(
            "test",
            vec![0.1, 0.2, 0.3],
            1.0,
            DeltaSource::SelfLearning {
                task_context: "test".to_string(),
            },
        );

        let result = apply_delta_to_embedding(&embedding, &delta, 1.0);

        assert!((result[0] - 1.1).abs() < 0.001);
        assert!((result[1] - 2.2).abs() < 0.001);
        assert!((result[2] - 3.3).abs() < 0.001);
    }

    #[test]
    fn test_delta_batch() {
        let deltas = vec![SemanticDelta::new(
            "test",
            vec![0.1, 0.2],
            0.8,
            DeltaSource::SelfLearning {
                task_context: "test".to_string(),
            },
        )];

        let batch = DeltaBatch::new("instance-1", deltas);

        assert_eq!(batch.deltas.len(), 1);
        assert_eq!(batch.source_instance, "instance-1");
    }
}

//! # Thermogram
//!
//! A plastic memory capsule with hot/cold tensor states, rule-governed deltas,
//! and hash-chained auditability.
//!
//! ## Core Concept
//!
//! Traditional storage is either:
//! - **Mutable** (databases, files) - fast but no audit trail
//! - **Immutable** (Engram, Git) - auditable but can't evolve
//!
//! Thermogram combines both in a single file with dual thermal states:
//! - **Hot tensors** - High plasticity, volatile, session-local
//! - **Cold tensors** - Crystallized, stable, personality backbone
//! - **Delta chain** (append-only) - fast writes with audit trail
//! - **Plasticity rules** (STDP-like) - when to update vs create new
//! - **Hash chain** - cryptographic audit trail
//! - **Consolidation** - crystallizes hot → cold, prunes weak entries
//! - **Warming** - reactivates cold → hot when needed
//! - **Engram export** - archive without deletion
//!
//! ## Thermal States
//!
//! Single file, two internal states, bidirectional transitions:
//! - **Hot**: Fast updates, session-local, high plasticity
//! - **Cold**: Slow changes, crystallized personality, stable
//! - Consolidation moves strong hot entries to cold
//! - Warming moves cold entries back to hot when reactivated
//!
//! ## Use Cases
//!
//! 1. **LLM Activation Mining** - Cluster centroids evolve as new patterns discovered
//! 2. **Agent Memory** - Episodic memory with replay and consolidation
//! 3. **Knowledge Graphs** - Concepts strengthen/weaken over time
//! 4. **Neural Weights** - Save checkpoints with full training history
//! 5. **Personality Substrate** - Synaptic weights that define identity
//!
//! ## Example
//!
//! ```rust,no_run
//! use thermogram::{Thermogram, PlasticityRule, Delta};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Create new thermogram for LLM activation clusters
//! let mut thermo = Thermogram::new("llm_clusters", PlasticityRule::stdp_like());
//!
//! // Apply delta (cluster centroid update)
//! let new_centroid = vec![0.5_f32; 2048];
//! let delta = Delta::update(
//!     "cluster_0",
//!     bincode::serialize(&new_centroid)?,
//!     "llm_mining",
//!     0.8,
//!     thermo.dirty_chain.head_hash.clone(),
//! );
//! thermo.apply_delta(delta)?;
//!
//! // Read current state (hot tensors take priority)
//! let centroid = thermo.read("cluster_0")?;
//!
//! // Consolidate (crystallize hot → cold, prune weak)
//! thermo.consolidate()?;
//!
//! // Export to JSON
//! thermo.export_to_json("llm_knowledge_v1.json")?;
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod delta;
pub mod plasticity;
pub mod consolidation;
pub mod hash_chain;
pub mod export;
pub mod error;
pub mod plasticity_engine;
pub mod embedded_snn;
pub mod ternary;

// Re-exports
pub use crate::core::{Thermogram, ThermalState, ThermalConfig, CrystallizationResult, ThermogramStats};
pub use crate::delta::{Delta, DeltaType};
pub use crate::plasticity::{PlasticityRule, UpdatePolicy};
pub use crate::consolidation::{ConsolidationPolicy, ConsolidationTrigger};
pub use crate::hash_chain::HashChain;
pub use crate::error::{Error, Result};
pub use crate::plasticity_engine::{NeuromodState, NeuromodSyncConfig, PlasticityEngine};
pub use crate::embedded_snn::{EmbeddedSNN, EmbeddedSNNConfig};
pub use crate::ternary::{TernaryWeight, PackedTernary, TernaryLayer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_thermogram() {
        // Placeholder - real tests in integration tests
    }
}

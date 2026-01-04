//! # Thermogram
//!
//! A plastic memory capsule with dirty/clean states, rule-governed deltas,
//! and hash-chained auditability.
//!
//! ## Core Concept
//!
//! Traditional storage is either:
//! - **Mutable** (databases, files) - fast but no audit trail
//! - **Immutable** (Engram, Git) - auditable but can't evolve
//!
//! Thermogram combines both:
//! - **Dirty state** (append-only deltas) - fast, mutable, auditable
//! - **Clean state** (consolidated snapshot) - efficient reads
//! - **Plasticity rules** (STDP-like) - when to update vs create new
//! - **Hash chain** - cryptographic audit trail
//! - **Consolidation cycles** - dirty → clean on schedule
//! - **Engram export** - archive without deletion
//!
//! ## Use Cases
//!
//! 1. **LLM Activation Mining** - Cluster centroids evolve as new patterns discovered
//! 2. **Agent Memory** - Episodic memory with replay and consolidation
//! 3. **Knowledge Graphs** - Concepts strengthen/weaken over time
//! 4. **Neural Weights** - Save checkpoints with full training history
//!
//! ## Example
//!
//! ```rust,no_run
//! use thermogram::{Thermogram, PlasticityRule, Delta};
//!
//! // Create new thermogram for LLM activation clusters
//! let mut thermo = Thermogram::new("llm_clusters", PlasticityRule::stdp_like())?;
//!
//! // Apply delta (cluster centroid update)
//! thermo.apply_delta(Delta::update("cluster_0", new_centroid))?;
//!
//! // Read current state (dirty + clean merged)
//! let centroid = thermo.read("cluster_0")?;
//!
//! // Consolidate (dirty → clean)
//! thermo.consolidate()?;
//!
//! // Export to Engram (immutable archive)
//! thermo.export_to_engram("llm_knowledge_v1.eng")?;
//! # Ok::<(), anyhow::Error>(())
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

// Re-exports
pub use crate::core::Thermogram;
pub use crate::delta::{Delta, DeltaType};
pub use crate::plasticity::{PlasticityRule, UpdatePolicy};
pub use crate::consolidation::{ConsolidationPolicy, ConsolidationTrigger};
pub use crate::hash_chain::HashChain;
pub use crate::error::{Error, Result};
pub use crate::plasticity_engine::{NeuromodState, NeuromodSyncConfig, PlasticityEngine};
pub use crate::embedded_snn::{EmbeddedSNN, EmbeddedSNNConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_thermogram() {
        // Placeholder - real tests in integration tests
    }
}

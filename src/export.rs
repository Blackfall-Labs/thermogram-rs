//! Export - Archive Thermogram to Engram
//!
//! Exports consolidated state to immutable Engram archives without deletion.
//! Maintains full audit trail by including hash chain.
//!
//! ## Signal-Native Export
//!
//! All strength values are Signal (polarity + magnitude). Export uses JSON
//! representation for human-readable inspection.

use crate::core::Thermogram;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use ternary_signal::Signal;

/// Export format for Engram archival
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramExport {
    /// Thermogram ID
    pub thermogram_id: String,

    /// Thermogram name
    pub name: String,

    /// Consolidated state (key-value pairs)
    pub state: Vec<EngramEntry>,

    /// Full delta history (audit trail)
    pub history: Vec<EngramDelta>,

    /// Export metadata
    pub metadata: EngramExportMetadata,
}

/// Entry in exported state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramEntry {
    pub key: String,
    pub value: Vec<Signal>,
    /// Strength as Signal (polarity + magnitude)
    pub strength: Signal,
    pub update_count: usize,
}

/// Delta in exported history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramDelta {
    pub timestamp: String,
    pub delta_type: String,
    pub key: String,
    /// Strength as Signal
    pub strength: Signal,
    pub hash: String,
}

/// Metadata for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramExportMetadata {
    pub exported_at: String,
    pub thermogram_created_at: String,
    pub last_consolidation: String,
    pub total_deltas: usize,
    pub total_consolidations: usize,
    pub total_entries: usize,
}

impl Thermogram {
    /// Export to Engram-compatible format
    ///
    /// This consolidates the Thermogram first, then exports to a format
    /// suitable for Engram archival.
    pub fn export_to_engram_data(&mut self) -> Result<EngramExport> {
        // Consolidate first to ensure state is up to date
        self.consolidate()?;

        // Convert both hot and cold entries (cold entries are the crystallized state)
        let mut state: Vec<EngramEntry> = self
            .cold_entries
            .values()
            .map(|entry| EngramEntry {
                key: entry.key.clone(),
                value: entry.value.clone(),
                strength: entry.strength,
                update_count: entry.update_count,
            })
            .collect();

        // Also include hot entries (session-local, not yet crystallized)
        state.extend(self.hot_entries.values().map(|entry| EngramEntry {
            key: entry.key.clone(),
            value: entry.value.clone(),
            strength: entry.strength,
            update_count: entry.update_count,
        }));

        // Convert delta history
        let history: Vec<EngramDelta> = self
            .dirty_chain
            .deltas
            .iter()
            .map(|delta| EngramDelta {
                timestamp: delta.metadata.timestamp.to_rfc3339(),
                delta_type: format!("{:?}", delta.delta_type),
                key: delta.key.clone(),
                strength: delta.metadata.strength,
                hash: delta.hash.clone(),
            })
            .collect();

        let total_entries = state.len();

        let export = EngramExport {
            thermogram_id: self.id.clone(),
            name: self.name.clone(),
            state,
            history,
            metadata: EngramExportMetadata {
                exported_at: chrono::Utc::now().to_rfc3339(),
                thermogram_created_at: self.metadata.created_at.to_rfc3339(),
                last_consolidation: self.metadata.last_consolidation.to_rfc3339(),
                total_deltas: self.metadata.total_deltas,
                total_consolidations: self.metadata.total_consolidations,
                total_entries,
            },
        };

        Ok(export)
    }

    /// Export to JSON file (basic export)
    pub fn export_to_json(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let export = self.export_to_engram_data()?;

        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;

        Ok(())
    }

    #[cfg(feature = "engram-export")]
    /// Export to Engram archive
    ///
    /// Requires the `engram-export` feature to be enabled.
    pub fn export_to_engram(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use engram::{Engram, Manifest, ManifestEntry};

        let export = self.export_to_engram_data()?;

        // Create Engram manifest
        let mut manifest = Manifest::new(&self.name, "thermogram_export");

        // Add state entries
        manifest.add_entry(ManifestEntry {
            path: "state.json".to_string(),
            hash: String::new(), // Will be computed
            size: 0,             // Will be computed
            metadata: serde_json::json!({
                "type": "thermogram_state",
                "entry_count": export.state.len(),
            }),
        });

        // Add history
        manifest.add_entry(ManifestEntry {
            path: "history.json".to_string(),
            hash: String::new(),
            size: 0,
            metadata: serde_json::json!({
                "type": "thermogram_history",
                "delta_count": export.history.len(),
            }),
        });

        // Create Engram and pack
        let mut engram = Engram::create(path, manifest)?;

        // Write state
        let state_json = serde_json::to_vec_pretty(&export.state)?;
        engram.add_file("state.json", &state_json)?;

        // Write history
        let history_json = serde_json::to_vec_pretty(&export.history)?;
        engram.add_file("history.json", &history_json)?;

        // Finalize
        engram.finalize()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::Delta;
    use crate::plasticity::PlasticityRule;

    #[test]
    fn test_export_to_json() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("export.json");

        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta = Delta::create("key1", vec![Signal::positive(100)], "source");
        thermo.apply_delta(delta).unwrap();

        thermo.export_to_json(&path).unwrap();

        // Verify file exists and contains data
        let json = std::fs::read_to_string(&path).unwrap();
        assert!(json.contains("thermogram_id"));
        assert!(json.contains("key1"));
    }

    #[test]
    fn test_export_data_structure() {
        let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

        let delta = Delta::create("key1", vec![Signal::positive(100)], "source");
        thermo.apply_delta(delta).unwrap();

        let export = thermo.export_to_engram_data().unwrap();

        assert_eq!(export.name, "test");
        assert_eq!(export.state.len(), 1);
        assert_eq!(export.state[0].key, "key1");
    }
}

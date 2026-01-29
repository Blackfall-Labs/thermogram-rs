//! Binary Codec — `.thermo` v1 format
//!
//! Custom hand-rolled binary serialization for Thermograms. All data is Signal-native.
//!
//! ## Wire Format
//!
//! ```text
//! [Header 16B] [Identity] [ThermalConfig] [Metadata] [PlasticityRule]
//! [ConsolidationPolicy] [Entries x4] [HashChain]
//! ```
//!
//! All multi-byte integers are little-endian. Strings are length-prefixed UTF-8.
//! Signals are 2 bytes: `[polarity as u8, magnitude]`.

use std::io::{self, Read, Cursor};
use ternary_signal::Signal;
use chrono::{DateTime, Utc, TimeZone};

use crate::consolidation::{
    ConsolidatedEntry, ConsolidationPolicy, ConsolidationTrigger,
};
use crate::core::{ThermalConfig, ThermalState, Thermogram, ThermogramMetadata};
use crate::delta::{Delta, DeltaMetadata, DeltaType};
use crate::hash_chain::HashChain;
use crate::plasticity::{PlasticityRule, UpdatePolicy};
use crate::error::{Error, Result};

// ── Constants ──────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"THRM";
const VERSION: u16 = 1;

// ── Public API ─────────────────────────────────────────────────────

/// Encode a Thermogram to binary `.thermo` v1 format.
pub fn encode(thermo: &Thermogram) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(8192);

    // Write placeholder header (we'll patch total_size and checksum after)
    let header_pos = buf.len();
    write_bytes(&mut buf, MAGIC)?;
    write_u16(&mut buf, VERSION)?;
    write_u16(&mut buf, 0)?; // flags (reserved)
    write_u32(&mut buf, 0)?; // total_size placeholder
    write_u32(&mut buf, 0)?; // checksum placeholder

    // Identity
    write_str(&mut buf, &thermo.name)?;
    write_str(&mut buf, &thermo.id)?;

    // Thermal config
    write_thermal_config(&mut buf, &thermo.thermal_config)?;

    // Metadata
    write_metadata(&mut buf, &thermo.metadata)?;

    // Plasticity rule
    write_plasticity_rule(&mut buf, &thermo.plasticity_rule)?;

    // Consolidation policy
    write_consolidation_policy(&mut buf, &thermo.consolidation_policy)?;

    // Entry layers (hot, warm, cool, cold)
    for state in ThermalState::all() {
        write_entry_layer(&mut buf, thermo.entries(state))?;
    }

    // Hash chain
    write_hash_chain(&mut buf, &thermo.dirty_chain)?;

    // Patch header: total size
    let total_size = buf.len() as u32;
    buf[header_pos + 8..header_pos + 12].copy_from_slice(&total_size.to_le_bytes());

    // Patch header: CRC32 of everything after header
    let checksum = crc32fast::hash(&buf[16..]);
    buf[header_pos + 12..header_pos + 16].copy_from_slice(&checksum.to_le_bytes());

    Ok(buf)
}

/// Decode a Thermogram from binary `.thermo` v1 format.
pub fn decode(data: &[u8]) -> Result<Thermogram> {
    let mut r = Cursor::new(data);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| Error::Deserialization(format!("header magic: {}", e)))?;
    if &magic != MAGIC {
        return Err(Error::Deserialization("invalid magic: not a .thermo file".into()));
    }

    let version = read_u16(&mut r)?;
    if version != VERSION {
        return Err(Error::Deserialization(format!("unsupported version: {}", version)));
    }

    let _flags = read_u16(&mut r)?;
    let total_size = read_u32(&mut r)?;
    let stored_checksum = read_u32(&mut r)?;

    // Verify checksum
    if data.len() < 16 {
        return Err(Error::Deserialization("file too short".into()));
    }
    let computed_checksum = crc32fast::hash(&data[16..]);
    if stored_checksum != computed_checksum {
        return Err(Error::Deserialization(format!(
            "checksum mismatch: stored={}, computed={}",
            stored_checksum, computed_checksum
        )));
    }

    if total_size as usize != data.len() {
        return Err(Error::Deserialization(format!(
            "size mismatch: header says {} but file is {} bytes",
            total_size,
            data.len()
        )));
    }

    // Identity
    let name = read_str(&mut r)?;
    let id = read_str(&mut r)?;

    // Thermal config
    let thermal_config = read_thermal_config(&mut r)?;

    // Metadata
    let metadata = read_metadata(&mut r)?;

    // Plasticity rule
    let plasticity_rule = read_plasticity_rule(&mut r)?;

    // Consolidation policy
    let consolidation_policy = read_consolidation_policy(&mut r)?;

    // Entry layers
    let hot_entries = read_entry_layer(&mut r)?;
    let warm_entries = read_entry_layer(&mut r)?;
    let cool_entries = read_entry_layer(&mut r)?;
    let cold_entries = read_entry_layer(&mut r)?;

    // Hash chain
    let dirty_chain = read_hash_chain(&mut r)?;

    Ok(Thermogram {
        id,
        name,
        hot_entries,
        warm_entries,
        cool_entries,
        cold_entries,
        dirty_chain,
        plasticity_rule,
        consolidation_policy,
        thermal_config,
        metadata,
    })
}

// ── Primitives ─────────────────────────────────────────────────────

fn write_u8(w: &mut Vec<u8>, v: u8) -> io::Result<()> {
    w.push(v);
    Ok(())
}

fn write_u16(w: &mut Vec<u8>, v: u16) -> io::Result<()> {
    w.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn write_u32(w: &mut Vec<u8>, v: u32) -> io::Result<()> {
    w.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn write_u64(w: &mut Vec<u8>, v: u64) -> io::Result<()> {
    w.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn write_i64(w: &mut Vec<u8>, v: i64) -> io::Result<()> {
    w.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn write_bytes(w: &mut Vec<u8>, data: &[u8]) -> io::Result<()> {
    w.extend_from_slice(data);
    Ok(())
}

fn write_signal(w: &mut Vec<u8>, s: Signal) -> io::Result<()> {
    w.push(s.polarity as u8);
    w.push(s.magnitude);
    Ok(())
}

fn write_signals(w: &mut Vec<u8>, signals: &[Signal]) -> io::Result<()> {
    write_u32(w, signals.len() as u32)?;
    for s in signals {
        write_signal(w, *s)?;
    }
    Ok(())
}

fn write_str(w: &mut Vec<u8>, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u16(w, bytes.len() as u16)?;
    w.extend_from_slice(bytes);
    Ok(())
}

fn write_opt_str(w: &mut Vec<u8>, s: &Option<String>) -> io::Result<()> {
    match s {
        Some(s) => {
            write_u8(w, 1)?;
            write_str(w, s)?;
        }
        None => {
            write_u8(w, 0)?;
        }
    }
    Ok(())
}

fn write_bool(w: &mut Vec<u8>, v: bool) -> io::Result<()> {
    w.push(if v { 1 } else { 0 });
    Ok(())
}

// ── Read primitives ────────────────────────────────────────────────

fn read_u8(r: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read u8: {}", e)))?;
    Ok(buf[0])
}

fn read_u16(r: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read u16: {}", e)))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read u32: {}", e)))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read u64: {}", e)))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut Cursor<&[u8]>) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read i64: {}", e)))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_signal(r: &mut Cursor<&[u8]>) -> Result<Signal> {
    let polarity = read_u8(r)? as i8;
    let magnitude = read_u8(r)?;
    Ok(Signal::new(polarity, magnitude))
}

fn read_signals(r: &mut Cursor<&[u8]>) -> Result<Vec<Signal>> {
    let count = read_u32(r)? as usize;
    let mut signals = Vec::with_capacity(count);
    for _ in 0..count {
        signals.push(read_signal(r)?);
    }
    Ok(signals)
}

fn read_str(r: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u16(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| Error::Deserialization(format!("read str: {}", e)))?;
    String::from_utf8(buf).map_err(|e| Error::Deserialization(format!("invalid UTF-8: {}", e)))
}

fn read_opt_str(r: &mut Cursor<&[u8]>) -> Result<Option<String>> {
    let has = read_u8(r)?;
    if has == 1 {
        Ok(Some(read_str(r)?))
    } else {
        Ok(None)
    }
}

fn read_bool(r: &mut Cursor<&[u8]>) -> Result<bool> {
    Ok(read_u8(r)? != 0)
}

// ── Timestamp helpers ──────────────────────────────────────────────

fn write_timestamp(w: &mut Vec<u8>, ts: &DateTime<Utc>) -> io::Result<()> {
    // Store seconds + nanoseconds separately to preserve full precision
    // (needed because delta hashes depend on to_rfc3339() output)
    write_i64(w, ts.timestamp())?;
    write_u32(w, ts.timestamp_subsec_nanos())
}

fn read_timestamp(r: &mut Cursor<&[u8]>) -> Result<DateTime<Utc>> {
    let secs = read_i64(r)?;
    let nanos = read_u32(r)?;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .ok_or_else(|| Error::Deserialization(format!("invalid timestamp: {}s {}ns", secs, nanos)))
}

// ── Section encoders ───────────────────────────────────────────────

fn write_thermal_config(w: &mut Vec<u8>, cfg: &ThermalConfig) -> io::Result<()> {
    // decay_rates [Signal; 4]
    for s in &cfg.decay_rates {
        write_signal(w, *s)?;
    }
    // promotion_thresholds [Signal; 4]
    for s in &cfg.promotion_thresholds {
        write_signal(w, *s)?;
    }
    // demotion_thresholds [Signal; 4]
    for s in &cfg.demotion_thresholds {
        write_signal(w, *s)?;
    }
    // min_observations [u64; 4]
    for &obs in &cfg.min_observations {
        write_u64(w, obs as u64)?;
    }
    // allow_demotion [bool; 4]
    for &b in &cfg.allow_demotion {
        write_bool(w, b)?;
    }
    // prune_threshold: Signal
    write_signal(w, cfg.prune_threshold)?;
    // crystallization_threshold: Signal
    write_signal(w, cfg.crystallization_threshold)?;
    // allow_warming: bool
    write_bool(w, cfg.allow_warming)?;
    // warming_delta: Signal
    write_signal(w, cfg.warming_delta)?;
    Ok(())
}

fn read_thermal_config(r: &mut Cursor<&[u8]>) -> Result<ThermalConfig> {
    let mut decay_rates = [Signal::ZERO; 4];
    for s in &mut decay_rates {
        *s = read_signal(r)?;
    }

    let mut promotion_thresholds = [Signal::ZERO; 4];
    for s in &mut promotion_thresholds {
        *s = read_signal(r)?;
    }

    let mut demotion_thresholds = [Signal::ZERO; 4];
    for s in &mut demotion_thresholds {
        *s = read_signal(r)?;
    }

    let mut min_observations = [0usize; 4];
    for obs in &mut min_observations {
        *obs = read_u64(r)? as usize;
    }

    let mut allow_demotion = [false; 4];
    for b in &mut allow_demotion {
        *b = read_bool(r)?;
    }

    let prune_threshold = read_signal(r)?;
    let crystallization_threshold = read_signal(r)?;
    let allow_warming = read_bool(r)?;
    let warming_delta = read_signal(r)?;

    Ok(ThermalConfig {
        decay_rates,
        promotion_thresholds,
        demotion_thresholds,
        min_observations,
        allow_demotion,
        prune_threshold,
        crystallization_threshold,
        allow_warming,
        warming_delta,
    })
}

fn write_metadata(w: &mut Vec<u8>, meta: &ThermogramMetadata) -> io::Result<()> {
    write_timestamp(w, &meta.created_at)?;
    write_timestamp(w, &meta.last_consolidation)?;
    write_u32(w, meta.total_deltas as u32)?;
    write_u32(w, meta.total_consolidations as u32)?;

    // Custom blob
    match &meta.custom {
        Some(data) => {
            write_u8(w, 1)?;
            write_u32(w, data.len() as u32)?;
            write_bytes(w, data)?;
        }
        None => {
            write_u8(w, 0)?;
        }
    }
    Ok(())
}

fn read_metadata(r: &mut Cursor<&[u8]>) -> Result<ThermogramMetadata> {
    let created_at = read_timestamp(r)?;
    let last_consolidation = read_timestamp(r)?;
    let total_deltas = read_u32(r)? as usize;
    let total_consolidations = read_u32(r)? as usize;

    let has_custom = read_u8(r)?;
    let custom = if has_custom == 1 {
        let len = read_u32(r)? as usize;
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)
            .map_err(|e| Error::Deserialization(format!("custom blob: {}", e)))?;
        Some(buf)
    } else {
        None
    };

    Ok(ThermogramMetadata {
        created_at,
        last_consolidation,
        total_deltas,
        total_consolidations,
        custom,
    })
}

fn write_plasticity_rule(w: &mut Vec<u8>, rule: &PlasticityRule) -> io::Result<()> {
    let policy_byte = match rule.policy {
        UpdatePolicy::STDP => 0u8,
        UpdatePolicy::Replace => 1,
        UpdatePolicy::EMA => 2,
        UpdatePolicy::Bayesian => 3,
        UpdatePolicy::WTA => 4,
    };
    write_u8(w, policy_byte)?;
    write_signal(w, rule.novelty_threshold)?;
    write_signal(w, rule.merge_threshold)?;
    write_signal(w, rule.decay_rate)?;
    write_signal(w, rule.prune_threshold)?;
    write_signal(w, rule.learning_rate)?;
    Ok(())
}

fn read_plasticity_rule(r: &mut Cursor<&[u8]>) -> Result<PlasticityRule> {
    let policy_byte = read_u8(r)?;
    let policy = match policy_byte {
        0 => UpdatePolicy::STDP,
        1 => UpdatePolicy::Replace,
        2 => UpdatePolicy::EMA,
        3 => UpdatePolicy::Bayesian,
        4 => UpdatePolicy::WTA,
        _ => return Err(Error::Deserialization(format!("unknown policy: {}", policy_byte))),
    };

    Ok(PlasticityRule {
        policy,
        novelty_threshold: read_signal(r)?,
        merge_threshold: read_signal(r)?,
        decay_rate: read_signal(r)?,
        prune_threshold: read_signal(r)?,
        learning_rate: read_signal(r)?,
    })
}

fn write_consolidation_policy(w: &mut Vec<u8>, policy: &ConsolidationPolicy) -> io::Result<()> {
    write_bool(w, policy.enable_pruning)?;
    write_bool(w, policy.enable_merging)?;
    write_u8(w, policy.triggers.len() as u8)?;
    for trigger in &policy.triggers {
        match trigger {
            ConsolidationTrigger::DeltaCount(n) => {
                write_u8(w, 0)?;
                write_u32(w, *n as u32)?;
            }
            ConsolidationTrigger::TimePeriod { hours } => {
                write_u8(w, 1)?;
                write_u64(w, *hours)?;
            }
            ConsolidationTrigger::DirtySize { bytes } => {
                write_u8(w, 2)?;
                write_u32(w, *bytes as u32)?;
            }
            ConsolidationTrigger::Manual => {
                write_u8(w, 3)?;
            }
        }
    }
    Ok(())
}

fn read_consolidation_policy(r: &mut Cursor<&[u8]>) -> Result<ConsolidationPolicy> {
    let enable_pruning = read_bool(r)?;
    let enable_merging = read_bool(r)?;
    let trigger_count = read_u8(r)? as usize;
    let mut triggers = Vec::with_capacity(trigger_count);
    for _ in 0..trigger_count {
        let tag = read_u8(r)?;
        let trigger = match tag {
            0 => ConsolidationTrigger::DeltaCount(read_u32(r)? as usize),
            1 => ConsolidationTrigger::TimePeriod { hours: read_u64(r)? },
            2 => ConsolidationTrigger::DirtySize { bytes: read_u32(r)? as usize },
            3 => ConsolidationTrigger::Manual,
            _ => return Err(Error::Deserialization(format!("unknown trigger type: {}", tag))),
        };
        triggers.push(trigger);
    }

    Ok(ConsolidationPolicy {
        triggers,
        enable_pruning,
        enable_merging,
    })
}

fn write_entry_layer(
    w: &mut Vec<u8>,
    entries: &std::collections::HashMap<String, ConsolidatedEntry>,
) -> io::Result<()> {
    write_u32(w, entries.len() as u32)?;
    for entry in entries.values() {
        write_str(w, &entry.key)?;
        write_signals(w, &entry.value)?;
        write_signal(w, entry.strength)?;
        write_timestamp(w, &entry.updated_at)?;
        write_u32(w, entry.update_count as u32)?;
    }
    Ok(())
}

fn read_entry_layer(
    r: &mut Cursor<&[u8]>,
) -> Result<std::collections::HashMap<String, ConsolidatedEntry>> {
    let count = read_u32(r)? as usize;
    let mut entries = std::collections::HashMap::with_capacity(count);
    for _ in 0..count {
        let key = read_str(r)?;
        let value = read_signals(r)?;
        let strength = read_signal(r)?;
        let updated_at = read_timestamp(r)?;
        let update_count = read_u32(r)? as usize;

        entries.insert(
            key.clone(),
            ConsolidatedEntry {
                key,
                value,
                strength,
                updated_at,
                update_count,
            },
        );
    }
    Ok(entries)
}

fn write_hash_chain(w: &mut Vec<u8>, chain: &HashChain) -> io::Result<()> {
    write_u32(w, chain.deltas.len() as u32)?;
    write_opt_str(w, &chain.head_hash)?;

    for delta in &chain.deltas {
        write_str(w, &delta.id)?;

        let dt_byte = match delta.delta_type {
            DeltaType::Create => 0u8,
            DeltaType::Update => 1,
            DeltaType::Delete => 2,
            DeltaType::Merge => 3,
        };
        write_u8(w, dt_byte)?;

        write_str(w, &delta.key)?;
        write_signals(w, &delta.value)?;

        // Metadata
        write_timestamp(w, &delta.metadata.timestamp)?;
        write_str(w, &delta.metadata.source)?;
        write_signal(w, delta.metadata.strength)?;

        // observation_count
        match delta.metadata.observation_count {
            Some(n) => {
                write_u8(w, 1)?;
                write_u32(w, n as u32)?;
            }
            None => {
                write_u8(w, 0)?;
            }
        }

        // custom blob
        match &delta.metadata.custom {
            Some(data) => {
                write_u8(w, 1)?;
                write_u32(w, data.len() as u32)?;
                write_bytes(w, data)?;
            }
            None => {
                write_u8(w, 0)?;
            }
        }

        // prev_hash
        write_opt_str(w, &delta.prev_hash)?;

        // hash
        write_str(w, &delta.hash)?;
    }
    Ok(())
}

fn read_hash_chain(r: &mut Cursor<&[u8]>) -> Result<HashChain> {
    let delta_count = read_u32(r)? as usize;
    let head_hash = read_opt_str(r)?;

    let mut deltas = Vec::with_capacity(delta_count);
    for _ in 0..delta_count {
        let id = read_str(r)?;

        let dt_byte = read_u8(r)?;
        let delta_type = match dt_byte {
            0 => DeltaType::Create,
            1 => DeltaType::Update,
            2 => DeltaType::Delete,
            3 => DeltaType::Merge,
            _ => return Err(Error::Deserialization(format!("unknown delta type: {}", dt_byte))),
        };

        let key = read_str(r)?;
        let value = read_signals(r)?;

        // Metadata
        let timestamp = read_timestamp(r)?;
        let source = read_str(r)?;
        let strength = read_signal(r)?;

        let has_obs = read_u8(r)?;
        let observation_count = if has_obs == 1 {
            Some(read_u32(r)? as usize)
        } else {
            None
        };

        let has_custom = read_u8(r)?;
        let custom = if has_custom == 1 {
            let len = read_u32(r)? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)
                .map_err(|e| Error::Deserialization(format!("delta custom: {}", e)))?;
            Some(buf)
        } else {
            None
        };

        let prev_hash = read_opt_str(r)?;
        let hash = read_str(r)?;

        deltas.push(Delta {
            id,
            delta_type,
            key,
            value,
            metadata: DeltaMetadata {
                timestamp,
                source,
                strength,
                observation_count,
                custom,
            },
            prev_hash,
            hash,
        });
    }

    Ok(HashChain { deltas, head_hash })
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plasticity::PlasticityRule;

    #[test]
    fn test_round_trip_empty() {
        let thermo = Thermogram::new("empty_test", PlasticityRule::stdp_like());
        let encoded = encode(&thermo).unwrap();

        // Verify magic
        assert_eq!(&encoded[0..4], b"THRM");
        // Verify version
        assert_eq!(u16::from_le_bytes([encoded[4], encoded[5]]), 1);

        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded.name, "empty_test");
        assert!(decoded.hot_entries.is_empty());
        assert!(decoded.cold_entries.is_empty());
        assert!(decoded.dirty_chain.is_empty());
    }

    #[test]
    fn test_round_trip_with_entries() {
        let mut thermo = Thermogram::new("entries_test", PlasticityRule::stdp_like());

        // Add entries to different layers
        thermo.hot_entries.insert(
            "hot_key".to_string(),
            ConsolidatedEntry {
                key: "hot_key".to_string(),
                value: vec![Signal::positive(100), Signal::negative(50)],
                strength: Signal::positive(200),
                updated_at: Utc::now(),
                update_count: 3,
            },
        );

        thermo.cold_entries.insert(
            "cold_key".to_string(),
            ConsolidatedEntry {
                key: "cold_key".to_string(),
                value: vec![Signal::positive(255), Signal::ZERO, Signal::negative(128)],
                strength: Signal::positive(240),
                updated_at: Utc::now(),
                update_count: 50,
            },
        );

        let encoded = encode(&thermo).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.name, "entries_test");
        assert_eq!(decoded.hot_entries.len(), 1);
        assert_eq!(decoded.cold_entries.len(), 1);

        let hot = decoded.hot_entries.get("hot_key").unwrap();
        assert_eq!(hot.value.len(), 2);
        assert_eq!(hot.value[0], Signal::positive(100));
        assert_eq!(hot.value[1], Signal::negative(50));
        assert_eq!(hot.strength, Signal::positive(200));
        assert_eq!(hot.update_count, 3);

        let cold = decoded.cold_entries.get("cold_key").unwrap();
        assert_eq!(cold.value.len(), 3);
        assert_eq!(cold.strength, Signal::positive(240));
    }

    #[test]
    fn test_round_trip_with_deltas() {
        let mut thermo = Thermogram::new("delta_test", PlasticityRule::stdp_like());

        let delta = crate::delta::Delta::create("key1", vec![Signal::positive(100)], "source");
        thermo.apply_delta(delta).unwrap();

        let encoded = encode(&thermo).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.dirty_chain.len(), 1);
        let d = &decoded.dirty_chain.deltas[0];
        assert_eq!(d.key, "key1");
        assert_eq!(d.value, vec![Signal::positive(100)]);
        assert!(decoded.dirty_chain.verify().is_ok());
    }

    #[test]
    fn test_binary_is_smaller_than_json() {
        let mut thermo = Thermogram::new("size_test", PlasticityRule::stdp_like());

        // Add 20 entries with 100 signals each
        for i in 0..20 {
            thermo.hot_entries.insert(
                format!("key_{}", i),
                ConsolidatedEntry {
                    key: format!("key_{}", i),
                    value: (0..100).map(|v| Signal::positive(v as u8)).collect(),
                    strength: Signal::positive(200),
                    updated_at: Utc::now(),
                    update_count: i + 1,
                },
            );
        }

        let binary = encode(&thermo).unwrap();
        let json = serde_json::to_string_pretty(&thermo).unwrap();

        assert!(
            binary.len() < json.len(),
            "binary ({} bytes) should be smaller than JSON ({} bytes)",
            binary.len(),
            json.len()
        );

        // Binary should be at least 10x smaller for signal-heavy payloads
        let ratio = json.len() as f64 / binary.len() as f64;
        assert!(ratio > 10.0, "expected >10x reduction, got {:.1}x", ratio);

        // Verify round-trip
        let decoded = decode(&binary).unwrap();
        assert_eq!(decoded.hot_entries.len(), 20);
    }

    #[test]
    fn test_corrupted_magic_rejected() {
        let thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        let mut encoded = encode(&thermo).unwrap();
        encoded[0] = b'X'; // Corrupt magic
        assert!(decode(&encoded).is_err());
    }

    #[test]
    fn test_corrupted_checksum_rejected() {
        let thermo = Thermogram::new("test", PlasticityRule::stdp_like());
        let mut encoded = encode(&thermo).unwrap();
        // Corrupt a byte in the body (after header)
        if encoded.len() > 20 {
            encoded[20] ^= 0xFF;
        }
        assert!(decode(&encoded).is_err());
    }

    #[test]
    fn test_all_thermal_configs() {
        for config_fn in [ThermalConfig::default, ThermalConfig::fast_learner, ThermalConfig::organic] {
            let mut thermo = Thermogram::new("config_test", PlasticityRule::stdp_like());
            thermo.thermal_config = config_fn();

            let encoded = encode(&thermo).unwrap();
            let decoded = decode(&encoded).unwrap();

            for i in 0..4 {
                assert_eq!(decoded.thermal_config.decay_rates[i], thermo.thermal_config.decay_rates[i]);
                assert_eq!(decoded.thermal_config.promotion_thresholds[i], thermo.thermal_config.promotion_thresholds[i]);
                assert_eq!(decoded.thermal_config.demotion_thresholds[i], thermo.thermal_config.demotion_thresholds[i]);
                assert_eq!(decoded.thermal_config.min_observations[i], thermo.thermal_config.min_observations[i]);
                assert_eq!(decoded.thermal_config.allow_demotion[i], thermo.thermal_config.allow_demotion[i]);
            }
            assert_eq!(decoded.thermal_config.prune_threshold, thermo.thermal_config.prune_threshold);
            assert_eq!(decoded.thermal_config.allow_warming, thermo.thermal_config.allow_warming);
        }
    }

    #[test]
    fn test_all_plasticity_rules() {
        for rule_fn in [PlasticityRule::stdp_like, PlasticityRule::conservative, PlasticityRule::aggressive] {
            let mut thermo = Thermogram::new("rule_test", rule_fn());
            let encoded = encode(&thermo).unwrap();
            let decoded = decode(&encoded).unwrap();

            assert_eq!(decoded.plasticity_rule.policy, thermo.plasticity_rule.policy);
            assert_eq!(decoded.plasticity_rule.novelty_threshold, thermo.plasticity_rule.novelty_threshold);
            assert_eq!(decoded.plasticity_rule.learning_rate, thermo.plasticity_rule.learning_rate);
        }
    }
}

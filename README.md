# Thermogram

**Signal-native plastic memory capsules with 4-temperature tensor states, embedded SNN plasticity, and binary persistence.**

Thermogram is a synaptic storage system that mimics biological memory consolidation. All values use [`ternary-signal::Signal`](https://crates.io/crates/ternary-signal) — a 2-byte `(polarity: i8, magnitude: u8)` type. No floats in storage.

## Core Features

- **4-Temperature States** — Hot (working) / Warm (session) / Cool (expertise) / Cold (identity)
- **Bidirectional Flow** — Entries cement forward when reinforced, degrade backward when unused
- **Signal-Native** — All strengths, thresholds, and values use `Signal` (2 bytes). Zero floats in storage.
- **Binary `.thermo` Format** — CRC32-verified, ~30x smaller than JSON. `THRM` magic header.
- **Embedded SNN** — Runtime STDP plasticity, homeostasis, competition, neuromodulation
- **Hash-Chained Audit Trail** — SHA-256 chain over all deltas
- **Colonies** — Multiple thermograms per mesh that grow, split, merge
- **Distillation** — Share semantic deltas across instances

## Installation

```toml
[dependencies]
thermogram = "0.5"
ternary-signal = "0.1"
```

## Quick Start

```rust
use thermogram::{Thermogram, PlasticityRule, Delta, Signal};

// Create thermogram with STDP plasticity
let mut thermo = Thermogram::new("neural_weights", PlasticityRule::stdp_like());

// Store signal vectors (2 bytes per Signal, not JSON text)
let weights: Vec<Signal> = (0..64).map(|i| Signal::positive(i as u8)).collect();
let delta = Delta::update(
    "layer_0",
    weights,
    "training",
    Signal::positive(204), // ~0.8 strength
    thermo.dirty_chain.head_hash.clone(),
);
thermo.apply_delta(delta)?;

// Reinforce successful patterns (promotes toward Cold)
thermo.reinforce("layer_0", Signal::positive(51)); // +0.2

// Run thermal transitions (promotes/demotes based on strength)
thermo.run_thermal_transitions()?;

// Consolidate (dirty -> clean state)
thermo.consolidate()?;

// Save as binary .thermo v1 (CRC32 verified)
thermo.save("neural_weights.thermo")?;

// Load (auto-detects binary vs legacy JSON)
let loaded = Thermogram::load("neural_weights.thermo")?;
```

## The 4-Temperature Model

| Temperature | Analog | Decay Rate | Behavior |
|-------------|--------|------------|----------|
| **Hot** | Working memory | Fast | Volatile, immediate task, rebuilt on boot |
| **Warm** | Short-term | Medium | Session learning, persists across tasks |
| **Cool** | Procedural/skill | Slow | Expertise, long-term mastery |
| **Cold** | Core identity | Glacial | Personality backbone, constitutional |

```
HOT <-> WARM <-> COOL <-> COLD
 |        |        |        |
fast    medium   slow    glacial
decay   decay    decay   decay
```

- **Cement forward**: `reinforce()` strengthens entries, promotes to colder layers
- **Degrade backward**: `weaken()` or natural decay demotes to hotter layers
- **Automatic transitions**: `run_thermal_transitions()` handles promotion/demotion based on configurable thresholds

### Thermal Configurations

```rust
let config = ThermalConfig::default();      // Balanced defaults
let config = ThermalConfig::fast_learner(); // Faster promotion, for agents
let config = ThermalConfig::organic();      // Slow, gradual emergence
```

## Binary Format (`.thermo` v1)

All persistence uses a custom binary codec — not JSON, not bincode.

```
[Header 16B: THRM + version + size + CRC32]
[Identity] [ThermalConfig] [Metadata] [PlasticityRule]
[ConsolidationPolicy] [Entries x4 layers] [HashChain]
```

**Signal encoding**: 2 bytes per Signal (`[polarity as u8, magnitude]`).
Strings are length-prefixed UTF-8. Integers are little-endian.
Timestamps are `(i64 seconds, u32 nanoseconds)` for hash-stable round-trips.

Legacy JSON files are detected and loaded automatically (no `THRM` magic = JSON fallback).

### Size Comparison (measured)

| Payload | Binary | JSON | Reduction |
|---------|--------|------|-----------|
| 20 entries x 100 signals | 4,804 B | 150,702 B | **31x** |
| 100 entries x 64 signals | 16,906 B | 486,747 B | **29x** |
| Empty thermogram | 206 B | 2,393 B | **12x** |

## Embedded SNN — Runtime Plasticity

Each thermogram can host an embedded spiking neural network (SNN) that actively shapes connections at runtime:

```rust
use thermogram::{EmbeddedSNN, EmbeddedSNNConfig, NeuromodState};

let config = EmbeddedSNNConfig {
    num_neurons: 100,
    input_dim: 2048,
    stdp_lr: 0.01,
    homeostasis_target: 0.1,
    competition_strength: 0.5,
    decay_rate: 0.001,
    use_ternary: true,
    ..Default::default()
};
let mut snn = EmbeddedSNN::new(config);

let mut neuromod = NeuromodState::baseline();
neuromod.reward(0.3); // Dopamine spike -> increases learning rate

let activation = vec![0.5; 2048]; // From your model's hidden state
let deltas = snn.process(&activation, &neuromod)?;

for delta in deltas {
    thermogram.apply_delta(delta)?;
}
```

### Neuromodulation

| Neuromodulator | Effect on Plasticity |
|----------------|---------------------|
| **Dopamine** | Learning rate multiplier (reward signal) |
| **Serotonin** | Decay rate modulation (confidence) |
| **Norepinephrine** | Competition strength (arousal/attention) |
| **Acetylcholine** | Gating modulation (focus) |

```rust
let mut neuromod = NeuromodState::baseline();
neuromod.reward(0.3);  // Dopamine up -> faster learning
neuromod.stress(0.2);  // Serotonin down, NE up -> faster forgetting, more competition
neuromod.focus(0.2);   // Acetylcholine up -> sharper attention gating
neuromod.decay(0.1);   // Natural decay back to baseline
```

## Performance (criterion benchmarks)

Measured on Windows x86_64, release profile:

### Core Operations

| Operation | Time | Throughput |
|-----------|------|------------|
| Delta create | 525 ns | 1.9M ops/sec |
| Hash verify (SHA-256) | 297 ns | 3.4M ops/sec |
| Plasticity update (STDP) | 3.2 ns | 312M ops/sec |
| Plasticity update (EMA) | 2.7 ns | 370M ops/sec |
| Neuromod sync | 4.0 ns | 250M ops/sec |
| Read (clean, 1K entries) | 50 ns | 20M ops/sec |
| Read (dirty fallback) | 47 ns | 21M ops/sec |

### Scaling

| Operation | 10 | 100 | 1,000 | 10,000 |
|-----------|-----|------|-------|--------|
| Delta append | 2.4 us | 10 us | 40 us | 425 us |
| Consolidation | 2.7 us | 24 us | 4.1 us* | — |
| SNN tick (neurons) | 14 us | 72 us | 142 us | 284 us |

*1000-delta consolidation is fast because overlapping keys reduce the entry map size.

### Binary Codec

| Entries (64 signals each) | Encode | Decode |
|---------------------------|--------|--------|
| 10 | 1.6 us | 4.6 us |
| 100 | 15 us | 44 us |
| 1,000 | 152 us | 450 us |

Run benchmarks: `cargo bench --bench core_operations`

## Testing

**97 tests passing** (73 unit + 15 adversarial + 8 concurrency + 1 doc-test)

- Binary codec round-trip (all configs, all plasticity rules, entries + deltas)
- Corruption detection (magic tampering, CRC32 mismatch)
- Hash chain tamper detection
- Adversarial inputs (empty keys, 10M signal payloads, NaN/Inf injection)
- Concurrency (parallel writes, concurrent save/load, consolidation under read)
- Colony split/merge/balance
- Distillation delta sharing

```bash
cargo test              # All tests
cargo test -- --nocapture  # With output
cargo bench             # Criterion benchmarks
```

## Architecture

```
thermogram/
  src/
    core.rs           # Thermogram struct, 4-temp layers, save/load
    codec.rs          # Binary .thermo v1 encoder/decoder
    delta.rs          # Delta (append-only change record)
    hash_chain.rs     # SHA-256 chained audit trail
    plasticity.rs     # STDP/EMA/Bayesian/WTA plasticity rules
    consolidation.rs  # Dirty -> clean state transitions
    embedded_snn.rs   # Spiking neural network plasticity engine
    plasticity_engine.rs # Neuromodulation state + sync
    colony.rs         # Multi-thermogram management
    distillation.rs   # Semantic delta sharing across instances
    ternary.rs        # TernaryWeight, PackedTernary, TernaryLayer
    export.rs         # JSON + Engram archive export
    error.rs          # Error types
```

## Signal Type

All stored values use `ternary-signal::Signal`:

```rust
#[repr(C)]
pub struct Signal {
    pub polarity: i8,   // -1, 0, +1
    pub magnitude: u8,  // 0-255 intensity
}
```

2 bytes on disk. No float conversions in the storage path.

Strength values that were previously `f32` in `[0.0, 1.0]` are now `Signal::positive(magnitude)` where `magnitude = (f32 * 255) as u8`.

## License

MIT OR Apache-2.0

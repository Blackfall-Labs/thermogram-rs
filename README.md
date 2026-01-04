# Thermogram 🧠

**Living knowledge files with embedded SNN plasticity**

Thermogram is a plastic memory capsule that combines:

- **Dirty/Clean dual states** - Fast mutable deltas + consolidated snapshots
- **Embedded SNN plasticity engine** - STDP, homeostasis, competition, decay
- **Hash-chained audit trail** - Cryptographic verification of all changes
- **Optional neuromod sync** - Colony-wide chemical balance coordination
- **Consolidation cycles** - Brain-like sleep/replay
- **Engram export** - Archive to immutable format without deletion

## The Problem

Traditional storage is either:

- **Mutable** (databases, files) - Fast but no audit trail, hard to prove integrity
- **Immutable** (Git, Engram) - Auditable but can't evolve organically

Brains don't work like either - they're **plastic with constraints**. Connections strengthen and weaken based on rules (STDP, homeostasis), not arbitrary writes.

## The Solution

Thermogram is a **single file** that contains:

1. **Clean state** - Consolidated snapshot (prototypes, weights, indexes)
2. **Dirty state** - Append-only delta log (hash-chained)
3. **Plasticity engine** - Small SNN that generates deltas via spiking dynamics
4. **Neuromodulation** - Optional sync with external chemical balance

```
┌─────────────────────────────────────────────────────┐
│ Thermogram: llm_clusters.thermo                    │
├─────────────────────────────────────────────────────┤
│ Clean State (Consolidated)                         │
│  ├─ Concept Prototypes [100 x 2048]                │
│  ├─ Associative Weights (sparse)                   │
│  └─ Indexes (ANN, routing tables)                  │
├─────────────────────────────────────────────────────┤
│ Dirty State (Append-Only Deltas)                   │
│  ├─ DELTA_PROTO(cluster_5, Δvector, lr, evidence)  │
│  ├─ DELTA_EDGE(12, 34, Δw, stdp, evidence)         │
│  ├─ DECAY(epoch, params)                           │
│  └─ ... (hash-chained)                             │
├─────────────────────────────────────────────────────┤
│ Plasticity Engine (Embedded SNN)                   │
│  ├─ 100 neurons (spiking)                          │
│  ├─ STDP: cells that fire together wire together   │
│  ├─ Homeostasis: prevent runaway strengthening     │
│  ├─ Competition: winner-take-most                  │
│  ├─ Decay: natural forgetting                      │
│  └─ Gating: context-dependent activation           │
├─────────────────────────────────────────────────────┤
│ Neuromodulation (Synced or Independent)            │
│  ├─ Dopamine: 0.6 (learning rate ↑)                │
│  ├─ Serotonin: 0.5 (decay rate ←)                  │
│  ├─ Norepinephrine: 0.4 (competition ↓)            │
│  └─ Acetylcholine: 0.7 (attention ↑)               │
└─────────────────────────────────────────────────────┘
```

## Colony Architecture

Multiple Thermograms can coexist as **independent organisms** or **sync neuromodulation** for colony-wide coordination:

```
Astromind (Central)
  ├─ InnerPilot SNN
  │   └─ Neuromodulation: {dopamine, serotonin, norepinephrine, acetylcholine}
  │
  └─ Thermogram Colony
      ├─ llm_clusters.thermo        ← syncs neuromod
      ├─ dialogue_patterns.thermo   ← syncs neuromod
      ├─ trust_graph.thermo         ← independent
      └─ code_knowledge.thermo      ← independent

// Dopamine spike in Astromind → all synced Thermograms increase learning rate
// Stress → all synced Thermograms increase decay, prune weak connections
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
thermogram = "0.1"
```

Or with optional Engram integration:

```toml
[dependencies]
thermogram = { version = "0.1", features = ["engram-export"] }
```

## Usage

### Basic Example

```rust
use thermogram::{Thermogram, PlasticityRule, Delta};

// Create Thermogram for LLM activation clusters
let mut thermo = Thermogram::new("llm_clusters", PlasticityRule::stdp_like());

// Apply delta (e.g., cluster centroid update from LLM mining)
let new_centroid = vec![0.5_f32; 2048];
let delta = Delta::update(
    "cluster_0",
    bincode::serialize(&new_centroid)?,
    "llm_mining",
    0.8, // strength
    thermo.dirty_chain.head_hash.clone(),
);
thermo.apply_delta(delta)?;

// Read current state (dirty + clean merged)
let cluster = thermo.read("cluster_0")?;

// Manual consolidation (or auto-trigger after N deltas)
let result = thermo.consolidate()?;
println!("Consolidated {} entries", result.entries_consolidated);

// Save to disk (hot file)
thermo.save("data/llm_clusters.thermo")?;

// Export to JSON
thermo.export_to_json("exports/llm_knowledge_v1.json")?;
```

### With Embedded SNN Plasticity

```rust
use thermogram::{EmbeddedSNN, EmbeddedSNNConfig, NeuromodState, PlasticityEngine};

// Create SNN plasticity engine
let config = EmbeddedSNNConfig::default(); // 100 neurons
let mut snn = EmbeddedSNN::new(config);

// Process activation from LLM layer
let activation = vec![0.5; 2048]; // From layer 16 hidden state
let neuromod = NeuromodState::baseline();

// SNN generates deltas via plasticity rules:
// - STDP strengthens co-firing neurons
// - Homeostasis prevents runaway
// - Competition enforces sparsity
// - Decay prunes weak connections
let deltas = snn.process(&activation, &neuromod)?;

// Apply deltas to Thermogram
for delta in deltas {
    thermo.apply_delta(delta)?;
}
```

### Colony with Neuromod Sync

```rust
use thermogram::{EmbeddedSNN, EmbeddedSNNConfig, NeuromodState, NeuromodSyncConfig, PlasticityEngine};

// Central neuromodulation state (from Astromind)
let mut central_neuromod = NeuromodState::baseline();

// SNN 1: Full sync with central state
let config1 = EmbeddedSNNConfig::default();
let mut snn1 = EmbeddedSNN::new(config1);

// SNN 2: Independent (sync_rate = 0.0)
let config2 = EmbeddedSNNConfig::default();
let mut snn2 = EmbeddedSNN::new(config2);

// Reward signal → increase dopamine
central_neuromod.reward(0.3);

// Sync to colony
snn1.sync_neuromod(&central_neuromod); // Picks up dopamine spike
// snn2 runs independently with NeuromodState::baseline()

// Process activations with synchronized neuromod
let activation = vec![0.5; 2048];
let deltas1 = snn1.process(&activation, &central_neuromod)?; // Uses synced state
let deltas2 = snn2.process(&activation, &NeuromodState::baseline())?; // Independent
```

## Key Features

### 1. Plastic Memory with Audit Trail

- Every change is a **rule-governed delta** (not arbitrary overwrites)
- Hash-chained for tamper evidence
- Can replay full history to verify integrity

### 2. Brain-Like Plasticity

- **STDP**: Cells that fire together wire together
- **Homeostasis**: Prevent runaway strengthening
- **Competition**: Winner-take-most (enforces sparsity)
- **Decay**: Natural forgetting of unused connections
- **Gating**: Context-dependent activation

### 3. Hot File Format

- Lives on disk as active, processable file
- Can be opened, processed, consolidated, closed
- ThermogramManager handles colony lifecycle
- Automatic consolidation triggers

### 4. Scalable Colonies

- Each Thermogram specializes in a domain
- Independent or synchronized neuromodulation
- Add new Thermograms without touching existing ones
- Colony can scale to hundreds of specialized memories

### 5. Archive Without Deletion

- Export to Engram (immutable) when saturated
- Keep full learning history
- Never lose the "why" and "how"
- Cold storage for non-active knowledge

## Architecture Decisions

See `engineering/` for detailed design rationale:

- **Colony architecture**: Independent organisms with optional chemical sync
- **Embedded SNN**: Why spiking dynamics instead of backprop
- **Dual state**: Why dirty/clean instead of single mutable state
- **Hash chains**: Why cryptographic audit trail matters
- **Plasticity rules**: How STDP/homeostasis/competition interact

## Performance

**CPU-only, production-ready** ⚡

| Operation         | Time                | Throughput   |
| ----------------- | ------------------- | ------------ |
| **Read**          | 17-59ns             | 17M+ ops/sec |
| **Write (delta)** | 660ns               | 1.5M ops/sec |
| **Consolidation** | 17µs (1000 deltas)  | 60K/sec      |
| **SNN tick**      | 151µs (100 neurons) | 6.6K/sec     |
| **Neuromod sync** | 4ns                 | 244M/sec     |

- **No GPU required** - Pure Rust, runs anywhere
- **Low memory** - 1-10MB per Thermogram
- **Edge-friendly** - Suitable for embedded/offline deployments

See [PERFORMANCE.md](PERFORMANCE.md) for comprehensive benchmarks and scaling analysis.

## Testing & Security

**60/60 tests passing** ✅

- **36 unit tests** - Core functionality
- **15 adversarial tests** - Hash tampering, invalid chains, corrupted deltas, fork attacks, NaN/Inf protection
- **8 concurrency tests** - Thread safety, race conditions, state isolation
- **1 doc test** - API correctness

**Security verified:**

- ✅ SHA-256 hash-chained audit trail
- ✅ Tamper detection at every delta link
- ✅ Fork attack prevention
- ✅ Corruption detection through hash recomputation
- ✅ NaN/Inf protection in SNN

See [HARDENING_SUMMARY.md](HARDENING_SUMMARY.md) for complete testing report.

## Status

**v0.1.0 - Production Ready** 🚀

- ✅ Core delta/hash chain/consolidation
- ✅ Embedded SNN with STDP/homeostasis/competition/decay
- ✅ Neuromodulation with optional sync
- ✅ Save/load to disk
- ✅ JSON export
- ✅ **Comprehensive testing** (60/60 passing)
- ✅ **Benchmarked** (8 benchmark suites)
- ✅ **Security verified** (adversarial testing)
- ✅ **Performance documented** (see PERFORMANCE.md)
- 🚧 ThermogramManager (colony lifecycle)
- 🚧 Engram export (requires engram-rs integration)
- 🚧 Astromind integration (next phase)

## Next Steps

1. **Astromind Integration** - Wire into LLM activation mining pipeline
2. **ThermogramManager** - Manage colony lifecycle, auto-consolidation triggers
3. **Engram Export** - Consolidate to immutable archive format
4. **Real-world testing** - Use for actual LLM knowledge extraction on Qwen-2.5-3B

## License

MIT OR Apache-2.0

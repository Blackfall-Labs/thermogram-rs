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

## Usage

### Basic Example

```rust
use thermogram::{Thermogram, PlasticityRule, EmbeddedSNN, EmbeddedSNNConfig};

// Create Thermogram for LLM activation clusters
let mut thermo = Thermogram::new_with_snn(
    "llm_clusters",
    PlasticityRule::stdp_like(),
    EmbeddedSNNConfig::default(),
);

// Process activation from LLM mining
let activation = vec![0.5; 2048]; // From layer 16 hidden state
thermo.process_activation(&activation)?;

// Deltas generated automatically via SNN plasticity:
// - STDP strengthens co-firing neurons
// - Homeostasis prevents runaway
// - Competition enforces sparsity
// - Decay prunes weak connections

// Read current state (dirty + clean merged)
let cluster = thermo.read("cluster_5")?;

// Manual consolidation (or auto-trigger after N deltas)
let result = thermo.consolidate()?;
println!("Pruned {} weak connections", result.entries_pruned);

// Save to disk (hot file)
thermo.save("data/llm_clusters.thermo")?;

// Export to Engram (immutable archive)
thermo.export_to_json("exports/llm_knowledge_v1.json")?;
```

### Colony with Neuromod Sync

```rust
use thermogram::{NeuromodState, NeuromodSyncConfig};

// Central neuromodulation state (from Astromind)
let mut central_neuromod = NeuromodState::baseline();

// Thermogram 1: Synced
let mut thermo1 = Thermogram::new_with_snn(
    "dialogue",
    PlasticityRule::stdp_like(),
    EmbeddedSNNConfig::default(),
);
thermo1.set_neuromod_sync(NeuromodSyncConfig::full_sync());

// Thermogram 2: Independent
let mut thermo2 = Thermogram::new_with_snn(
    "trust",
    PlasticityRule::conservative(),
    EmbeddedSNNConfig::default(),
);
thermo2.set_neuromod_sync(NeuromodSyncConfig::independent());

// Reward signal → increase dopamine
central_neuromod.reward(0.3);

// Sync to colony
thermo1.sync_neuromod(&central_neuromod); // Picks up dopamine spike
thermo2.sync_neuromod(&central_neuromod); // Ignores (independent)
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

## Status

- ✅ Core delta/hash chain/consolidation
- ✅ Embedded SNN with STDP/homeostasis/competition/decay
- ✅ Neuromodulation with optional sync
- ✅ Save/load to disk
- ✅ JSON export
- 🚧 ThermogramManager (colony lifecycle)
- 🚧 Engram export (requires engram-rs integration)
- 🚧 Astromind integration

## Next Steps

1. **ThermogramManager** - Manage colony lifecycle, auto-consolidation
2. **Astromind Integration** - Wire into LLM activation mining pipeline
3. **Benchmarks** - Measure plasticity overhead, consolidation performance
4. **Real-world testing** - Use for actual LLM knowledge extraction

## License

MIT OR Apache-2.0

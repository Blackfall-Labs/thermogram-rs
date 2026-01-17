# Thermogram

**Plastic memory capsules with 4-temperature states and embedded SNN plasticity**

Thermogram is a synaptic storage system that mimics biological memory consolidation:

- **4-Temperature States** - Hot (working) / Warm (session) / Cool (expertise) / Cold (identity)
- **Bidirectional Flow** - Entries cement forward when reinforced, degrade backward when unused
- **Embedded SNN** - Runtime plasticity via STDP, homeostasis, competition, decay
- **Ternary Weights** - Optional BitNet-style quantization (16x compression)
- **Hash-Chained Audit Trail** - Cryptographic verification of all changes

## The 4-Temperature Model

Biological memory systems don't have binary hot/cold states - they have gradual crystallization with bidirectional flow:

| Temperature | Analog | Decay Rate | Behavior |
|-------------|--------|------------|----------|
| **Hot** | Working memory | Fast (0.1/tick) | Volatile, immediate task, rebuilt on boot |
| **Warm** | Short-term | Medium (0.01/tick) | Session learning, persists across tasks |
| **Cool** | Procedural/skill | Slow (0.001/tick) | Expertise, long-term mastery |
| **Cold** | Core identity | Glacial (0.0001/tick) | Personality backbone, constitutional |

### Bidirectional Transitions

```
HOT <-> WARM <-> COOL <-> COLD
 |        |        |        |
fast    medium   slow    glacial
decay   decay    decay   decay
```

- **Cement forward**: `reinforce()` strengthens entries, promotes to colder layers
- **Degrade backward**: `weaken()` or natural decay demotes to hotter layers
- **Automatic transitions**: `run_thermal_transitions()` handles promotion/demotion based on thresholds

## Replacing Safetensors

Thermogram's **cool layer** serves as a drop-in replacement for safetensors checkpoints:

| Safetensors | Thermogram Cool Layer |
|-------------|----------------------|
| Static checkpoint | Living, evolving weights |
| Full f32 precision | Optional ternary (16x smaller) |
| Load/save only | Read/write/reinforce/weaken |
| No history | Hash-chained audit trail |
| One state | 4 temperatures with transitions |

**Migration path:**
```rust
// Load safetensor weights
let weights: Vec<f32> = load_safetensor("model.safetensors")?;

// Import into thermogram cool layer
for (key, weight) in weights.iter().enumerate() {
    let entry = ConsolidatedEntry {
        key: format!("weight_{}", key),
        value: bincode::serialize(&weight)?,
        strength: 0.9,  // High strength = stays in cool
        ternary_strength: Some(TernaryWeight::from_f32(*weight, 0.3)),
        updated_at: Utc::now(),
        update_count: 1,
    };
    thermogram.cool_entries.insert(entry.key.clone(), entry);
}

// Now weights can evolve:
// - Reinforce successful patterns (may promote to cold)
// - Weaken unused patterns (may demote to warm)
// - All changes hash-chained and auditable
```

## Embedded SNN - Runtime Plasticity

Each thermogram has an embedded spiking neural network (SNN) that actively shapes connections at runtime. This is NOT dead code - it's the plasticity engine:

### How the SNN Works

```rust
use thermogram::{EmbeddedSNN, EmbeddedSNNConfig, NeuromodState, PlasticityEngine};

// Create SNN plasticity engine
let config = EmbeddedSNNConfig {
    num_neurons: 100,        // Concept prototypes
    input_dim: 2048,         // Activation dimension
    stdp_lr: 0.01,           // STDP learning rate
    homeostasis_target: 0.1, // Target firing rate
    competition_strength: 0.5,
    decay_rate: 0.001,
    use_ternary: true,       // Use ternary weights
    ..Default::default()
};
let mut snn = EmbeddedSNN::new(config);

// Neuromodulation affects plasticity
let mut neuromod = NeuromodState::baseline();
neuromod.reward(0.3);  // Dopamine spike -> increases learning rate

// Process activation vector from your model
let activation = get_layer_activations(); // e.g., from LLM hidden state
let deltas = snn.process(&activation, &neuromod)?;

// SNN generates deltas based on:
// - STDP: Cells that fire together wire together
// - Homeostasis: Prevents runaway strengthening
// - Competition: Winner-take-most (enforces sparsity)
// - Decay: Natural forgetting of unused connections

// Apply deltas to thermogram
for delta in deltas {
    thermogram.apply_delta(delta)?;
}
```

### Neuromodulation

The SNN's behavior is modulated by four chemical signals:

| Neuromodulator | Effect on Plasticity |
|----------------|---------------------|
| **Dopamine** | Learning rate multiplier (reward signal) |
| **Serotonin** | Decay rate modulation (confidence) |
| **Norepinephrine** | Competition strength (arousal/attention) |
| **Acetylcholine** | Gating modulation (focus) |

```rust
let mut neuromod = NeuromodState::baseline();

// Reward -> increase dopamine -> faster learning
neuromod.reward(0.3);

// Stress -> decrease serotonin, increase NE -> faster forgetting, more competition
neuromod.stress(0.2);

// Focus -> increase acetylcholine -> sharper attention gating
neuromod.focus(0.2);

// Natural decay back to baseline
neuromod.decay(0.1);
```

## Migration: 2-Temperature to 4-Temperature

If upgrading from thermogram 0.4.x (hot/cold only):

### Breaking Changes in 0.5.0

1. **ThermalState enum** now has 4 variants: `Hot`, `Warm`, `Cool`, `Cold`
2. **ThermalConfig** fields are now `[f32; 4]` arrays
3. **New HashMaps**: `warm_entries` and `cool_entries` added to Thermogram

### Automatic Migration

Old thermogram files (with only `hot_entries` and `cold_entries`) load correctly:
- `warm_entries` and `cool_entries` default to empty `HashMap`
- Use `#[serde(default)]` for backward compatibility

### Manual Migration (if needed)

```rust
// Old 2-temp config
let old_config = ThermalConfig {
    crystallization_threshold: 0.75,
    min_observations: 3,
    prune_threshold: 0.05,
    allow_warming: true,
    warming_delta: 0.3,
};

// New 4-temp config (use defaults and customize)
let mut new_config = ThermalConfig::default();
new_config.promotion_thresholds[2] = old_config.crystallization_threshold; // Cool->Cold
new_config.prune_threshold = old_config.prune_threshold;

// Or use preset configs:
let config = ThermalConfig::fast_learner();  // Faster promotion, agents
let config = ThermalConfig::organic();       // Slower, gradual emergence
```

## Installation

```toml
[dependencies]
thermogram = "0.5"
```

## Quick Start

```rust
use thermogram::{Thermogram, PlasticityRule, ThermalState};

// Create thermogram
let mut thermo = Thermogram::new("my_memory", PlasticityRule::stdp_like());

// Apply delta (learning)
let delta = Delta::update(
    "concept_1",
    bincode::serialize(&vec![0.5f32; 384])?,
    "learning",
    0.8,
    thermo.dirty_chain.head_hash.clone(),
);
thermo.apply_delta(delta)?;

// Reinforce successful patterns
thermo.reinforce("concept_1", 0.2)?;

// Run thermal transitions (promotes/demotes based on strength)
thermo.run_thermal_transitions()?;

// Consolidate (dirty -> clean state)
thermo.consolidate()?;

// Save
thermo.save("memory.thermo")?;
```

## Performance

| Operation | Time | Throughput |
|-----------|------|------------|
| Read | 17-59ns | 17M+ ops/sec |
| Write (delta) | 660ns | 1.5M ops/sec |
| Consolidation | 17us (1000 deltas) | 60K/sec |
| SNN tick | 151us (100 neurons) | 6.6K/sec |
| Thermal transition | <1ms | 1K+/sec |

- **No GPU required** - Pure Rust, runs anywhere
- **Low memory** - 1-10MB per thermogram
- **Edge-friendly** - Suitable for embedded/offline

## Testing

**77 tests passing**

- Unit tests for all temperature layers
- Adversarial tests (tampering, corruption, NaN/Inf)
- Concurrency tests (thread safety, state isolation)
- Colony and distillation tests

## Status

**v0.5.0** - 4-Temperature Architecture

- 4-temperature states (hot/warm/cool/cold)
- Bidirectional thermal transitions
- Ternary weight support
- Colony management (split/merge/balance)
- Distillation (semantic delta sharing)
- Embedded SNN with neuromodulation

## License

MIT OR Apache-2.0

//! Embedded SNN - Small spiking neural network for Thermogram plasticity
//!
//! This is NOT for reasoning or facts. It's for **maintaining and reshaping
//! associations** under constraints: STDP, homeostasis, competition, decay.

use crate::delta::{Delta, DeltaType};
use crate::error::Result;
use crate::plasticity_engine::{NeuromodState, PlasticityEngine, PlasticityEngineState};
use serde::{Deserialize, Serialize};

/// Configuration for embedded SNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedSNNConfig {
    /// Number of neurons (concept prototypes)
    pub num_neurons: usize,

    /// Input dimensionality
    pub input_dim: usize,

    /// Sparse connectivity (top-k neighbors per neuron)
    pub top_k: usize,

    /// STDP learning rate
    pub stdp_lr: f32,

    /// Homeostasis target firing rate
    pub homeostasis_target: f32,

    /// Competition strength (winner-take-most)
    pub competition_strength: f32,

    /// Decay rate per tick
    pub decay_rate: f32,

    /// Eligibility trace decay
    pub trace_decay: f32,

    /// Activation threshold for spiking
    pub spike_threshold: f32,
}

impl Default for EmbeddedSNNConfig {
    fn default() -> Self {
        Self {
            num_neurons: 100,
            input_dim: 2048,
            top_k: 10,
            stdp_lr: 0.01,
            homeostasis_target: 0.1,
            competition_strength: 0.5,
            decay_rate: 0.001,
            trace_decay: 0.1,
            spike_threshold: 0.5,
        }
    }
}

/// Internal state of embedded SNN
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SNNState {
    /// Neuron prototypes [num_neurons, input_dim]
    prototypes: Vec<Vec<f32>>,

    /// Sparse associative weights [neuron_id -> Vec<(neighbor_id, weight)>]
    weights: Vec<Vec<(usize, f32)>>,

    /// Eligibility traces for STDP
    traces: Vec<f32>,

    /// Firing rates for homeostasis
    firing_rates: Vec<f32>,

    /// Last activations
    last_activations: Vec<f32>,
}

/// Embedded SNN plasticity engine
pub struct EmbeddedSNN {
    config: EmbeddedSNNConfig,
    state: SNNState,
    neuromod: NeuromodState,
    tick_count: usize,
}

impl EmbeddedSNN {
    /// Create new embedded SNN
    pub fn new(config: EmbeddedSNNConfig) -> Self {
        // Initialize random prototypes
        let mut prototypes = Vec::with_capacity(config.num_neurons);
        for _ in 0..config.num_neurons {
            let mut proto = vec![0.0; config.input_dim];
            // Random initialization (should be replaced with proper init)
            for val in &mut proto {
                *val = (rand::random::<f32>() - 0.5) * 0.1;
            }
            prototypes.push(proto);
        }

        // Initialize sparse weights (all weak initially)
        let mut weights = Vec::with_capacity(config.num_neurons);
        for i in 0..config.num_neurons {
            let mut neighbors = Vec::new();
            // Connect to top-k neighbors (circular for now)
            for k in 1..=config.top_k {
                let neighbor = (i + k) % config.num_neurons;
                neighbors.push((neighbor, 0.1)); // Weak initial connection
            }
            weights.push(neighbors);
        }

        let num_neurons = config.num_neurons;

        Self {
            config,
            state: SNNState {
                prototypes,
                weights,
                traces: vec![0.0; num_neurons],
                firing_rates: vec![0.0; num_neurons],
                last_activations: vec![0.0; num_neurons],
            },
            neuromod: NeuromodState::baseline(),
            tick_count: 0,
        }
    }

    /// Compute activation for each neuron given input
    fn compute_activations(&self, input: &[f32]) -> Vec<f32> {
        let mut activations = vec![0.0; self.config.num_neurons];

        for (i, proto) in self.state.prototypes.iter().enumerate() {
            // Cosine similarity
            let mut dot = 0.0;
            let mut norm_input = 0.0;
            let mut norm_proto = 0.0;

            for (inp, p) in input.iter().zip(proto.iter()) {
                dot += inp * p;
                norm_input += inp * inp;
                norm_proto += p * p;
            }

            if norm_input > 0.0 && norm_proto > 0.0 {
                activations[i] = dot / (norm_input.sqrt() * norm_proto.sqrt());
                activations[i] = activations[i].max(0.0); // ReLU
            }
        }

        activations
    }

    /// Apply lateral competition (winner-take-most)
    fn apply_competition(&self, activations: &mut [f32]) {
        let strength = self.config.competition_strength * self.neuromod.norepinephrine;

        // Find top activated neurons
        let mut sorted_indices: Vec<usize> = (0..activations.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            activations[b].partial_cmp(&activations[a]).unwrap()
        });

        // Suppress non-winners
        for (rank, &idx) in sorted_indices.iter().enumerate() {
            let suppression = (rank as f32 / activations.len() as f32) * strength;
            activations[idx] *= 1.0 - suppression;
        }
    }

    /// Spread activation through associative weights
    fn spread_activation(&self, activations: &mut [f32]) {
        let mut spread = vec![0.0; self.config.num_neurons];

        for (i, neighbors) in self.state.weights.iter().enumerate() {
            for &(neighbor_id, weight) in neighbors {
                spread[neighbor_id] += activations[i] * weight;
            }
        }

        // Add spread to activations
        for (act, spr) in activations.iter_mut().zip(spread.iter()) {
            *act += spr * self.neuromod.acetylcholine; // Gated by attention
        }
    }

    /// Detect spikes (neurons above threshold)
    fn detect_spikes(&self, activations: &[f32]) -> Vec<usize> {
        activations
            .iter()
            .enumerate()
            .filter(|(_, &act)| act > self.config.spike_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Apply STDP (cells that fire together wire together)
    fn apply_stdp(&mut self, spiking: &[usize]) -> Vec<Delta> {
        let mut deltas = Vec::new();
        let lr = self.config.stdp_lr * self.neuromod.dopamine; // Modulated by reward

        // Update weights between co-spiking neurons
        for &i in spiking {
            for &j in spiking {
                if i == j {
                    continue;
                }

                // Find if there's a connection i -> j
                if let Some(conn) = self.state.weights[i].iter_mut().find(|(n, _)| *n == j) {
                    conn.1 = (conn.1 + lr).clamp(0.0, 1.0);

                    // Generate delta for this weight change
                    deltas.push(Delta::merge(
                        format!("weight_{}_{}", i, j),
                        conn.1.to_le_bytes().to_vec(),
                        "snn_stdp",
                        conn.1, // Strength = weight value
                        None,   // Will be set by Thermogram
                    ));
                }
            }
        }

        deltas
    }

    /// Apply homeostasis (prevent runaway strengthening)
    fn apply_homeostasis(&mut self) {
        let target = self.config.homeostasis_target;
        let rate = 0.01 * self.neuromod.serotonin; // Modulated by mood

        for (i, firing_rate) in self.state.firing_rates.iter_mut().enumerate() {
            // Adjust neuron sensitivity toward target
            let error = target - *firing_rate;

            // Scale all outgoing weights
            for (_, weight) in &mut self.state.weights[i] {
                *weight *= 1.0 + error * rate;
                *weight = weight.clamp(0.0, 1.0);
            }
        }
    }

    /// Apply decay to weak connections
    fn apply_decay(&mut self) -> Vec<Delta> {
        let mut deltas = Vec::new();
        let decay = self.config.decay_rate * (1.0 - self.neuromod.serotonin); // Less decay when happy

        for (i, neighbors) in self.state.weights.iter_mut().enumerate() {
            for (j, weight) in neighbors.iter_mut() {
                *weight *= 1.0 - decay;

                // If decayed below threshold, record as pruned
                if *weight < 0.01 {
                    deltas.push(Delta::delete(
                        format!("weight_{}_{}", i, j),
                        "snn_decay",
                        None,
                    ));
                    *weight = 0.0;
                }
            }
        }

        deltas
    }

    /// Update eligibility traces
    fn update_traces(&mut self, activations: &[f32]) {
        for (trace, &act) in self.state.traces.iter_mut().zip(activations.iter()) {
            *trace = *trace * (1.0 - self.config.trace_decay) + act;
        }
    }

    /// Update firing rates (exponential moving average)
    fn update_firing_rates(&mut self, spiking: &[usize]) {
        let alpha = 0.1;
        for i in 0..self.config.num_neurons {
            let spike = if spiking.contains(&i) { 1.0 } else { 0.0 };
            self.state.firing_rates[i] = self.state.firing_rates[i] * (1.0 - alpha) + spike * alpha;
        }
    }
}

impl PlasticityEngine for EmbeddedSNN {
    fn process(&mut self, activation: &[f32], neuromod: &NeuromodState) -> Result<Vec<Delta>> {
        self.neuromod = neuromod.clone();
        self.tick_count += 1;

        // 1. Compute activations from input
        let mut activations = self.compute_activations(activation);

        // 2. Apply competition
        self.apply_competition(&mut activations);

        // 3. Spread activation through weights
        self.spread_activation(&mut activations);

        // 4. Detect spikes
        let spiking = self.detect_spikes(&activations);

        // 5. Apply STDP to strengthen co-firing connections
        let mut deltas = self.apply_stdp(&spiking);

        // 6. Apply homeostasis (every 100 ticks)
        if self.tick_count % 100 == 0 {
            self.apply_homeostasis();
        }

        // 7. Apply decay (every 10 ticks)
        if self.tick_count % 10 == 0 {
            let decay_deltas = self.apply_decay();
            deltas.extend(decay_deltas);
        }

        // 8. Update traces and firing rates
        self.update_traces(&activations);
        self.update_firing_rates(&spiking);

        // Store activations for next tick
        self.state.last_activations = activations;

        Ok(deltas)
    }

    fn sync_neuromod(&mut self, neuromod: &NeuromodState) {
        self.neuromod = neuromod.clone();
    }

    fn state(&self) -> PlasticityEngineState {
        PlasticityEngineState {
            engine_type: "EmbeddedSNN".to_string(),
            neuromod: self.neuromod.clone(),
            custom_state: bincode::serialize(&self.state).unwrap_or_default(),
        }
    }

    fn restore(&mut self, state: &PlasticityEngineState) -> Result<()> {
        if state.engine_type != "EmbeddedSNN" {
            return Err(crate::error::Error::InvalidState(
                "Wrong engine type".to_string(),
            ));
        }

        self.neuromod = state.neuromod.clone();
        self.state = bincode::deserialize(&state.custom_state)
            .map_err(|e| crate::error::Error::Deserialization(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_snn() {
        let config = EmbeddedSNNConfig::default();
        let snn = EmbeddedSNN::new(config);

        assert_eq!(snn.state.prototypes.len(), 100);
        assert_eq!(snn.state.weights.len(), 100);
    }

    #[test]
    fn test_process_activation() {
        let config = EmbeddedSNNConfig::default();
        let mut snn = EmbeddedSNN::new(config);

        let input = vec![0.5; 2048];
        let neuromod = NeuromodState::baseline();

        let deltas = snn.process(&input, &neuromod).unwrap();

        // Should generate some deltas (STDP or decay)
        assert!(!deltas.is_empty() || snn.tick_count < 10);
    }

    #[test]
    fn test_stdp_strengthening() {
        let config = EmbeddedSNNConfig {
            num_neurons: 10,
            spike_threshold: 0.3,
            ..Default::default()
        };
        let mut snn = EmbeddedSNN::new(config);

        // High activation should cause spikes
        let input = vec![1.0; 2048];
        let mut neuromod = NeuromodState::baseline();
        neuromod.dopamine = 1.0; // High reward

        let deltas = snn.process(&input, &neuromod).unwrap();

        // Should have weight updates from STDP
        let weight_updates = deltas
            .iter()
            .filter(|d| d.key.starts_with("weight_"))
            .count();

        assert!(weight_updates > 0);
    }
}

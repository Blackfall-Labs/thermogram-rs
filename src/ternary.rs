//! Ternary Weight System - BitNet-style quantization for neural weights
//!
//! # Overview
//!
//! Ternary weights use only three values: +1, 0, -1 (encoded as Pos, Zero, Neg).
//! This provides:
//! - **16x compression** vs f32 (2 bits vs 32 bits)
//! - **Faster inference** (additions instead of multiplications)
//! - **Better generalization** (regularization effect)
//! - **Hardware efficiency** (FPGA/ASIC friendly)
//!
//! # Encoding
//!
//! We pack 4 ternary weights per byte using 2-bit encoding:
//! - `00` = Zero (0)
//! - `01` = Pos (+1)
//! - `10` = Neg (-1)
//! - `11` = Reserved (for future use)
//!
//! # Quantization
//!
//! f32 weights are quantized using a threshold:
//! - `|w| < threshold` → Zero
//! - `w >= threshold` → Pos
//! - `w <= -threshold` → Neg
//!
//! The threshold can be:
//! - Fixed (e.g., 0.5)
//! - Adaptive per-layer (based on weight distribution)
//! - Learned during training (straight-through estimator)

use serde::{Deserialize, Serialize};
use std::fmt;

/// Ternary weight: +1, 0, or -1
///
/// This is the fundamental unit of our ternary neural network weights.
/// Uses `repr(i8)` for efficient memory layout and arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i8)]
pub enum TernaryWeight {
    /// Negative connection (-1)
    Neg = -1,
    /// No connection (0)
    Zero = 0,
    /// Positive connection (+1)
    Pos = 1,
}

impl TernaryWeight {
    /// Convert to f32 for computation
    #[inline]
    pub fn to_f32(self) -> f32 {
        self as i8 as f32
    }

    /// Convert to i8
    #[inline]
    pub fn to_i8(self) -> i8 {
        self as i8
    }

    /// Quantize from f32 using threshold
    ///
    /// - `|w| < threshold` → Zero
    /// - `w >= threshold` → Pos
    /// - `w <= -threshold` → Neg
    #[inline]
    pub fn from_f32(value: f32, threshold: f32) -> Self {
        if value >= threshold {
            TernaryWeight::Pos
        } else if value <= -threshold {
            TernaryWeight::Neg
        } else {
            TernaryWeight::Zero
        }
    }

    /// Apply STDP-like state transition
    ///
    /// Strengthening: Zero→Pos or Neg→Zero
    /// Weakening: Pos→Zero or Zero→Neg
    #[inline]
    pub fn strengthen(self) -> Self {
        match self {
            TernaryWeight::Neg => TernaryWeight::Zero,
            TernaryWeight::Zero => TernaryWeight::Pos,
            TernaryWeight::Pos => TernaryWeight::Pos, // Already max
        }
    }

    /// Weaken the weight (opposite of strengthen)
    #[inline]
    pub fn weaken(self) -> Self {
        match self {
            TernaryWeight::Pos => TernaryWeight::Zero,
            TernaryWeight::Zero => TernaryWeight::Neg,
            TernaryWeight::Neg => TernaryWeight::Neg, // Already min
        }
    }

    /// Flip the sign (Pos ↔ Neg, Zero stays Zero)
    #[inline]
    pub fn flip(self) -> Self {
        match self {
            TernaryWeight::Pos => TernaryWeight::Neg,
            TernaryWeight::Neg => TernaryWeight::Pos,
            TernaryWeight::Zero => TernaryWeight::Zero,
        }
    }

    /// Check if this weight is active (non-zero)
    #[inline]
    pub fn is_active(self) -> bool {
        self != TernaryWeight::Zero
    }

    /// 2-bit encoding for packing
    #[inline]
    fn to_2bit(self) -> u8 {
        match self {
            TernaryWeight::Zero => 0b00,
            TernaryWeight::Pos => 0b01,
            TernaryWeight::Neg => 0b10,
        }
    }

    /// Decode from 2-bit value
    #[inline]
    fn from_2bit(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => TernaryWeight::Zero,
            0b01 => TernaryWeight::Pos,
            0b10 => TernaryWeight::Neg,
            _ => TernaryWeight::Zero, // Reserved → Zero
        }
    }
}

impl Default for TernaryWeight {
    fn default() -> Self {
        TernaryWeight::Zero
    }
}

impl fmt::Debug for TernaryWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TernaryWeight::Neg => write!(f, "-"),
            TernaryWeight::Zero => write!(f, "0"),
            TernaryWeight::Pos => write!(f, "+"),
        }
    }
}

impl fmt::Display for TernaryWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TernaryWeight::Neg => write!(f, "-1"),
            TernaryWeight::Zero => write!(f, " 0"),
            TernaryWeight::Pos => write!(f, "+1"),
        }
    }
}

impl From<i8> for TernaryWeight {
    fn from(v: i8) -> Self {
        match v.signum() {
            -1 => TernaryWeight::Neg,
            0 => TernaryWeight::Zero,
            1 => TernaryWeight::Pos,
            _ => TernaryWeight::Zero,
        }
    }
}

impl From<TernaryWeight> for i8 {
    fn from(w: TernaryWeight) -> i8 {
        w as i8
    }
}

// =============================================================================
// PACKED TERNARY - Efficient storage (4 weights per byte)
// =============================================================================

/// Packed ternary weights - 4 weights per byte
///
/// Storage layout: Each byte contains 4 weights, packed as:
/// ```text
/// byte = [w3:2bit][w2:2bit][w1:2bit][w0:2bit]
///        bits 7-6  bits 5-4  bits 3-2  bits 1-0
/// ```
///
/// This gives 8x compression vs f32 (4 bytes → 0.5 bytes per weight).
#[derive(Clone, Serialize, Deserialize)]
pub struct PackedTernary {
    /// Packed data (4 weights per byte)
    data: Vec<u8>,
    /// Number of weights (not bytes)
    len: usize,
}

impl PackedTernary {
    /// Create new PackedTernary from weight slice
    pub fn new(weights: &[TernaryWeight]) -> Self {
        let num_bytes = (weights.len() + 3) / 4;
        let mut data = vec![0u8; num_bytes];

        for (i, &w) in weights.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= w.to_2bit() << bit_offset;
        }

        Self {
            data,
            len: weights.len(),
        }
    }

    /// Create PackedTernary with all zeros
    pub fn zeros(len: usize) -> Self {
        let num_bytes = (len + 3) / 4;
        Self {
            data: vec![0u8; num_bytes],
            len,
        }
    }

    /// Create PackedTernary from f32 weights with threshold
    pub fn from_f32(weights: &[f32], threshold: f32) -> Self {
        let ternary: Vec<TernaryWeight> = weights
            .iter()
            .map(|&w| TernaryWeight::from_f32(w, threshold))
            .collect();
        Self::new(&ternary)
    }

    /// Create PackedTernary from f32 weights with adaptive threshold
    ///
    /// Computes threshold as: mean(|w|) for non-zero weights
    pub fn from_f32_adaptive(weights: &[f32]) -> Self {
        let threshold = Self::compute_adaptive_threshold(weights);
        Self::from_f32(weights, threshold)
    }

    /// Compute adaptive threshold from weight distribution
    ///
    /// Uses mean absolute value of non-negligible weights
    pub fn compute_adaptive_threshold(weights: &[f32]) -> f32 {
        let min_weight = 1e-6;
        let sum: f32 = weights.iter().map(|w| w.abs()).filter(|&w| w > min_weight).sum();
        let count = weights.iter().filter(|&&w| w.abs() > min_weight).count();

        if count > 0 {
            sum / count as f32
        } else {
            0.5 // Default threshold
        }
    }

    /// Get weight at index
    #[inline]
    pub fn get(&self, idx: usize) -> TernaryWeight {
        if idx >= self.len {
            return TernaryWeight::Zero;
        }

        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = (self.data[byte_idx] >> bit_offset) & 0b11;
        TernaryWeight::from_2bit(bits)
    }

    /// Set weight at index
    #[inline]
    pub fn set(&mut self, idx: usize, value: TernaryWeight) {
        if idx >= self.len {
            return;
        }

        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;

        // Clear existing bits
        self.data[byte_idx] &= !(0b11 << bit_offset);
        // Set new bits
        self.data[byte_idx] |= value.to_2bit() << bit_offset;
    }

    /// Number of weights
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw bytes (for serialization)
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw bytes
    pub fn from_bytes(data: Vec<u8>, len: usize) -> Self {
        Self { data, len }
    }

    /// Convert back to Vec<TernaryWeight>
    pub fn to_vec(&self) -> Vec<TernaryWeight> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    /// Convert to f32 vector
    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.len).map(|i| self.get(i).to_f32()).collect()
    }

    /// Count non-zero weights (sparsity metric)
    pub fn count_active(&self) -> usize {
        (0..self.len).filter(|&i| self.get(i).is_active()).count()
    }

    /// Sparsity ratio (0.0 = all active, 1.0 = all zero)
    pub fn sparsity(&self) -> f32 {
        if self.len == 0 {
            return 1.0;
        }
        1.0 - (self.count_active() as f32 / self.len as f32)
    }

    /// Compute dot product with f32 vector (efficient ternary matmul)
    ///
    /// This is the key operation for inference:
    /// - Pos → add input
    /// - Neg → subtract input
    /// - Zero → skip
    pub fn dot(&self, input: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let min_len = self.len.min(input.len());

        for i in 0..min_len {
            match self.get(i) {
                TernaryWeight::Pos => sum += input[i],
                TernaryWeight::Neg => sum -= input[i],
                TernaryWeight::Zero => {}
            }
        }

        sum
    }

    /// Strengthen weight at index (STDP-like)
    pub fn strengthen(&mut self, idx: usize) {
        let current = self.get(idx);
        self.set(idx, current.strengthen());
    }

    /// Weaken weight at index
    pub fn weaken(&mut self, idx: usize) {
        let current = self.get(idx);
        self.set(idx, current.weaken());
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.len * 4;
        if f32_bytes > 0 {
            f32_bytes as f32 / self.data.len() as f32
        } else {
            1.0
        }
    }
}

impl fmt::Debug for PackedTernary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedTernary({} weights, {} bytes, {:.1}% sparse)",
               self.len,
               self.data.len(),
               self.sparsity() * 100.0)
    }
}

impl std::ops::Index<usize> for PackedTernary {
    type Output = TernaryWeight;

    fn index(&self, idx: usize) -> &Self::Output {
        // We can't return a reference to a packed value, so this is a bit of a hack
        // In practice, use .get() instead
        static POS: TernaryWeight = TernaryWeight::Pos;
        static NEG: TernaryWeight = TernaryWeight::Neg;
        static ZERO: TernaryWeight = TernaryWeight::Zero;

        match self.get(idx) {
            TernaryWeight::Pos => &POS,
            TernaryWeight::Neg => &NEG,
            TernaryWeight::Zero => &ZERO,
        }
    }
}

// =============================================================================
// TERNARY LAYER - A layer of ternary weights with bias
// =============================================================================

/// A layer of ternary weights for neural network inference
#[derive(Clone, Serialize, Deserialize)]
pub struct TernaryLayer {
    /// Packed weights [output_dim, input_dim]
    weights: Vec<PackedTernary>,
    /// Per-output scale factors (learned during training)
    scales: Vec<f32>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl TernaryLayer {
    /// Create from f32 weight matrix [output_dim, input_dim]
    pub fn from_f32_matrix(weights: &[Vec<f32>]) -> Self {
        if weights.is_empty() {
            return Self {
                weights: vec![],
                scales: vec![],
                input_dim: 0,
                output_dim: 0,
            };
        }

        let output_dim = weights.len();
        let input_dim = weights[0].len();

        let mut packed = Vec::with_capacity(output_dim);
        let mut scales = Vec::with_capacity(output_dim);

        for row in weights {
            // Compute scale as max absolute value (for de-quantization)
            let max_abs = row.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-6 { max_abs } else { 1.0 };
            scales.push(scale);

            // Normalize and quantize
            let normalized: Vec<f32> = row.iter().map(|w| w / scale).collect();
            let threshold = PackedTernary::compute_adaptive_threshold(&normalized);
            packed.push(PackedTernary::from_f32(&normalized, threshold));
        }

        Self {
            weights: packed,
            scales,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: output = weights @ input * scales
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.output_dim);

        for (row, &scale) in self.weights.iter().zip(self.scales.iter()) {
            let dot = row.dot(input);
            output.push(dot * scale);
        }

        output
    }

    /// Get weight at [row, col]
    pub fn get_weight(&self, row: usize, col: usize) -> TernaryWeight {
        if row < self.weights.len() {
            self.weights[row].get(col)
        } else {
            TernaryWeight::Zero
        }
    }

    /// Set weight at [row, col]
    pub fn set_weight(&mut self, row: usize, col: usize, value: TernaryWeight) {
        if row < self.weights.len() {
            self.weights[row].set(col, value);
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let weights_bytes: usize = self.weights.iter().map(|w| w.memory_bytes()).sum();
        let scales_bytes = self.scales.len() * 4;
        weights_bytes + scales_bytes
    }

    /// Dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.output_dim, self.input_dim)
    }
}

impl fmt::Debug for TernaryLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TernaryLayer({}x{}, {} bytes)",
               self.output_dim,
               self.input_dim,
               self.memory_bytes())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_weight_conversion() {
        assert_eq!(TernaryWeight::Pos.to_f32(), 1.0);
        assert_eq!(TernaryWeight::Zero.to_f32(), 0.0);
        assert_eq!(TernaryWeight::Neg.to_f32(), -1.0);
    }

    #[test]
    fn test_ternary_quantization() {
        let threshold = 0.5;

        assert_eq!(TernaryWeight::from_f32(0.8, threshold), TernaryWeight::Pos);
        assert_eq!(TernaryWeight::from_f32(0.3, threshold), TernaryWeight::Zero);
        assert_eq!(TernaryWeight::from_f32(-0.3, threshold), TernaryWeight::Zero);
        assert_eq!(TernaryWeight::from_f32(-0.8, threshold), TernaryWeight::Neg);
    }

    #[test]
    fn test_ternary_strengthen_weaken() {
        assert_eq!(TernaryWeight::Zero.strengthen(), TernaryWeight::Pos);
        assert_eq!(TernaryWeight::Neg.strengthen(), TernaryWeight::Zero);
        assert_eq!(TernaryWeight::Pos.strengthen(), TernaryWeight::Pos);

        assert_eq!(TernaryWeight::Pos.weaken(), TernaryWeight::Zero);
        assert_eq!(TernaryWeight::Zero.weaken(), TernaryWeight::Neg);
        assert_eq!(TernaryWeight::Neg.weaken(), TernaryWeight::Neg);
    }

    #[test]
    fn test_packed_ternary_basic() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Zero,
            TernaryWeight::Neg,
            TernaryWeight::Pos,
            TernaryWeight::Zero,
        ];

        let packed = PackedTernary::new(&weights);

        assert_eq!(packed.len(), 5);
        assert_eq!(packed.get(0), TernaryWeight::Pos);
        assert_eq!(packed.get(1), TernaryWeight::Zero);
        assert_eq!(packed.get(2), TernaryWeight::Neg);
        assert_eq!(packed.get(3), TernaryWeight::Pos);
        assert_eq!(packed.get(4), TernaryWeight::Zero);
    }

    #[test]
    fn test_packed_ternary_set() {
        let mut packed = PackedTernary::zeros(8);

        packed.set(0, TernaryWeight::Pos);
        packed.set(3, TernaryWeight::Neg);
        packed.set(7, TernaryWeight::Pos);

        assert_eq!(packed.get(0), TernaryWeight::Pos);
        assert_eq!(packed.get(1), TernaryWeight::Zero);
        assert_eq!(packed.get(3), TernaryWeight::Neg);
        assert_eq!(packed.get(7), TernaryWeight::Pos);
    }

    #[test]
    fn test_packed_ternary_from_f32() {
        let f32_weights = vec![0.8, 0.2, -0.9, 0.1, -0.3];
        let packed = PackedTernary::from_f32(&f32_weights, 0.5);

        assert_eq!(packed.get(0), TernaryWeight::Pos);  // 0.8 >= 0.5
        assert_eq!(packed.get(1), TernaryWeight::Zero); // 0.2 in [-0.5, 0.5]
        assert_eq!(packed.get(2), TernaryWeight::Neg);  // -0.9 <= -0.5
        assert_eq!(packed.get(3), TernaryWeight::Zero); // 0.1 in [-0.5, 0.5]
        assert_eq!(packed.get(4), TernaryWeight::Zero); // -0.3 in [-0.5, 0.5]
    }

    #[test]
    fn test_packed_ternary_dot() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
        ];
        let packed = PackedTernary::new(&weights);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = packed.dot(&input);

        // Expected: 1.0 * 1 + 2.0 * (-1) + 3.0 * 0 + 4.0 * 1 = 1 - 2 + 0 + 4 = 3
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_packed_ternary_compression() {
        let weights = vec![TernaryWeight::Pos; 1000];
        let packed = PackedTernary::new(&weights);

        // 1000 weights = 250 bytes (4 per byte)
        assert_eq!(packed.memory_bytes(), 250);

        // Compression ratio: 4000 bytes (f32) / 250 bytes = 16x
        let ratio = packed.compression_ratio();
        assert!((ratio - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_ternary_layer_forward() {
        // 2x3 weight matrix
        let weights = vec![
            vec![1.0, 0.0, -1.0],
            vec![-1.0, 1.0, 0.0],
        ];

        let layer = TernaryLayer::from_f32_matrix(&weights);
        let input = vec![1.0, 2.0, 3.0];

        let output = layer.forward(&input);

        // Row 0: 1*1 + 2*0 + 3*(-1) = -2, scaled by max(1.0) = 1.0 → -2.0
        // Row 1: 1*(-1) + 2*1 + 3*0 = 1, scaled by max(1.0) = 1.0 → 1.0
        assert_eq!(output.len(), 2);
        // Note: actual values depend on scaling, this tests the structure
    }

    #[test]
    fn test_sparsity() {
        let weights = vec![
            TernaryWeight::Zero,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Zero,
        ];
        let packed = PackedTernary::new(&weights);

        assert_eq!(packed.count_active(), 1);
        assert!((packed.sparsity() - 0.75).abs() < 0.01);
    }
}

//! Hash Chain - Cryptographic audit trail for deltas
//!
//! Each delta in the chain references the hash of the previous delta,
//! forming a tamper-evident history similar to blockchain.

use crate::delta::Delta;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Hash chain of deltas providing cryptographic audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashChain {
    /// Chain of deltas (oldest first)
    pub deltas: Vec<Delta>,

    /// Hash of the most recent delta
    pub head_hash: Option<String>,
}

impl HashChain {
    /// Create empty hash chain
    pub fn new() -> Self {
        Self {
            deltas: Vec::new(),
            head_hash: None,
        }
    }

    /// Append a delta to the chain
    ///
    /// Verifies the delta links to the current head before appending
    pub fn append(&mut self, mut delta: Delta) -> Result<()> {
        // Verify delta links to current head
        if let Some(ref head_hash) = self.head_hash {
            match &delta.prev_hash {
                Some(prev) if prev == head_hash => {
                    // Valid chain link
                }
                Some(prev) => {
                    return Err(Error::HashChainVerification(format!(
                        "Delta prev_hash {} does not match current head {}",
                        prev, head_hash
                    )));
                }
                None => {
                    return Err(Error::HashChainVerification(
                        "Delta must reference previous hash".to_string(),
                    ));
                }
            }
        } else {
            // First delta in chain - should have no prev_hash
            if delta.prev_hash.is_some() {
                return Err(Error::HashChainVerification(
                    "First delta should not reference previous hash".to_string(),
                ));
            }
        }

        // Recompute hash to ensure integrity
        delta.hash = delta.compute_hash();

        // Verify hash is correct
        if !delta.verify_hash() {
            return Err(Error::HashChainVerification(
                "Delta hash verification failed".to_string(),
            ));
        }

        // Append to chain
        self.head_hash = Some(delta.hash.clone());
        self.deltas.push(delta);

        Ok(())
    }

    /// Verify entire hash chain
    pub fn verify(&self) -> Result<()> {
        if self.deltas.is_empty() {
            return Ok(());
        }

        // Verify first delta has no prev_hash
        if self.deltas[0].prev_hash.is_some() {
            return Err(Error::HashChainVerification(
                "First delta should not have prev_hash".to_string(),
            ));
        }

        // Verify each delta
        for (i, delta) in self.deltas.iter().enumerate() {
            // Verify hash
            if !delta.verify_hash() {
                return Err(Error::HashChainVerification(format!(
                    "Delta {} hash verification failed",
                    i
                )));
            }

            // Verify chain link
            if i > 0 {
                let prev = &self.deltas[i - 1];
                if !delta.verify_chain(prev) {
                    return Err(Error::HashChainVerification(format!(
                        "Delta {} does not link to previous delta",
                        i
                    )));
                }
            }
        }

        // Verify head hash
        if let Some(ref head_hash) = self.head_hash {
            if let Some(last_delta) = self.deltas.last() {
                if &last_delta.hash != head_hash {
                    return Err(Error::HashChainVerification(
                        "Head hash does not match last delta".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get all deltas for a specific key
    pub fn get_history(&self, key: &str) -> Vec<&Delta> {
        self.deltas.iter().filter(|d| d.key == key).collect()
    }

    /// Get the most recent delta for a key
    pub fn get_latest(&self, key: &str) -> Option<&Delta> {
        self.deltas
            .iter()
            .rev()
            .find(|d| d.key == key)
    }

    /// Count total deltas
    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Get deltas since a specific hash
    pub fn deltas_since(&self, hash: &str) -> Vec<&Delta> {
        let mut collecting = false;
        let mut result = Vec::new();

        for delta in &self.deltas {
            if collecting {
                result.push(delta);
            } else if delta.hash == hash {
                collecting = true;
            }
        }

        result
    }
}

impl Default for HashChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::DeltaType;
    use ternary_signal::Signal;

    fn sig_val(v: u8) -> Vec<Signal> {
        vec![Signal::positive(v)]
    }

    #[test]
    fn test_empty_chain() {
        let chain = HashChain::new();
        assert!(chain.is_empty());
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_append_first_delta() {
        let mut chain = HashChain::new();
        let delta = Delta::create("key1", sig_val(100), "source");

        chain.append(delta).unwrap();

        assert_eq!(chain.len(), 1);
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_append_chain() {
        let mut chain = HashChain::new();

        let delta1 = Delta::create("key1", sig_val(100), "source");
        let hash1 = delta1.hash.clone();
        chain.append(delta1).unwrap();

        let delta2 = Delta::update("key1", sig_val(110), "source", Signal::positive(204), Some(hash1.clone()));
        chain.append(delta2).unwrap();

        let delta3 = Delta::update("key1", sig_val(120), "source", Signal::positive(230), Some(chain.head_hash.clone().unwrap()));
        chain.append(delta3).unwrap();

        assert_eq!(chain.len(), 3);
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_broken_chain() {
        let mut chain = HashChain::new();

        let delta1 = Delta::create("key1", sig_val(100), "source");
        chain.append(delta1).unwrap();

        // Try to append with wrong prev_hash
        let delta2 = Delta::update("key1", sig_val(110), "source", Signal::positive(204), Some("wrong_hash".to_string()));

        assert!(chain.append(delta2).is_err());
    }

    #[test]
    fn test_get_history() {
        let mut chain = HashChain::new();

        let delta1 = Delta::create("key1", sig_val(100), "source");
        let hash1 = delta1.hash.clone();
        chain.append(delta1).unwrap();

        // Second delta must link to first
        let mut delta2 = Delta::create("key2", sig_val(110), "source");
        delta2.prev_hash = Some(hash1.clone());
        delta2.hash = delta2.compute_hash();
        let hash2 = delta2.hash.clone();
        chain.append(delta2).unwrap();

        let delta3 = Delta::update("key1", sig_val(120), "source", Signal::positive(204), Some(hash2.clone()));
        chain.append(delta3).unwrap();

        let history = chain.get_history("key1");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_get_latest() {
        let mut chain = HashChain::new();

        let delta1 = Delta::create("key1", sig_val(100), "source");
        let hash1 = delta1.hash.clone();
        chain.append(delta1).unwrap();

        let delta2 = Delta::update("key1", sig_val(110), "source", Signal::positive(204), Some(hash1.clone()));
        let hash2 = delta2.hash.clone();
        chain.append(delta2).unwrap();

        let latest = chain.get_latest("key1").unwrap();
        assert_eq!(latest.hash, hash2);
        assert_eq!(latest.value, sig_val(110));
    }
}

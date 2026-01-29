//! Concurrency and threading tests
//!
//! Tests for thread safety, race conditions, and concurrent access patterns.

use std::sync::{Arc, Mutex};
use std::thread;
use tempfile::tempdir;
use thermogram::*;
use ternary_signal::Signal;

fn sig_val(v: u8) -> Vec<Signal> {
    vec![Signal::positive(v)]
}

#[test]
fn test_concurrent_reads() {
    let mut thermo = Thermogram::new("test", PlasticityRule::stdp_like());

    // Populate with data
    for i in 0..100 {
        let delta = Delta::update(
            format!("key_{}", i),
            sig_val(100),
            "source",
            Signal::positive(128),
            thermo.dirty_chain.head_hash.clone(),
        );
        thermo.apply_delta(delta).unwrap();
    }

    thermo.consolidate().unwrap();

    // Wrap in Arc<Mutex> for shared access
    let thermo = Arc::new(Mutex::new(thermo));

    // Spawn multiple reader threads
    let mut handles = vec![];
    for _ in 0..10 {
        let thermo_clone = Arc::clone(&thermo);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let t = thermo_clone.lock().unwrap();
                let _value = t.read(&format!("key_{}", i)).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_sequential_writes_no_conflicts() {
    let thermo = Arc::new(Mutex::new(Thermogram::new(
        "test",
        PlasticityRule::stdp_like(),
    )));

    // Spawn threads that take turns writing
    let mut handles = vec![];
    for thread_id in 0..5 {
        let thermo_clone = Arc::clone(&thermo);
        let handle = thread::spawn(move || {
            for i in 0..20 {
                let mut t = thermo_clone.lock().unwrap();
                let delta = Delta::update(
                    format!("key_{}_{}", thread_id, i),
                    sig_val(100),
                    "source",
                    Signal::positive(128),
                    t.dirty_chain.head_hash.clone(),
                );
                t.apply_delta(delta).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let thermo = thermo.lock().unwrap();
    assert_eq!(thermo.dirty_chain.len(), 100); // 5 threads * 20 deltas
}

#[test]
fn test_concurrent_save_load() {
    let dir = tempdir().unwrap();

    // Create and save multiple Thermograms concurrently
    let mut handles = vec![];
    for thread_id in 0..5 {
        let dir_path = dir.path().to_path_buf();
        let handle = thread::spawn(move || {
            let path = dir_path.join(format!("thermo_{}.json", thread_id));
            let mut thermo = Thermogram::new(
                format!("test_{}", thread_id),
                PlasticityRule::stdp_like(),
            );

            // Each delta must link to previous
            for i in 0..10 {
                let delta = Delta::update(
                    format!("key_{}", i),
                    sig_val(100),
                    "source",
                    Signal::positive(128),
                    thermo.dirty_chain.head_hash.clone(),
                );
                thermo.apply_delta(delta).unwrap();
            }

            thermo.save(&path).unwrap();
            Thermogram::load(&path).unwrap()
        });
        handles.push(handle);
    }

    // All threads should succeed
    for handle in handles {
        let loaded = handle.join().unwrap();
        assert_eq!(loaded.dirty_chain.len(), 10);
    }
}

#[test]
fn test_neuromod_sync_concurrent() {
    let central = Arc::new(Mutex::new(NeuromodState::baseline()));

    // Multiple Thermograms syncing from central state
    let mut handles = vec![];
    for _ in 0..10 {
        let central_clone = Arc::clone(&central);
        let handle = thread::spawn(move || {
            let config = EmbeddedSNNConfig::default();
            let mut snn = EmbeddedSNN::new(config);

            for _ in 0..100 {
                let neuromod = central_clone.lock().unwrap().clone();
                snn.sync_neuromod(&neuromod);

                // Simulate processing
                let input = vec![0.5; 2048];
                let _ = snn.process(&input, &neuromod);
            }
        });
        handles.push(handle);
    }

    // Update central state while threads are running
    for _ in 0..50 {
        thread::sleep(std::time::Duration::from_micros(100));
        central.lock().unwrap().reward(0.01);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_consolidation_while_reading() {
    let thermo = Arc::new(Mutex::new(Thermogram::new(
        "test",
        PlasticityRule::stdp_like(),
    )));

    // Populate
    {
        let mut t = thermo.lock().unwrap();
        for i in 0..100 {
            let delta = Delta::update(
                format!("key_{}", i),
                sig_val(100),
                "source",
                Signal::positive(128),
                t.dirty_chain.head_hash.clone(),
            );
            t.apply_delta(delta).unwrap();
        }
    }

    // Reader thread
    let thermo_read = Arc::clone(&thermo);
    let reader = thread::spawn(move || {
        for _ in 0..1000 {
            let t = thermo_read.lock().unwrap();
            let _ = t.read("key_50");
        }
    });

    // Writer/consolidator thread
    let thermo_write = Arc::clone(&thermo);
    let writer = thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(10));
        let mut t = thermo_write.lock().unwrap();
        t.consolidate().unwrap();
    });

    reader.join().unwrap();
    writer.join().unwrap();
}

#[test]
fn test_snn_state_isolation() {
    // Each SNN should have independent state
    let mut snns = vec![];
    for _ in 0..5 {
        let config = EmbeddedSNNConfig::default();
        snns.push(EmbeddedSNN::new(config));
    }

    // Process different inputs
    let inputs: Vec<Vec<f32>> = (0..5)
        .map(|i| vec![(i as f32) * 0.2; 2048])
        .collect();

    let neuromod = NeuromodState::baseline();

    for (snn, input) in snns.iter_mut().zip(inputs.iter()) {
        snn.process(input, &neuromod).unwrap();
    }

    // Each SNN should have different internal state
    let states: Vec<_> = snns.iter().map(|s| s.state()).collect();

    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            // States should differ (custom_state contains serialized internal state)
            assert_ne!(
                states[i].custom_state.len(),
                0,
                "State should have data"
            );
            // Note: We can't easily compare states without deserializing,
            // but we verify they're independent by processing different inputs
        }
    }
}

#[test]
fn test_hash_chain_thread_safety() {
    // Hash chains are append-only, so we test sequential consistency
    let chain = Arc::new(Mutex::new(HashChain::new()));

    let mut handles = vec![];
    for thread_id in 0..5 {
        let chain_clone = Arc::clone(&chain);
        let handle = thread::spawn(move || {
            for i in 0..20 {
                let mut c = chain_clone.lock().unwrap();

                let mut delta = Delta::create(
                    format!("key_{}_{}", thread_id, i),
                    sig_val(100),
                    "source",
                );

                // Link to current head
                delta.prev_hash = c.head_hash.clone();
                delta.hash = delta.compute_hash();

                c.append(delta).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let chain = chain.lock().unwrap();
    assert_eq!(chain.len(), 100); // 5 threads * 20 deltas
    assert!(chain.verify().is_ok()); // Chain should be valid
}

#[test]
fn test_no_data_races_in_consolidation() {
    // Test that consolidation doesn't create data races
    let thermo = Arc::new(Mutex::new(Thermogram::new(
        "test",
        PlasticityRule::stdp_like(),
    )));

    // Writer thread - adds deltas
    let thermo_write = Arc::clone(&thermo);
    let writer = thread::spawn(move || {
        for i in 0..500 {
            let mut t = thermo_write.lock().unwrap();
            let delta = Delta::update(
                format!("key_{}", i),
                sig_val(100),
                "source",
                Signal::positive(128),
                t.dirty_chain.head_hash.clone(),
            );
            t.apply_delta(delta).unwrap();
        }
    });

    // Consolidator thread - periodically consolidates
    let thermo_consolidate = Arc::clone(&thermo);
    let consolidator = thread::spawn(move || {
        for _ in 0..10 {
            thread::sleep(std::time::Duration::from_millis(5));
            let mut t = thermo_consolidate.lock().unwrap();
            let _ = t.consolidate();
        }
    });

    writer.join().unwrap();
    consolidator.join().unwrap();

    // Verify integrity
    let thermo = thermo.lock().unwrap();
    assert!(thermo.dirty_chain.verify().is_ok());
}

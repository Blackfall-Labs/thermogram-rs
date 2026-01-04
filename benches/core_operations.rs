use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use thermogram::*;

fn bench_delta_creation(c: &mut Criterion) {
    c.bench_function("delta_create", |b| {
        b.iter(|| {
            Delta::create(
                black_box("test_key"),
                black_box(b"test_value".to_vec()),
                black_box("source"),
            )
        })
    });
}

fn bench_delta_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_append");

    for chain_size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(chain_size),
            chain_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut thermo = Thermogram::new("bench", PlasticityRule::stdp_like());
                        for i in 0..size {
                            let delta = Delta::update(
                                format!("key_{}", i),
                                b"value".to_vec(),
                                "source",
                                0.5,
                                thermo.dirty_chain.head_hash.clone(),
                            );
                            thermo.apply_delta(delta).unwrap();
                        }
                        thermo
                    },
                    |mut thermo| {
                        let delta = Delta::update(
                            "new_key",
                            b"new_value".to_vec(),
                            "source",
                            0.5,
                            thermo.dirty_chain.head_hash.clone(),
                        );
                        thermo.apply_delta(delta).unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_hash_verification(c: &mut Criterion) {
    let delta = Delta::create("key", b"value".to_vec(), "source");

    c.bench_function("hash_verify", |b| {
        b.iter(|| {
            black_box(delta.verify_hash())
        })
    });
}

fn bench_consolidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("consolidation");

    for dirty_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dirty_size),
            dirty_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut thermo = Thermogram::new("bench", PlasticityRule::stdp_like());
                        for i in 0..size {
                            let delta = Delta::update(
                                format!("key_{}", i % 100), // Some overlap
                                b"value".to_vec(),
                                "source",
                                0.5,
                                thermo.dirty_chain.head_hash.clone(),
                            );
                            thermo.apply_delta(delta).unwrap();
                        }
                        thermo
                    },
                    |mut thermo| {
                        thermo.consolidate().unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_snn_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("snn_tick");

    for num_neurons in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_neurons),
            num_neurons,
            |b, &neurons| {
                let config = EmbeddedSNNConfig {
                    num_neurons: neurons,
                    ..Default::default()
                };
                let mut snn = EmbeddedSNN::new(config);
                let input = vec![0.5; 2048];
                let neuromod = NeuromodState::baseline();

                b.iter(|| {
                    snn.process(&input, &neuromod).unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_read_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_operations");

    // Setup Thermogram with various sizes
    for entries in [10, 100, 1000].iter() {
        let mut thermo = Thermogram::new("bench", PlasticityRule::stdp_like());
        for i in 0..*entries {
            let delta = Delta::update(
                format!("key_{}", i),
                b"value".to_vec(),
                "source",
                0.5,
                thermo.dirty_chain.head_hash.clone(),
            );
            thermo.apply_delta(delta).unwrap();
        }
        thermo.consolidate().unwrap();

        group.bench_with_input(
            BenchmarkId::new("read_clean", entries),
            entries,
            |b, _| {
                b.iter(|| {
                    thermo.read(black_box("key_50")).unwrap();
                })
            },
        );

        // Add some dirty state
        for i in 0..10 {
            let delta = Delta::update(
                format!("key_{}", i),
                b"updated".to_vec(),
                "source",
                0.8,
                thermo.dirty_chain.head_hash.clone(),
            );
            thermo.apply_delta(delta).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("read_dirty", entries),
            entries,
            |b, _| {
                b.iter(|| {
                    thermo.read(black_box("key_5")).unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_plasticity_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("plasticity_rules");

    for rule_type in ["stdp", "ema", "bayesian"].iter() {
        let rule = match *rule_type {
            "stdp" => PlasticityRule::stdp_like(),
            "ema" => PlasticityRule::conservative(),
            "bayesian" => PlasticityRule::bayesian(),
            _ => unreachable!(),
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(rule_type),
            &rule,
            |b, rule| {
                b.iter(|| {
                    rule.apply_update(
                        black_box(0.5),
                        black_box(0.8),
                        black_box(100.0),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_neuromod_sync(c: &mut Criterion) {
    let config = NeuromodSyncConfig::default();
    let mut local = NeuromodState::baseline();
    let mut external = NeuromodState::baseline();
    external.dopamine = 1.0;

    c.bench_function("neuromod_sync", |b| {
        b.iter(|| {
            config.apply_sync(&mut local, &external);
        })
    });
}

criterion_group!(
    benches,
    bench_delta_creation,
    bench_delta_append,
    bench_hash_verification,
    bench_consolidation,
    bench_snn_tick,
    bench_read_operations,
    bench_plasticity_rules,
    bench_neuromod_sync,
);
criterion_main!(benches);

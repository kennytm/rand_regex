use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use rand::distributions::{Distribution, Standard, Uniform};
use rand::{Rng, SeedableRng};
use rand_distr::Alphanumeric;
use rand_regex::Regex;
use rand_xorshift::XorShiftRng;

fn alphanumeric_baseline(b: &mut Bencher<'_>) {
    let mut rng = XorShiftRng::seed_from_u64(0);
    let count_distr = Uniform::new_inclusive(10, 20);
    b.iter(|| {
        let count = count_distr.sample(&mut rng);
        Alphanumeric
            .sample_iter(&mut rng)
            .take(count)
            .collect::<Vec<u8>>()
    });
}

fn alphanumeric_rand_regex(b: &mut Bencher<'_>) {
    let regex = Regex::compile("[0-9a-zA-Z]{10,20}", 100).unwrap();
    let mut rng = XorShiftRng::seed_from_u64(0);
    b.iter(|| -> Vec<u8> { rng.sample(&regex) })
}

fn all_char_baseline(b: &mut Bencher<'_>) {
    let mut rng = XorShiftRng::seed_from_u64(0);
    b.iter(|| {
        Distribution::<char>::sample_iter(Standard, &mut rng)
            .take(10)
            .collect::<String>()
    });
}

fn all_char_rand_regex(b: &mut Bencher<'_>) {
    let regex = Regex::compile("(?s:.{10})", 100).unwrap();
    let mut rng = XorShiftRng::seed_from_u64(0);
    b.iter(|| -> Vec<u8> { rng.sample(&regex) })
}

fn run_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("alphanumeric");
    group.bench_function("baseline", alphanumeric_baseline);
    group.bench_function("rand_regex", alphanumeric_rand_regex);
    group.finish();

    let mut group = c.benchmark_group("all_char");
    group.bench_function("baseline", all_char_baseline);
    group.bench_function("rand_regex", all_char_rand_regex);
    group.finish();
}

criterion_group!(benches, run_benchmark);
criterion_main!(benches);

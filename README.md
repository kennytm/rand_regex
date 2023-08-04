`rand_regex`
============

[![Crates.io](https://img.shields.io/crates/v/rand_regex.svg)](https://crates.io/crates/rand_regex)
[![Build status](https://github.com/kennytm/rand_regex/workflows/Rust/badge.svg)](https://github.com/kennytm/rand_regex/actions?query=workflow%3ARust)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.txt)

Generates random strings and byte strings matching a regex.

Examples
--------

```rust
use rand::{SeedableRng, Rng};

let mut rng = rand_xorshift::XorShiftRng::from_seed(*b"The initial seed");

// creates a generator for sampling strings
let gen = rand_regex::Regex::compile(r"\d{4}-\d{2}-\d{2}", 100).unwrap();

// sample a few strings randomly
let samples = (&mut rng).sample_iter(&gen).take(3).collect::<Vec<String>>();

// all Unicode characters are included when sampling
assert_eq!(samples, vec![
    "꘥᥉১᪕-꧷៩-୦۱".to_string(),
    "𞋴۰𑋸꣕-᥆꧰-෮᪑".to_string(),
    "𑋲𐒥४౫-9႙-९౨".to_string()
]);

// you could use `regex_syntax::Hir` to include more options
let mut parser = regex_syntax::ParserBuilder::new().unicode(false).build();
let hir = parser.parse(r"\d{4}-\d{2}-\d{2}").unwrap();
let gen = rand_regex::Regex::with_hir(hir, 100).unwrap();
let samples = (&mut rng).sample_iter(&gen).take(3).collect::<Vec<String>>();
assert_eq!(samples, vec![
    "8922-87-63".to_string(),
    "3149-18-88".to_string(),
    "5420-58-55".to_string(),
]);
```

Acknowledgement
---------------

`rand_regex` is heavily inspired by [`regex_generate`] and [`proptest`].

[`regex_generate`]: https://crates.io/crates/regex_generate
[`proptest`]: https://crates.io/crates/proptest

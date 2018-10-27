`rand_regex`
============

[![Crates.io](https://img.shields.io/crates/v/rand_regex.svg)](https://crates.io/crates/rand_regex)
[![Build Status](https://travis-ci.com/kennytm/rand_regex.svg?branch=master)](https://travis-ci.com/kennytm/rand_regex)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.txt)

Generates random strings and byte strings matching a regex.

Examples
--------

```rust
extern crate rand_regex;
extern crate regex_syntax;
extern crate rand;
use rand::{SeedableRng, Rng};

let mut rng = rand::prng::XorShiftRng::from_seed(*b"The initial seed");

// creates a generator for sampling strings
let gen = rand_regex::Regex::compile(r"\d{4}-\d{2}-\d{2}", 100).unwrap();

// sample a few strings randomly
let samples = rng.sample_iter(&gen).take(3).collect::<Vec<String>>();

// all Unicode characters are included when sampling
assert_eq!(samples, vec![
    "១५᥏꣒-૨٩-٨𑁩".to_string(),
    "႘꘦൩᥌-꧶߉-8၀".to_string(),
    "௨൫𑣦០-𖭔༡-෩༤".to_string(),
]);

// you could use `regex_syntax::Hir` to include more options
let mut parser = regex_syntax::ParserBuilder::new().unicode(false).build();
let hir = parser.parse(r"\d{4}-\d{2}-\d{2}").unwrap();
let gen = rand_regex::Regex::with_hir(hir, 100).unwrap();
let samples = rng.sample_iter(&gen).take(3).collect::<Vec<String>>();
assert_eq!(samples, vec![
    "5786-30-81".to_string(),
    "4990-38-85".to_string(),
    "4514-20-35".to_string(),
]);
```

Acknowledgement
---------------

`rand_regex` is heavily inspired by [`regex_generate`] and [`proptest`].

[rand::Distribution]: https://docs.rs/rand/*/rand/distributions/trait.Distribution.html
[`regex_generate`]: https://crates.io/crates/regex_generate
[`proptest`]: https://crates.io/crates/proptest

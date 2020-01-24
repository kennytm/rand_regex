Changelog
=========

v0.14.0 (2020 Jan 25)
---------------------

Increased minimal Rust version.

Depends on `rand 0.7` and `regex_syntax 0.6`. The minimal Rust version is 1.40.

Improved generation algorithm. Sampling performance of regex involving lots of
character classes are greatly improved.

| Regex                | v0.13.1 | v0.14.0 |
|:---------------------|--------:|--------:|
| `[0-9a-zA-Z]{10,20}` | 268 ns  | 150 ns  |
| `(?s:.{10})`         | 209 ns  | 190 ns  |

v0.13.1 (2019 Dec 22)
---------------------

Updated `rand` dependency.

Depends on `rand 0.7` and `regex_syntax 0.6`. The minimal Rust version is 1.32.

v0.12.0 (2018 Nov 25)
---------------------

Updated `rand` dependency.

Depends on `rand 0.6` and `regex_syntax 0.6`. The minimal Rust version is 1.30.

v0.11.0 (2018 Oct 25)
---------------------

Initial version.

Depends on `rand 0.5` and `regex_syntax 0.6`. The minimal Rust version is 1.30.

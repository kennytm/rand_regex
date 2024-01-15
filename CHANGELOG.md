Changelog
=========

v0.17.0 (2024 Jan 16)
---------------------

Updated `regex-syntax` dependency.

Depends on `rand 0.8` and `regex-syntax 0.8`.

v0.16.0 (2023 Aug 4)
--------------------

Updated `regex-syntax` dependency.

Depends on `rand 0.8` and `regex-syntax 0.7`. The minimal Rust version is 1.71.1.

v0.15.1 (2021 Feb 12)
---------------------

No longer enables all default `rand` features.

v0.15.0 (2021 Feb 12)
---------------------

Updated `rand` dependency.

Depends on `rand 0.8` and `regex_syntax 0.6`. The minimal Rust version is 1.40.

v0.14.2 (2020 Feb 2)
--------------------

Added `Regex::is_ascii()` method to check if the regex is ASCII-only.

Added type `Encoding` to represent ASCII, UTF-8 or binary encoding.
Added `Regex::encoding()` method to return the narrowest string encoding.

Added type `EncodedString` to adjoin a byte string with its encoding.
Added an additional `impl Distribution<EncodedString> for Regex`.

v0.14.1 (2020 Jan 27)
---------------------

Added an additional `impl Distribution<Result<String, FromUtf8Error>> for Regex`.

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

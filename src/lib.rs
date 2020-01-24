#![warn(missing_docs, clippy::pedantic)]

//! Generates strings are byte strings following rule of a regular expression.
//!
//! ```
//! # #[cfg(feature = "unicode")] {
//! use rand::{SeedableRng, Rng};
//!
//! let mut rng = rand_xorshift::XorShiftRng::from_seed(*b"The initial seed");
//!
//! // creates a generator for sampling strings
//! let gen = rand_regex::Regex::compile(r"\d{4}-\d{2}-\d{2}", 100).unwrap();
//!
//! // sample a few strings randomly
//! let samples = (&mut rng).sample_iter(&gen).take(3).collect::<Vec<String>>();
//!
//! // all Unicode characters are included when sampling
//! assert_eq!(samples, vec![
//!     "᱃៧७᧗-꤂႔-૪۰".to_string(),
//!     "𝟽٩𑃶᱒-៤꣖-൭᧓".to_string(),
//!     "𑃰꩗१௭-9၅-६௫".to_string(),
//! ]);
//!
//! // you could use `regex_syntax::Hir` to include more options
//! let mut parser = regex_syntax::ParserBuilder::new().unicode(false).build();
//! let hir = parser.parse(r"\d{4}-\d{2}-\d{2}").unwrap();
//! let gen = rand_regex::Regex::with_hir(hir, 100).unwrap();
//! let samples = (&mut rng).sample_iter(&gen).take(3).collect::<Vec<String>>();
//! assert_eq!(samples, vec![
//!     "8922-87-63".to_string(),
//!     "3149-18-88".to_string(),
//!     "5420-58-55".to_string(),
//! ]);
//! # }
//! ```

#![allow(clippy::must_use_candidate)]

use rand::distributions::uniform::Uniform;
use rand::distributions::Distribution;
use rand::Rng;
use regex_syntax::hir::{self, ClassBytes, ClassUnicode, Hir, HirKind, Repetition};
use regex_syntax::Parser;
use std::borrow::Borrow;
use std::char;
use std::error;
use std::fmt::{self, Debug};
use std::mem;

const SHORT_UNICODE_CLASS_COUNT: usize = 64;

/// Error returned by [`Regex::compile()`] and [`Regex::with_hir()`].
///
/// # Examples
///
/// ```
/// let gen = rand_regex::Regex::compile(r"^.{4}\b.{4}$", 100);
/// assert_eq!(gen.err(), Some(rand_regex::Error::Anchor));
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Error {
    /// Anchors (`^`, `$`, `\A`, `\z`) and word boundary assertions (`\b`, `\B`)
    /// are not supported.
    ///
    /// If you really need to include anchors, please consider using rejection
    /// sampling e.g.
    ///
    /// ```rust
    /// # #[cfg(feature = "unicode")] {
    /// use rand::Rng;
    ///
    /// // create the generator without the anchor
    /// let gen = rand_regex::Regex::compile(r".{4}.{4}", 100).unwrap();
    ///
    /// // later filter the sampled result using a regex with the anchor
    /// let filter_regex = regex::Regex::new(r"^.{4}\b.{4}$").unwrap();
    /// let _sample = rand::thread_rng()
    ///     .sample_iter::<String, _>(&gen)
    ///     .filter(|s| filter_regex.is_match(s))
    ///     .next()
    ///     .unwrap();
    /// # }
    /// ```
    Anchor,

    /// The input regex has a syntax error.
    ///
    /// # Examples
    ///
    /// ```
    /// let gen = rand_regex::Regex::compile(r"(", 100);
    /// assert!(match gen {
    ///     Err(rand_regex::Error::Syntax(_)) => true,
    ///     _ => false,
    /// });
    /// ```
    Syntax(regex_syntax::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anchor => f.write_str("anchor is not supported"),
            Self::Syntax(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Anchor => None,
            Self::Syntax(e) => Some(e),
        }
    }
}

impl From<regex_syntax::Error> for Error {
    fn from(e: regex_syntax::Error) -> Self {
        Self::Syntax(e)
    }
}

/// A random distribution which generates strings matching the specified regex.
#[derive(Clone, Debug)]
pub struct Regex {
    compiled: Compiled,
    capacity: usize,
    is_utf8: bool,
}

impl Distribution<Vec<u8>> for Regex {
    /// Samples a random byte string satisfying the regex.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u8> {
        let mut ctx = EvalCtx {
            output: Vec::with_capacity(self.capacity),
            rng,
        };
        ctx.eval(&self.compiled);
        ctx.output
    }
}

impl Distribution<String> for Regex {
    /// Samples a random string satisfying the regex.
    ///
    /// # Panics
    ///
    /// If the regex produced some non-UTF-8 byte sequence, this method will
    /// panic. You may want to check [`is_utf8()`](Regex::is_utf8) to ensure the
    /// regex will only generate valid Unicode strings, or sample a `Vec<u8>`
    /// and manually check for UTF-8 validity.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> String {
        let bytes = <Self as Distribution<Vec<u8>>>::sample(self, rng);
        if self.is_utf8 {
            unsafe { String::from_utf8_unchecked(bytes) }
        } else {
            String::from_utf8(bytes).unwrap()
        }
    }
}

impl Default for Regex {
    /// Creates an empty regex which generates empty strings.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::Rng;
    ///
    /// let gen = rand_regex::Regex::default();
    /// assert_eq!(rand::thread_rng().sample::<String, _>(&gen), "");
    /// ```
    #[inline]
    fn default() -> Self {
        Self {
            compiled: Compiled::default(),
            capacity: 0,
            is_utf8: true,
        }
    }
}

impl Regex {
    /// Checks if the regex can only produce valid Unicode strings.
    ///
    /// # Examples
    ///
    /// ```
    /// let utf8_hir = regex_syntax::ParserBuilder::new()
    ///     .unicode(false)
    ///     .allow_invalid_utf8(true)
    ///     .build()
    ///     .parse(r"[\x00-\x7f]")
    ///     .unwrap();
    /// let utf8_gen = rand_regex::Regex::with_hir(utf8_hir, 100).unwrap();
    /// assert_eq!(utf8_gen.is_utf8(), true);
    ///
    /// let non_utf8_hir = regex_syntax::ParserBuilder::new()
    ///     .unicode(false)
    ///     .allow_invalid_utf8(true)
    ///     .build()
    ///     .parse(r"[\x00-\xff]")
    ///     .unwrap();
    /// let non_utf8_gen = rand_regex::Regex::with_hir(non_utf8_hir, 100).unwrap();
    /// assert_eq!(non_utf8_gen.is_utf8(), false);
    /// ```
    #[inline]
    pub const fn is_utf8(&self) -> bool {
        self.is_utf8
    }

    /// Returns the maximum length the string this regex can generate.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "unicode")] {
    /// let gen = rand_regex::Regex::compile(r"\d{4}-\d{2}-\d{2}", 100).unwrap();
    /// assert_eq!(gen.capacity(), 34);
    /// // each `\d` can occupy 4 bytes
    /// # }
    /// ```
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compiles a regex pattern for string generation.
    ///
    /// If you need to supply additional flags to the pattern, please use
    /// [`Regex::with_hir()`] instead.
    ///
    /// The `max_repeat` parameter gives the maximum extra repeat counts
    /// the `x*`, `x+` and `x{n,}` operators will become, e.g.
    ///
    /// ```
    /// let gen = rand_regex::Regex::compile("a{4,}", 100).unwrap();
    /// // this will generate a string between 4 to 104 characters long.
    /// assert_eq!(gen.capacity(), 104);
    /// ```
    pub fn compile(pattern: &str, max_repeat: u32) -> Result<Self, Error> {
        let hir = Parser::new().parse(pattern)?;
        Self::with_hir(hir, max_repeat)
    }

    /// Compiles a parsed regex pattern for string generation.
    ///
    /// The [`Hir`] object can be obtained using [`regex_syntax::ParserBuilder`].
    ///
    /// The `max_repeat` parameter gives the maximum extra repeat counts
    /// the `x*`, `x+` and `x{n,}` operators will become.
    pub fn with_hir(hir: Hir, max_repeat: u32) -> Result<Self, Error> {
        match hir.into_kind() {
            HirKind::Empty => Ok(Self::default()),
            HirKind::Anchor(_) | HirKind::WordBoundary(_) => Err(Error::Anchor),
            HirKind::Group(hir::Group { hir, .. }) => Self::with_hir(*hir, max_repeat),

            HirKind::Literal(hir::Literal::Unicode(c)) => Ok(Self::with_unicode_literal(c)),
            HirKind::Literal(hir::Literal::Byte(b)) => Ok(Self::with_byte_literal(b)),
            HirKind::Class(hir::Class::Unicode(class)) => Ok(Self::with_unicode_class(&class)),
            HirKind::Class(hir::Class::Bytes(class)) => Ok(Self::with_byte_class(&class)),
            HirKind::Repetition(rep) => Self::with_repetition(rep, max_repeat),
            HirKind::Concat(hirs) => Self::with_sequence(hirs, max_repeat),
            HirKind::Alternation(hirs) => Self::with_choices(hirs, max_repeat),
        }
    }

    fn with_unicode_literal(c: char) -> Self {
        let mut buf = [0_u8; 4];
        let string = c.encode_utf8(&mut buf);
        Self {
            compiled: Kind::Literal(string.as_bytes().to_owned()).into(),
            capacity: string.len(),
            is_utf8: true,
        }
    }

    fn with_byte_literal(b: u8) -> Self {
        Self {
            compiled: Kind::Literal(vec![b]).into(),
            capacity: 1,
            is_utf8: b.is_ascii(),
        }
    }

    fn with_unicode_class(class: &ClassUnicode) -> Self {
        let capacity = class
            .iter()
            .last()
            .expect("at least 1 interval")
            .end()
            .len_utf8();
        let kind = if capacity == 1 {
            let ranges = class
                .iter()
                .map(|uc| hir::ClassBytesRange::new(uc.start() as u8, uc.end() as u8));
            Kind::ByteClass(ByteClass::compile(ranges))
        } else {
            compile_unicode_class(class.ranges())
        };
        Self {
            compiled: kind.into(),
            capacity,
            is_utf8: true,
        }
    }

    fn with_byte_class(class: &ClassBytes) -> Self {
        Self {
            compiled: Kind::ByteClass(ByteClass::compile(class.iter())).into(),
            capacity: 1,
            is_utf8: class.is_all_ascii(),
        }
    }

    fn with_repetition(rep: Repetition, max_repeat: u32) -> Result<Self, Error> {
        let (lower, upper) = match rep.kind {
            hir::RepetitionKind::ZeroOrOne => (0, 1),
            hir::RepetitionKind::ZeroOrMore => (0, max_repeat),
            hir::RepetitionKind::OneOrMore => (1, 1 + max_repeat),
            hir::RepetitionKind::Range(range) => match range {
                hir::RepetitionRange::Exactly(a) => (a, a),
                hir::RepetitionRange::AtLeast(a) => (a, a + max_repeat),
                hir::RepetitionRange::Bounded(a, b) => (a, b),
            },
        };

        // simplification: `(<any>){0}` is always empty.
        if upper == 0 {
            return Ok(Self::default());
        }

        let mut regex = Self::with_hir(*rep.hir, max_repeat)?;
        regex.capacity *= upper as usize;
        if lower == upper {
            regex.compiled.repeat_const *= upper;
        } else {
            regex
                .compiled
                .repeat_ranges
                .push(Uniform::new_inclusive(lower, upper));
        }

        // simplification: if the inner is an literal, replace `x{3}` by `xxx`.
        if let Kind::Literal(lit) = &mut regex.compiled.kind {
            if regex.compiled.repeat_const > 1 {
                *lit = lit.repeat(regex.compiled.repeat_const as usize);
                regex.compiled.repeat_const = 1;
            }
        }

        Ok(regex)
    }

    fn with_sequence(hirs: Vec<Hir>, max_repeat: u32) -> Result<Self, Error> {
        let mut seq = Vec::with_capacity(hirs.len());
        let mut capacity = 0;
        let mut is_utf8 = true;

        for hir in hirs {
            let regex = Self::with_hir(hir, max_repeat)?;
            capacity += regex.capacity;
            if is_utf8 && !regex.is_utf8 {
                is_utf8 = false;
            }
            let compiled = regex.compiled;
            if compiled.is_single() {
                // simplification: `x(yz)` = `xyz`
                if let Kind::Sequence(mut s) = compiled.kind {
                    seq.append(&mut s);
                    continue;
                }
            }
            seq.push(compiled);
        }

        // Further simplify by merging adjacent literals.
        let mut simplified = Vec::with_capacity(seq.len());
        let mut combined_lit = Vec::new();
        for cur in seq {
            if cur.is_single() {
                if let Kind::Literal(mut lit) = cur.kind {
                    combined_lit.append(&mut lit);
                    continue;
                }
            }
            if !combined_lit.is_empty() {
                simplified.push(Kind::Literal(mem::take(&mut combined_lit)).into());
            }
            simplified.push(cur);
        }

        if !combined_lit.is_empty() {
            simplified.push(Kind::Literal(combined_lit).into());
        }

        let compiled = match simplified.len() {
            0 => return Ok(Self::default()),
            1 => simplified.swap_remove(0),
            _ => Kind::Sequence(simplified).into(),
        };

        Ok(Self {
            compiled,
            capacity,
            is_utf8,
        })
    }

    fn with_choices(hirs: Vec<Hir>, max_repeat: u32) -> Result<Self, Error> {
        let mut choices = Vec::with_capacity(hirs.len());
        let mut capacity = 0;
        let mut is_utf8 = true;
        for hir in hirs {
            let regex = Self::with_hir(hir, max_repeat)?;
            if regex.capacity > capacity {
                capacity = regex.capacity;
            }
            if is_utf8 && !regex.is_utf8 {
                is_utf8 = false;
            }

            let compiled = regex.compiled;
            if compiled.is_single() {
                if let Kind::Any {
                    choices: mut sc, ..
                } = compiled.kind
                {
                    choices.append(&mut sc);
                    continue;
                }
            }
            choices.push(compiled);
        }
        Ok(Self {
            compiled: Kind::Any {
                index: Uniform::new(0, choices.len()),
                choices,
            }
            .into(),
            capacity,
            is_utf8,
        })
    }
}

/// Represents a compiled regex component.
#[derive(Clone, Debug)]
struct Compiled {
    // Constant part of repetition.
    repeat_const: u32,
    // Variable parts of repetition. The repeats are multiplied together.
    repeat_ranges: Vec<Uniform<u32>>,
    // Kind of atomic regex component.
    kind: Kind,
}

impl Default for Compiled {
    fn default() -> Self {
        Kind::default().into()
    }
}

impl Compiled {
    /// Returns whether this component has no repetition.
    fn is_single(&self) -> bool {
        self.repeat_const == 1 && self.repeat_ranges.is_empty()
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Literal(Vec<u8>),
    Sequence(Vec<Compiled>),
    Any {
        index: Uniform<usize>,
        choices: Vec<Compiled>,
    },
    LongUnicodeClass(LongUnicodeClass),
    ShortUnicodeClass(ShortUnicodeClass),
    ByteClass(ByteClass),
}

impl Default for Kind {
    fn default() -> Self {
        Self::Literal(Vec::new())
    }
}

impl From<Kind> for Compiled {
    fn from(kind: Kind) -> Self {
        Self {
            repeat_const: 1,
            repeat_ranges: Vec::new(),
            kind,
        }
    }
}

struct EvalCtx<'a, R: ?Sized + 'a> {
    output: Vec<u8>,
    rng: &'a mut R,
}

impl<'a, R: Rng + ?Sized + 'a> EvalCtx<'a, R> {
    fn eval(&mut self, compiled: &Compiled) {
        let count = compiled
            .repeat_ranges
            .iter()
            .fold(compiled.repeat_const, |c, u| c * u.sample(self.rng));

        match &compiled.kind {
            Kind::Literal(lit) => self.eval_literal(count, lit),
            Kind::Sequence(seq) => self.eval_sequence(count, seq),
            Kind::Any { index, choices } => self.eval_alt(count, index, choices),
            Kind::LongUnicodeClass(class) => self.eval_unicode_class(count, class),
            Kind::ShortUnicodeClass(class) => self.eval_unicode_class(count, class),
            Kind::ByteClass(class) => self.eval_byte_class(count, class),
        }
    }

    fn eval_literal(&mut self, count: u32, lit: &[u8]) {
        for _ in 0..count {
            self.output.extend_from_slice(lit);
        }
    }

    fn eval_sequence(&mut self, count: u32, seq: &[Compiled]) {
        for _ in 0..count {
            for compiled in seq {
                self.eval(compiled);
            }
        }
    }

    fn eval_alt(&mut self, count: u32, index: &Uniform<usize>, choices: &[Compiled]) {
        for _ in 0..count {
            let idx = index.sample(self.rng);
            self.eval(&choices[idx]);
        }
    }

    fn eval_unicode_class(&mut self, count: u32, class: &impl Distribution<char>) {
        let mut buf = [0; 4];
        for c in class.sample_iter(&mut self.rng).take(count as usize) {
            let bytes = c.encode_utf8(&mut buf).as_bytes();
            self.output.extend_from_slice(bytes);
        }
    }

    fn eval_byte_class(&mut self, count: u32, class: &ByteClass) {
        self.output
            .extend(self.rng.sample_iter(class).take(count as usize));
    }
}

/// A compiled Unicode class of more than 64 ranges.
#[derive(Clone, Debug)]
struct LongUnicodeClass {
    searcher: Uniform<u32>,
    ranges: Box<[(u32, u32)]>,
}

impl Distribution<char> for LongUnicodeClass {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> char {
        let normalized_index = self.searcher.sample(rng);
        let entry_index = self
            .ranges
            .binary_search_by(|(normalized_start, _)| normalized_start.cmp(&normalized_index))
            .unwrap_or_else(|e| e - 1);
        let code = normalized_index + self.ranges[entry_index].1;
        char::from_u32(code).expect("valid char")
    }
}

/// A compiled Unicode class of less than or equals to 64 ranges.
#[derive(Clone, Debug)]
struct ShortUnicodeClass {
    index: Uniform<usize>,
    cases: Box<[char]>,
}

impl Distribution<char> for ShortUnicodeClass {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> char {
        self.cases[self.index.sample(rng)]
    }
}

fn compile_unicode_class_with(ranges: &[hir::ClassUnicodeRange], mut push: impl FnMut(char, char)) {
    for range in ranges {
        let start = range.start();
        let end = range.end();
        if start <= '\u{d7ff}' && '\u{e000}' <= end {
            push(start, '\u{d7ff}');
            push('\u{e000}', end);
        } else {
            push(start, end);
        }
    }
}

fn compile_unicode_class(ranges: &[hir::ClassUnicodeRange]) -> Kind {
    let mut normalized_ranges = Vec::new();
    let mut normalized_len = 0;
    compile_unicode_class_with(ranges, |start, end| {
        let start = u32::from(start);
        let end = u32::from(end);
        normalized_ranges.push((normalized_len, start - normalized_len));
        normalized_len += end - start + 1;
    });

    if normalized_len as usize > SHORT_UNICODE_CLASS_COUNT {
        return Kind::LongUnicodeClass(LongUnicodeClass {
            searcher: Uniform::new(0, normalized_len),
            ranges: normalized_ranges.into_boxed_slice(),
        });
    }

    // the number of cases is too small. convert into a direct search array.
    let mut cases = Vec::with_capacity(normalized_len as usize);
    compile_unicode_class_with(ranges, |start, end| {
        for c in u32::from(start)..=u32::from(end) {
            cases.push(char::from_u32(c).expect("valid char"));
        }
    });

    Kind::ShortUnicodeClass(ShortUnicodeClass {
        index: Uniform::new(0, cases.len()),
        cases: cases.into_boxed_slice(),
    })
}

/// A compiled byte class.
#[derive(Clone, Debug)]
struct ByteClass {
    index: Uniform<usize>,
    cases: Box<[u8]>,
}

impl ByteClass {
    pub fn compile(ranges: impl IntoIterator<Item = impl Borrow<hir::ClassBytesRange>>) -> Self {
        let mut cases = Vec::with_capacity(256);
        for range in ranges {
            let range = range.borrow();
            cases.extend(range.start()..=range.end());
        }
        Self {
            index: Uniform::new(0, cases.len()),
            cases: cases.into_boxed_slice(),
        }
    }
}

impl Distribution<u8> for ByteClass {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u8 {
        self.cases[self.index.sample(rng)]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::thread_rng;
    use std::collections::HashSet;
    use std::ops::RangeInclusive;

    fn check_str(pattern: &str, distinct_count: RangeInclusive<usize>, run_count: usize) {
        let r = regex::Regex::new(pattern).unwrap();
        let gen = Regex::compile(pattern, 100).unwrap();
        assert!(gen.is_utf8());

        let mut rng = thread_rng();

        let mut gen_set = HashSet::<String>::with_capacity(run_count.min(*distinct_count.end()));
        for res in (&gen).sample_iter(&mut rng).take(run_count) {
            let res: String = res;
            assert!(res.len() <= gen.capacity());
            assert!(
                r.is_match(&res),
                "Wrong sample for pattern `{}`: `{}`",
                pattern,
                res
            );
            gen_set.insert(res);
        }
        let gen_count = gen_set.len();
        assert!(
            *distinct_count.start() <= gen_count && gen_count <= *distinct_count.end(),
            "Distinct samples generated for pattern `{}` outside the range {:?}: {} (examples:\n{})",
            pattern,
            distinct_count,
            gen_count,
            gen_set.iter().take(10).map(|s| format!(" - {:#?}\n", s)).collect::<String>(),
        );
    }

    fn run_count_for_distinct_count(distinct_count: usize) -> usize {
        // Suppose a regex can possibly generate N distinct strings uniformly. What is the
        // probability distribution of number of distinct strings R we can get by running the
        // generator M times?
        //
        // Assuming we can afford M ≫ N ≈ R, we could find out the probability which (N - R) strings
        // are still *not* generated after M iterations, which is P = (1 - (R/N)^M)^(Binomial[N,R])
        // ≈ 1 - Binomial[N,R] * (R/N)^M.
        //
        // Here we choose the lower bound as R+1 after solving M for P > 0.999999, or:
        //
        //  Binomial[N,R] * (R/N)^M < 10^(-6)
        //
        // We limit M ≤ 4096 to keep the test time short.

        if distinct_count <= 1 {
            return 8;
        }
        let n = distinct_count as f64;
        ((n.ln() + 6.0 * std::f64::consts::LN_10) / (n.ln() - (n - 1.0).ln())).ceil() as usize
    }

    #[test]
    fn sanity_test_run_count() {
        assert_eq!(run_count_for_distinct_count(1), 8);
        assert_eq!(run_count_for_distinct_count(2), 21);
        assert_eq!(run_count_for_distinct_count(3), 37);
        assert_eq!(run_count_for_distinct_count(10), 153);
        assert_eq!(run_count_for_distinct_count(26), 436);
        assert_eq!(run_count_for_distinct_count(62), 1104);
        assert_eq!(run_count_for_distinct_count(128), 2381);
        assert_eq!(run_count_for_distinct_count(214), 4096);
    }

    fn check_str_limited(pattern: &str, distinct_count: usize) {
        let run_count = run_count_for_distinct_count(distinct_count);
        check_str(pattern, distinct_count..=distinct_count, run_count);
    }

    fn check_str_unlimited(pattern: &str, min_distinct_count: usize) {
        check_str(pattern, min_distinct_count..=4096, 4096);
    }

    #[test]
    fn test_proptest() {
        check_str_limited("foo", 1);
        check_str_limited("foo|bar|baz", 3);
        check_str_limited("a{0,8}", 9);
        check_str_limited("a?", 2);
        check_str_limited("a*", 101);
        check_str_limited("a+", 101);
        check_str_limited("a{4,}", 101);
        check_str_limited("(foo|bar)(xyzzy|plugh)", 4);
        check_str_unlimited(".", 3489);
        check_str_unlimited("(?s).", 3489);
    }

    #[test]
    fn test_regex_generate() {
        check_str_limited("", 1);
        check_str_limited("aBcDe", 1);
        check_str_limited("[a-zA-Z0-9]", 62);
        check_str_limited("a{3,8}", 6);
        check_str_limited("a{3}", 1);
        check_str_limited("a{3}-a{3}", 1);
        check_str_limited("(abcde)", 1);
        check_str_limited("a?b?", 4);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_cases() {
        check_str_limited("(?i:fOo)", 8);
        check_str_limited("(?i:a|B)", 4);
        check_str_unlimited(r"(\p{Greek}\P{Greek})(?:\d{3,6})", 4096);
    }

    #[test]
    fn test_ascii_character_classes() {
        check_str_limited("[[:alnum:]]", 62);
        check_str_limited("[[:alpha:]]", 52);
        check_str_limited("[[:ascii:]]", 128);
        check_str_limited("[[:blank:]]", 2);
        check_str_limited("[[:cntrl:]]", 33);
        check_str_limited("[[:digit:]]", 10);
        check_str_limited("[[:graph:]]", 94);
        check_str_limited("[[:lower:]]", 26);
        check_str_limited("[[:print:]]", 95);
        check_str_limited("[[:punct:]]", 32);
        check_str_limited("[[:space:]]", 6);
        check_str_limited("[[:upper:]]", 26);
        check_str_limited("[[:word:]]", 63);
        check_str_limited("[[:xdigit:]]", 22);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_character_classes() {
        check_str_unlimited(r"\p{L}", 3224);
        check_str(r"\p{M}", 1627..=2268, 4096);
        check_str(r"\p{N}", 1420..=1754, 4096);
        check_str(r"\p{P}", 772..=792, 4096);
        check_str_unlimited(r"\p{S}", 2355);
        check_str_limited(r"\p{Z}", 19);
        check_str_unlimited(r"\p{C}", 3478);

        check_str_unlimited(r"\P{L}", 3479);
        check_str_unlimited(r"\P{M}", 3489);
        check_str_unlimited(r"\P{N}", 3489);
        check_str_unlimited(r"\P{P}", 3489);
        check_str_unlimited(r"\P{S}", 3489);
        check_str_unlimited(r"\P{Z}", 3489);
        check_str_unlimited(r"\P{C}", 3236);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_script_classes() {
        check_str(r"\p{Latin}", 1202..=1353, 4096);
        check_str(r"\p{Greek}", 512..=518, 4096);
        check_str(r"\p{Cyrillic}", 439..=443, 4096);
        check_str_limited(r"\p{Armenian}", 95);
        check_str_limited(r"\p{Hebrew}", 134);
        check_str(r"\p{Arabic}", 1156..=1281, 4096);
        check_str_limited(r"\p{Syriac}", 88);
        check_str_limited(r"\p{Thaana}", 50);
        check_str_limited(r"\p{Devanagari}", 154);
        check_str_limited(r"\p{Bengali}", 96);
        check_str_limited(r"\p{Gurmukhi}", 80);
        check_str_limited(r"\p{Gujarati}", 91);
        check_str_limited(r"\p{Oriya}", 90);
        check_str_limited(r"\p{Tamil}", 123);
        check_str_unlimited(r"\p{Hangul}", 2585);
        check_str(r"\p{Hiragana}", 376..=379, 4096);
        check_str(r"\p{Katakana}", 302..=304, 4096);
        check_str_unlimited(r"\p{Han}", 3163);
        check_str_limited(r"\p{Tagalog}", 20);
        check_str_limited(r"\p{Linear_B}", 211);
        check_str(r"\p{Inherited}", 564..=571, 4096);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_perl_classes() {
        check_str_unlimited(r"\d+", 4046);
        check_str_unlimited(r"\D+", 4085);
        check_str_unlimited(r"\s+", 3940);
        check_str_unlimited(r"\S+", 4085);
        check_str_unlimited(r"\w+", 4083);
        check_str_unlimited(r"\W+", 4085);
    }

    #[cfg(any())]
    fn dump_categories() {
        use regex_syntax::hir::*;

        let categories = &[r"\p{Nd}", r"\p{Greek}"];

        for cat in categories {
            if let HirKind::Class(Class::Unicode(cls)) =
                regex_syntax::Parser::new().parse(cat).unwrap().into_kind()
            {
                let s: u32 = cls
                    .iter()
                    .map(|r| u32::from(r.end()) - u32::from(r.start()) + 1)
                    .sum();
                println!("{} => {}", cat, s);
            }
        }
    }

    #[test]
    fn test_binary_generator() {
        const PATTERN: &str = r"PE\x00\x00.{20}";

        let r = regex::bytes::RegexBuilder::new(PATTERN)
            .unicode(false)
            .dot_matches_new_line(true)
            .build()
            .unwrap();

        let hir = regex_syntax::ParserBuilder::new()
            .unicode(false)
            .dot_matches_new_line(true)
            .allow_invalid_utf8(true)
            .build()
            .parse(PATTERN)
            .unwrap();

        let gen = Regex::with_hir(hir, 100).unwrap();
        assert_eq!(gen.capacity(), 24);
        assert!(!gen.is_utf8());

        let mut rng = thread_rng();
        for res in gen.sample_iter(&mut rng).take(8192) {
            let res: Vec<u8> = res;
            assert!(r.is_match(&res), "Wrong sample: {:?}, `{:?}`", r, res);
        }
    }

    #[test]
    #[should_panic(expected = "FromUtf8Error")]
    fn test_generating_non_utf8_string() {
        let hir = regex_syntax::ParserBuilder::new()
            .unicode(false)
            .allow_invalid_utf8(true)
            .build()
            .parse(r"\x88")
            .unwrap();

        let gen = Regex::with_hir(hir, 100).unwrap();
        assert!(!gen.is_utf8());

        let mut rng = thread_rng();
        let _: String = rng.sample(&gen);
    }
}

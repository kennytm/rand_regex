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
//!     "꘥᥉১᪕-꧷៩-୦۱".to_string(),
//!     "𞋴۰𑋸꣕-᥆꧰-෮᪑".to_string(),
//!     "𑋲𐒥४౫-9႙-९౨".to_string()
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

use rand::{
    distributions::{uniform::Uniform, Distribution},
    Rng,
};
use regex_syntax::{
    hir::{self, ClassBytes, ClassUnicode, Hir, HirKind, Repetition},
    Parser,
};
use std::{
    char,
    cmp::Ordering,
    error,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    mem,
    str::Utf8Error,
    string::FromUtf8Error,
};

// re-export
pub use regex_syntax;

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
    Syntax(Box<regex_syntax::Error>),
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
        Self::Syntax(Box::new(e))
    }
}

/// String encoding.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Encoding {
    /// ASCII.
    Ascii = 0,
    /// UTF-8.
    Utf8 = 1,
    /// Arbitrary bytes (no encoding).
    Binary = 2,
}

/// The internal representation of [`EncodedString`], separated out to prevent
/// unchecked construction of `Es::Ascii(non_ascii_string)`.
#[derive(Debug)]
enum Es {
    /// A string with ASCII content only.
    Ascii(String),
    /// A normal string encoded with valid UTF-8
    Utf8(String),
    /// A byte string which cannot be converted to UTF-8. Contains information
    /// of failure.
    Binary(FromUtf8Error),
}

/// A string together with its [`Encoding`].
#[derive(Debug)]
pub struct EncodedString(Es);

impl EncodedString {
    /// Obtains the raw bytes of this string.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.0 {
            Es::Ascii(s) | Es::Utf8(s) => s.as_bytes(),
            Es::Binary(e) => e.as_bytes(),
        }
    }

    /// Tries to convert this instance as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// If this instance is not compatible with UTF-8, returns an error in the
    /// same manner as [`std::str::from_utf8()`].
    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        match &self.0 {
            Es::Ascii(s) | Es::Utf8(s) => Ok(s),
            Es::Binary(e) => Err(e.utf8_error()),
        }
    }

    /// Returns the encoding of this string.
    pub fn encoding(&self) -> Encoding {
        match self.0 {
            Es::Ascii(_) => Encoding::Ascii,
            Es::Utf8(_) => Encoding::Utf8,
            Es::Binary(_) => Encoding::Binary,
        }
    }
}

impl From<EncodedString> for Vec<u8> {
    fn from(es: EncodedString) -> Self {
        match es.0 {
            Es::Ascii(s) | Es::Utf8(s) => s.into_bytes(),
            Es::Binary(e) => e.into_bytes(),
        }
    }
}

impl From<Vec<u8>> for EncodedString {
    fn from(b: Vec<u8>) -> Self {
        match String::from_utf8(b) {
            Ok(s) => Self::from(s),
            Err(e) => Self(Es::Binary(e)),
        }
    }
}

impl From<String> for EncodedString {
    fn from(s: String) -> Self {
        Self(if s.is_ascii() {
            Es::Ascii(s)
        } else {
            Es::Utf8(s)
        })
    }
}

impl TryFrom<EncodedString> for String {
    type Error = FromUtf8Error;
    fn try_from(es: EncodedString) -> Result<Self, Self::Error> {
        match es.0 {
            Es::Ascii(s) | Es::Utf8(s) => Ok(s),
            Es::Binary(e) => Err(e),
        }
    }
}

impl PartialEq for EncodedString {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for EncodedString {}

impl PartialOrd for EncodedString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl Ord for EncodedString {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl Hash for EncodedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

/// A random distribution which generates strings matching the specified regex.
#[derive(Clone, Debug)]
pub struct Regex {
    compiled: Compiled,
    capacity: usize,
    encoding: Encoding,
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
    /// regex will only generate valid Unicode strings, or sample a
    /// `Result<String, FromUtf8Error>` and manually handle the error.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> String {
        <Self as Distribution<Result<_, _>>>::sample(self, rng).unwrap()
    }
}

impl Distribution<Result<String, FromUtf8Error>> for Regex {
    /// Samples a random string satisfying the regex.
    ///
    /// The the sampled bytes sequence is not valid UTF-8, the sampling result
    /// is an Err value.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<String, FromUtf8Error> {
        let bytes = <Self as Distribution<Vec<u8>>>::sample(self, rng);
        if self.is_utf8() {
            unsafe { Ok(String::from_utf8_unchecked(bytes)) }
        } else {
            String::from_utf8(bytes)
        }
    }
}

impl Distribution<EncodedString> for Regex {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> EncodedString {
        let result = <Self as Distribution<Result<_, _>>>::sample(self, rng);
        EncodedString(match result {
            Err(e) => Es::Binary(e),
            Ok(s) => {
                if self.is_ascii() || s.is_ascii() {
                    Es::Ascii(s)
                } else {
                    Es::Utf8(s)
                }
            }
        })
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
            encoding: Encoding::Ascii,
        }
    }
}

impl Regex {
    /// Obtains the narrowest string encoding this regex can produce.
    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Checks if the regex can only produce ASCII strings.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii_gen = rand_regex::Regex::compile("[0-9]+", 100).unwrap();
    /// assert_eq!(ascii_gen.is_ascii(), true);
    /// let non_ascii_gen = rand_regex::Regex::compile(r"\d+", 100).unwrap();
    /// assert_eq!(non_ascii_gen.is_ascii(), false);
    /// ```
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        // FIXME remove the `as u8` once `PartialOrd` can be used in `const fn`.
        (self.encoding as u8) == (Encoding::Ascii as u8)
    }

    /// Checks if the regex can only produce valid Unicode strings.
    ///
    /// Due to complexity of regex pattern, this method may have false
    /// negative (returning false but still always produce valid UTF-8)
    ///
    /// # Examples
    ///
    /// ```
    /// let utf8_hir = regex_syntax::ParserBuilder::new()
    ///     .unicode(false)
    ///     .utf8(false)
    ///     .build()
    ///     .parse(r"[\x00-\x7f]")
    ///     .unwrap();
    /// let utf8_gen = rand_regex::Regex::with_hir(utf8_hir, 100).unwrap();
    /// assert_eq!(utf8_gen.is_utf8(), true);
    ///
    /// let non_utf8_hir = regex_syntax::ParserBuilder::new()
    ///     .unicode(false)
    ///     .utf8(false)
    ///     .build()
    ///     .parse(r"[\x00-\xff]")
    ///     .unwrap();
    /// let non_utf8_gen = rand_regex::Regex::with_hir(non_utf8_hir, 100).unwrap();
    /// assert_eq!(non_utf8_gen.is_utf8(), false);
    /// ```
    #[inline]
    pub const fn is_utf8(&self) -> bool {
        // FIXME remove the `as u8` once `PartialOrd` can be used in `const fn`.
        (self.encoding as u8) <= (Encoding::Utf8 as u8)
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
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern is not valid regex, or contains anchors
    /// (`^`, `$`, `\A`, `\z`) or word boundary assertions (`\b`, `\B`).
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
    ///
    /// # Errors
    ///
    /// Returns an error if the `Hir` object contains anchors (`^`, `$`, `\A`,
    /// `\z`) or word boundary assertions (`\b`, `\B`).
    pub fn with_hir(hir: Hir, max_repeat: u32) -> Result<Self, Error> {
        match hir.into_kind() {
            HirKind::Empty => Ok(Self::default()),
            HirKind::Look(_) => Err(Error::Anchor),
            HirKind::Capture(hir::Capture { sub, .. }) => Self::with_hir(*sub, max_repeat),

            HirKind::Literal(hir::Literal(bytes)) => Ok(Self::with_bytes_literal(bytes.into())),
            HirKind::Class(hir::Class::Unicode(class)) => Ok(Self::with_unicode_class(&class)),
            HirKind::Class(hir::Class::Bytes(class)) => Ok(Self::with_byte_class(&class)),
            HirKind::Repetition(rep) => Self::with_repetition(rep, max_repeat),
            HirKind::Concat(hirs) => Self::with_sequence(hirs, max_repeat),
            HirKind::Alternation(hirs) => Self::with_choices(hirs, max_repeat),
        }
    }

    fn with_bytes_literal(bytes: Vec<u8>) -> Self {
        let es = EncodedString::from(bytes);
        let encoding = es.encoding();
        let bytes = Vec::from(es);
        Self {
            capacity: bytes.len(),
            compiled: Kind::Literal(bytes).into(),
            encoding,
        }
    }

    fn with_unicode_class(class: &ClassUnicode) -> Self {
        if let Some(byte_class) = class.to_byte_class() {
            Self::with_byte_class(&byte_class)
        } else {
            Self {
                compiled: compile_unicode_class(class.ranges()).into(),
                capacity: class.maximum_len().unwrap_or(0),
                encoding: Encoding::Utf8,
            }
        }
    }

    fn with_byte_class(class: &ClassBytes) -> Self {
        Self {
            compiled: Kind::ByteClass(ByteClass::compile(class.ranges())).into(),
            capacity: 1,
            encoding: if class.is_ascii() {
                Encoding::Ascii
            } else {
                Encoding::Binary
            },
        }
    }

    fn with_repetition(rep: Repetition, max_repeat: u32) -> Result<Self, Error> {
        let lower = rep.min;
        let upper = rep.max.unwrap_or(lower + max_repeat);

        // simplification: `(<any>){0}` is always empty.
        if upper == 0 {
            return Ok(Self::default());
        }

        let mut regex = Self::with_hir(*rep.sub, max_repeat)?;
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
        let mut encoding = Encoding::Ascii;

        for hir in hirs {
            let regex = Self::with_hir(hir, max_repeat)?;
            capacity += regex.capacity;
            encoding = encoding.max(regex.encoding);
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
            encoding,
        })
    }

    fn with_choices(hirs: Vec<Hir>, max_repeat: u32) -> Result<Self, Error> {
        let mut choices = Vec::with_capacity(hirs.len());
        let mut capacity = 0;
        let mut encoding = Encoding::Ascii;
        for hir in hirs {
            let regex = Self::with_hir(hir, max_repeat)?;
            if regex.capacity > capacity {
                capacity = regex.capacity;
            }
            encoding = encoding.max(regex.encoding);

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
            encoding,
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
    fn compile(ranges: &[hir::ClassBytesRange]) -> Self {
        let mut cases = Vec::with_capacity(256);
        for range in ranges {
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

    fn check_str(
        pattern: &str,
        encoding: Encoding,
        distinct_count: RangeInclusive<usize>,
        run_count: usize,
    ) {
        let r = regex::Regex::new(pattern).unwrap();
        let gen = Regex::compile(pattern, 100).unwrap();
        assert!(gen.is_utf8());
        assert_eq!(gen.encoding(), encoding);

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

    fn check_str_limited(pattern: &str, encoding: Encoding, distinct_count: usize) {
        let run_count = run_count_for_distinct_count(distinct_count);
        check_str(
            pattern,
            encoding,
            distinct_count..=distinct_count,
            run_count,
        );
    }

    fn check_str_unlimited(pattern: &str, encoding: Encoding, min_distinct_count: usize) {
        check_str(pattern, encoding, min_distinct_count..=4096, 4096);
    }

    #[test]
    fn test_proptest() {
        check_str_limited("foo", Encoding::Ascii, 1);
        check_str_limited("foo|bar|baz", Encoding::Ascii, 3);
        check_str_limited("a{0,8}", Encoding::Ascii, 9);
        check_str_limited("a?", Encoding::Ascii, 2);
        check_str_limited("a*", Encoding::Ascii, 101);
        check_str_limited("a+", Encoding::Ascii, 101);
        check_str_limited("a{4,}", Encoding::Ascii, 101);
        check_str_limited("(foo|bar)(xyzzy|plugh)", Encoding::Ascii, 4);
        check_str_unlimited(".", Encoding::Utf8, 4075);
        check_str_unlimited("(?s).", Encoding::Utf8, 4075);
    }

    #[test]
    fn test_regex_generate() {
        check_str_limited("", Encoding::Ascii, 1);
        check_str_limited("aBcDe", Encoding::Ascii, 1);
        check_str_limited("[a-zA-Z0-9]", Encoding::Ascii, 62);
        check_str_limited("a{3,8}", Encoding::Ascii, 6);
        check_str_limited("a{3}", Encoding::Ascii, 1);
        check_str_limited("a{3}-a{3}", Encoding::Ascii, 1);
        check_str_limited("(abcde)", Encoding::Ascii, 1);
        check_str_limited("a?b?", Encoding::Ascii, 4);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_cases() {
        check_str_limited("(?i:fOo)", Encoding::Ascii, 8);
        check_str_limited("(?i:a|B)", Encoding::Ascii, 4);
        check_str_unlimited(r"(\p{Greek}\P{Greek})(?:\d{3,6})", Encoding::Utf8, 4096);
    }

    #[test]
    fn test_ascii_character_classes() {
        check_str_limited("[[:alnum:]]", Encoding::Ascii, 62);
        check_str_limited("[[:alpha:]]", Encoding::Ascii, 52);
        check_str_limited("[[:ascii:]]", Encoding::Ascii, 128);
        check_str_limited("[[:blank:]]", Encoding::Ascii, 2);
        check_str_limited("[[:cntrl:]]", Encoding::Ascii, 33);
        check_str_limited("[[:digit:]]", Encoding::Ascii, 10);
        check_str_limited("[[:graph:]]", Encoding::Ascii, 94);
        check_str_limited("[[:lower:]]", Encoding::Ascii, 26);
        check_str_limited("[[:print:]]", Encoding::Ascii, 95);
        check_str_limited("[[:punct:]]", Encoding::Ascii, 32);
        check_str_limited("[[:space:]]", Encoding::Ascii, 6);
        check_str_limited("[[:upper:]]", Encoding::Ascii, 26);
        check_str_limited("[[:word:]]", Encoding::Ascii, 63);
        check_str_limited("[[:xdigit:]]", Encoding::Ascii, 22);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn sanity_test_unicode_character_classes_size() {
        // This test records the number of characters in each unicode class.
        // If any of these test failed, please:
        //  1. update the RHS of the numbers
        //  2. increase the revision number of the regex-syntax dependency
        //  3. update the relevant ranges in the test_unicode_* functions.
        //
        // (for easy reference, there are 1_112_064 assignable code points)

        fn count_class_chars(pattern: &str) -> usize {
            use regex_syntax::{
                hir::{Class, HirKind},
                parse,
            };

            let hir = parse(pattern).unwrap();
            let HirKind::Class(Class::Unicode(cls)) = hir.into_kind() else {
                unreachable!()
            };
            // we assume all positive unicode classes do not cover the surrogate range.
            // otherwise `r.len()` is wrong.
            cls.iter().map(|r| r.len()).sum()
        }

        assert_eq!(count_class_chars(r"\p{L}"), 136_104);
        assert_eq!(count_class_chars(r"\p{M}"), 2_450);
        assert_eq!(count_class_chars(r"\p{N}"), 1_831);
        assert_eq!(count_class_chars(r"\p{P}"), 842);
        assert_eq!(count_class_chars(r"\p{S}"), 7_770);
        assert_eq!(count_class_chars(r"\p{Z}"), 19);
        assert_eq!(count_class_chars(r"\p{C}"), 965_096);

        assert_eq!(count_class_chars(r"\p{Latin}"), 1_481);
        assert_eq!(count_class_chars(r"\p{Greek}"), 518);
        assert_eq!(count_class_chars(r"\p{Cyrillic}"), 506);
        assert_eq!(count_class_chars(r"\p{Armenian}"), 96);
        assert_eq!(count_class_chars(r"\p{Hebrew}"), 134);
        assert_eq!(count_class_chars(r"\p{Arabic}"), 1_368);
        assert_eq!(count_class_chars(r"\p{Syriac}"), 88);
        assert_eq!(count_class_chars(r"\p{Thaana}"), 50);
        assert_eq!(count_class_chars(r"\p{Devanagari}"), 164);
        assert_eq!(count_class_chars(r"\p{Bengali}"), 96);
        assert_eq!(count_class_chars(r"\p{Gurmukhi}"), 80);
        assert_eq!(count_class_chars(r"\p{Gujarati}"), 91);
        assert_eq!(count_class_chars(r"\p{Oriya}"), 91);
        assert_eq!(count_class_chars(r"\p{Tamil}"), 123);
        assert_eq!(count_class_chars(r"\p{Hangul}"), 11_739);
        assert_eq!(count_class_chars(r"\p{Hiragana}"), 381);
        assert_eq!(count_class_chars(r"\p{Katakana}"), 321);
        assert_eq!(count_class_chars(r"\p{Han}"), 98_408);
        assert_eq!(count_class_chars(r"\p{Tagalog}"), 23);
        assert_eq!(count_class_chars(r"\p{Linear_B}"), 211);
        assert_eq!(count_class_chars(r"\p{Inherited}"), 657);

        assert_eq!(count_class_chars(r"\d"), 680);
        assert_eq!(count_class_chars(r"\s"), 25);
        assert_eq!(count_class_chars(r"\w"), 139_612);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_character_classes() {
        // The range describes the number of distinct strings we can get from the regex.
        //
        // Suppose the class has M members. If we randomly pick N items out of it with duplicates,
        // the chance that there are K distinct members is the classical occupancy distibution[1]:
        //
        //      Occ(K|N,M) = (S2(N,K) * M!) / ((M-K)! * M^N)
        //
        // where S2(N,K) are the Stirling numbers of the second kind.
        //
        // This distribution has mean and variance of
        //
        //      μ  = M * (1 - (1 - 1/M)^N)
        //      σ² = M * ((M(M-1))^N + (M-1)(M(M-2))^N - M(M-1)^(2N)) / M^(2N)
        //
        // which we can use to approximate as a normal distrubition and calculate the CDF to find
        // out the 0.0001% percentile as the lower bound of K.
        //
        // (The upper bound should always be M, the 100% percentile.)
        //
        // The Mathematica code to compute the lower bound of K is:
        //
        //      getInterval[m_, n_] := Block[{
        //          mean = m(1-(1-1/m)^n),
        //          var = m((m(m-1))^n+(m-1)(m(m-2))^n-m(m-1)^(2n))/m^(2n)
        //      }, InverseCDF[NormalDistribution[mean, Sqrt[var]], 1*^-6]
        //
        //      (* Usage: getInterval[2450, 4096.0`32] *)
        //
        // [1]: https://doi.org/10.1080/00031305.2019.1699445

        check_str_unlimited(r"\p{L}", Encoding::Utf8, 3999);
        check_str(r"\p{M}", Encoding::Utf8, 1918..=2450, 4096);
        check_str(r"\p{N}", Encoding::Utf8, 1582..=1831, 4096);
        check_str(r"\p{P}", Encoding::Utf8, 824..=842, 4096);
        check_str_unlimited(r"\p{S}", Encoding::Utf8, 3083);
        check_str_limited(r"\p{Z}", Encoding::Utf8, 19);
        check_str_unlimited(r"\p{C}", Encoding::Utf8, 4073);

        check_str_unlimited(r"\P{L}", Encoding::Utf8, 4074);
        check_str_unlimited(r"\P{M}", Encoding::Utf8, 4075);
        check_str_unlimited(r"\P{N}", Encoding::Utf8, 4075);
        check_str_unlimited(r"\P{P}", Encoding::Utf8, 4075);
        check_str_unlimited(r"\P{S}", Encoding::Utf8, 4075);
        check_str_unlimited(r"\P{Z}", Encoding::Utf8, 4075);
        check_str_unlimited(r"\P{C}", Encoding::Utf8, 4004);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_unicode_script_classes() {
        check_str(r"\p{Latin}", Encoding::Utf8, 1348..=1481, 4096);
        check_str(r"\p{Greek}", Encoding::Utf8, 516..=518, 4096);
        check_str(r"\p{Cyrillic}", Encoding::Utf8, 504..=506, 4096);
        check_str_limited(r"\p{Armenian}", Encoding::Utf8, 96);
        check_str_limited(r"\p{Hebrew}", Encoding::Utf8, 134);
        check_str(r"\p{Arabic}", Encoding::Utf8, 1264..=1368, 4096);
        check_str_limited(r"\p{Syriac}", Encoding::Utf8, 88);
        check_str_limited(r"\p{Thaana}", Encoding::Utf8, 50);
        check_str_limited(r"\p{Devanagari}", Encoding::Utf8, 164);
        check_str_limited(r"\p{Bengali}", Encoding::Utf8, 96);
        check_str_limited(r"\p{Gurmukhi}", Encoding::Utf8, 80);
        check_str_limited(r"\p{Gujarati}", Encoding::Utf8, 91);
        check_str_limited(r"\p{Oriya}", Encoding::Utf8, 91);
        check_str_limited(r"\p{Tamil}", Encoding::Utf8, 123);
        check_str_unlimited(r"\p{Hangul}", Encoding::Utf8, 3363);
        check_str_limited(r"\p{Hiragana}", Encoding::Utf8, 381);
        check_str_limited(r"\p{Katakana}", Encoding::Utf8, 321);
        check_str_unlimited(r"\p{Han}", Encoding::Utf8, 3970);
        check_str_limited(r"\p{Tagalog}", Encoding::Utf8, 23);
        check_str_limited(r"\p{Linear_B}", Encoding::Utf8, 211);
        check_str(r"\p{Inherited}", Encoding::Utf8, 650..=657, 4096);
    }

    #[test]
    #[cfg(feature = "unicode")]
    fn test_perl_classes() {
        check_str_unlimited(r"\d+", Encoding::Utf8, 4061);
        check_str_unlimited(r"\D+", Encoding::Utf8, 4096);
        check_str_unlimited(r"\s+", Encoding::Utf8, 4014);
        check_str_unlimited(r"\S+", Encoding::Utf8, 4096);
        check_str_unlimited(r"\w+", Encoding::Utf8, 4095);
        check_str_unlimited(r"\W+", Encoding::Utf8, 4096);
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
            .utf8(false)
            .build()
            .parse(PATTERN)
            .unwrap();

        let gen = Regex::with_hir(hir, 100).unwrap();
        assert_eq!(gen.capacity(), 24);
        assert!(!gen.is_utf8());
        assert_eq!(gen.encoding(), Encoding::Binary);

        let mut rng = thread_rng();
        for res in gen.sample_iter(&mut rng).take(8192) {
            let res: Vec<u8> = res;
            assert!(r.is_match(&res), "Wrong sample: {:?}, `{:?}`", r, res);
        }
    }

    #[test]
    fn test_encoding_generator_1() {
        let hir = regex_syntax::ParserBuilder::new()
            .unicode(false)
            .utf8(false)
            .build()
            .parse(r"[\x00-\xff]{2}")
            .unwrap();
        let gen = Regex::with_hir(hir, 100).unwrap();

        // This pattern will produce:
        //  - 16384 ASCII patterns (128^2)
        //  -  1920 UTF-8 patterns (30 * 64)
        //  - 47232 binary patterns (256^2 - 16384 - 1920)

        let mut encoding_counts = [0; 3];
        let mut rng = thread_rng();
        for encoded_string in gen.sample_iter(&mut rng).take(8192) {
            let encoded_string: EncodedString = encoded_string;
            let bytes = encoded_string.as_bytes();
            let encoding = encoded_string.encoding();
            assert_eq!(bytes.len(), 2);
            if bytes.is_ascii() {
                assert_eq!(encoding, Encoding::Ascii);
            } else if std::str::from_utf8(bytes).is_ok() {
                assert_eq!(encoding, Encoding::Utf8);
            } else {
                assert_eq!(encoding, Encoding::Binary);
            }
            encoding_counts[encoding as usize] += 1;
        }

        // the following ranges are 99.9999% confidence intervals of the multinomial distribution.
        assert!((1858..2243).contains(&encoding_counts[Encoding::Ascii as usize]));
        assert!((169..319).contains(&encoding_counts[Encoding::Utf8 as usize]));
        assert!((5704..6102).contains(&encoding_counts[Encoding::Binary as usize]));
    }

    #[test]
    fn test_encoding_generator_2() {
        let gen = Regex::compile(r"[\u{0}-\u{b5}]{2}", 100).unwrap();

        // This pattern will produce 32761 distinct outputs, with:
        //  - 16384 ASCII patterns
        //  - 16377 UTF-8 patterns

        let mut encoding_counts = [0; 2];
        let mut rng = thread_rng();
        for encoded_string in gen.sample_iter(&mut rng).take(8192) {
            let encoded_string: EncodedString = encoded_string;
            let encoding = encoded_string.encoding();
            let string = encoded_string.as_str().unwrap();
            assert_eq!(string.chars().count(), 2);
            if string.is_ascii() {
                assert_eq!(encoding, Encoding::Ascii);
                assert_eq!(string.len(), 2);
            } else {
                assert_eq!(encoding, Encoding::Utf8);
            }
            encoding_counts[encoding as usize] += 1;
        }

        // the following ranges are 99.9999% confidence intervals of the multinomial distribution.
        assert!((3876..4319).contains(&encoding_counts[Encoding::Ascii as usize]));
        assert!((3874..4317).contains(&encoding_counts[Encoding::Utf8 as usize]));
    }

    #[test]
    fn test_encoding_generator_3() {
        let gen = Regex::compile(r"[\u{0}-\u{7f}]{2}", 100).unwrap();
        let mut rng = thread_rng();
        for encoded_string in gen.sample_iter(&mut rng).take(8192) {
            let encoded_string: EncodedString = encoded_string;
            assert_eq!(encoded_string.encoding(), Encoding::Ascii);
            assert_eq!(String::try_from(encoded_string).unwrap().len(), 2);
        }
    }

    #[test]
    #[should_panic(expected = "FromUtf8Error")]
    fn test_generating_non_utf8_string() {
        let hir = regex_syntax::ParserBuilder::new()
            .unicode(false)
            .utf8(false)
            .build()
            .parse(r"\x88")
            .unwrap();

        let gen = Regex::with_hir(hir, 100).unwrap();
        assert!(!gen.is_utf8());
        assert_eq!(gen.encoding(), Encoding::Binary);

        let mut rng = thread_rng();
        let _: String = rng.sample(&gen);
    }
}

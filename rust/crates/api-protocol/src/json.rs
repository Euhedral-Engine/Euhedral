//! JSON values for the whole crate: the chat template's `tojson` form (Python's `json.dumps(value,
//! ensure_ascii=False)`), the compact form of tool-call arguments, and a strict parser.

use std::collections::HashSet;
use std::fmt::Write;

/// The deepest nesting a parsed document may have.
/// The deepest nesting of arrays and objects a document may have, and the most digits one number may have
/// (the integer digits for an integer; integer, fraction and exponent digits together for a float), and the
/// longest object key in UTF-16 code units (Jackson's characters): the limits of the reader the wire format is defined by.
const MAX_DEPTH: usize = 500;
const MAX_NUMBER_DIGITS: usize = 1000;
const MAX_NAME_UNITS: usize = 50_000;
/// The most keys of one object that the distinct reading compares one by one; past it the keys go into a set.
const LINEAR_KEY_LIMIT: usize = 8;

/// A JSON value whose objects keep their insertion order.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    /// An integer literal of any size; written as its canonical decimal digits.
    Integer(JsonInteger),
    /// A fractional or exponent literal; written as Python's `repr` of a float.
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    /// Members in the order they are written.
    Object(Vec<(String, JsonValue)>),
}

/// An integer of any size, held as its canonical decimal digits: no leading zeros or plus sign, and zero is `0`
/// whatever its sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonInteger(String);

impl JsonInteger {
    /// The integer written as an optional `-` and decimal digits; `None` for any other text.
    pub fn parse(text: &str) -> Option<Self> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let digits = digits.trim_start_matches('0');
        Some(Self(match (digits.is_empty(), negative) {
            (true, _) => "0".to_owned(),
            (false, true) => format!("-{digits}"),
            (false, false) => digits.to_owned(),
        }))
    }

    /// The canonical decimal digits.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<i128> for JsonInteger {
    fn from(value: i128) -> Self {
        Self(value.to_string())
    }
}

impl From<u128> for JsonInteger {
    fn from(value: u128) -> Self {
        Self(value.to_string())
    }
}

impl JsonValue {
    /// The text of a value as the wire format spells it in a `param`: strings bare, lists as `[a, b]`, objects as
    /// `{k=v}`, integers by their canonical digits (zero has no sign) and doubles as `NaN`, `Infinity`, `-Infinity` or the shortest
    /// digits in the compact form.
    pub fn param_text(&self) -> String {
        match self {
            JsonValue::Null => "null".to_owned(),
            JsonValue::Bool(flag) => flag.to_string(),
            JsonValue::Integer(digits) => digits.as_str().to_owned(),
            JsonValue::Float(number) => compact_float(*number),
            JsonValue::String(text) => text.clone(),
            JsonValue::Array(items) => {
                let items: Vec<String> = items.iter().map(JsonValue::param_text).collect();
                format!("[{}]", items.join(", "))
            }
            JsonValue::Object(members) => {
                let members: Vec<String> = members
                    .iter()
                    .map(|(key, value)| format!("{key}={}", value.param_text()))
                    .collect();
                format!("{{{}}}", members.join(", "))
            }
        }
    }

    /// The value as Python's `json.dumps(value, ensure_ascii=False)` writes it: `", "` and `": "` separators,
    /// insertion key order, only `"`, `\` and C0 controls escaped, non-finite floats as `Infinity`, `-Infinity`
    /// and `NaN`.
    pub fn to_python_json(&self) -> String {
        let mut out = String::new();
        self.write_python_json(&mut out);
        out
    }

    fn write_python_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Integer(value) => out.push_str(value.as_str()),
            Self::Float(value) => out.push_str(&python_float(*value)),
            Self::String(text) => write_string(text, out),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.write_python_json(out);
                }
                out.push(']');
            }
            Self::Object(members) => {
                out.push('{');
                for (index, (key, value)) in members.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write_string(key, out);
                    out.push_str(": ");
                    value.write_python_json(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for value in text.chars() {
        match value {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if control < ' ' => {
                let _ = write!(out, "\\u{:04x}", u32::from(control));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Python's `float.__repr__`: the shortest round-tripping digits, positional when the decimal exponent is in
/// [-4, 16), otherwise `d.ddde±XX`.
pub(crate) fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }
    // `{:e}` writes the shortest digits that round-trip: `d[.ddd]e<exponent>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exponent) {
        let positional = if exponent >= 0 {
            let integer_len = exponent as usize + 1;
            if digits.len() <= integer_len {
                format!("{digits}{}.0", "0".repeat(integer_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..integer_len], &digits[integer_len..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
        };
        return format!("{sign}{positional}");
    }
    let mantissa = if digits.len() == 1 {
        digits
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let exponent_sign = if exponent < 0 { '-' } else { '+' };
    format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
}

impl JsonValue {
    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(members) => members.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    /// The name of the value's JSON type; integers are `integer`, other numbers `number`.
    pub fn type_name(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Integer(_) => "integer",
            JsonValue::Float(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }

    /// False when a number anywhere in the value is infinite or not a number.
    pub fn is_finite(&self) -> bool {
        match self {
            JsonValue::Float(number) => number.is_finite(),
            JsonValue::Array(items) => items.iter().all(JsonValue::is_finite),
            JsonValue::Object(members) => members.iter().all(|(_, value)| value.is_finite()),
            _ => true,
        }
    }

    /// Reads one whole document strictly; `None` when `text` is not exactly one JSON value. Integers of any
    /// length keep their digits; a repeated object key replaces the earlier value in place.
    pub fn parse(text: &str) -> Option<JsonValue> {
        JsonValue::read(text, false)
    }

    /// As [`parse`](JsonValue::parse), except that an object with a repeated key is not a document: `None`. This is
    /// the reading that decides whether a model's answer is JSON.
    pub fn parse_distinct(text: &str) -> Option<JsonValue> {
        JsonValue::read(text, true)
    }

    fn read(text: &str, distinct: bool) -> Option<JsonValue> {
        let mut parser = Parser {
            text,
            position: 0,
            distinct,
            lone: Vec::new(),
        };
        parser.skip_space();
        let value = parser.value(0)?;
        parser.skip_space();
        (parser.position == text.len()).then_some(value)
    }

    /// The compact form: no spaces, escapes in capitals, other numbers in the form of Java's
    /// `Double.toString` (`1.0E10`, positional from `1e-3` up to `1e7`).
    pub fn to_compact_string(&self) -> String {
        let mut json = String::new();
        self.write_compact(&mut json);
        json
    }

    fn write_compact(&self, json: &mut String) {
        match self {
            JsonValue::Null => json.push_str("null"),
            JsonValue::Bool(flag) => json.push_str(if *flag { "true" } else { "false" }),
            JsonValue::Integer(digits) => json.push_str(digits.as_str()),
            JsonValue::Float(number) => json.push_str(&compact_float(*number)),
            JsonValue::String(text) => write_compact_string(text, json),
            JsonValue::Array(items) => {
                json.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        json.push(',');
                    }
                    item.write_compact(json);
                }
                json.push(']');
            }
            JsonValue::Object(members) => {
                json.push('{');
                for (index, (key, value)) in members.iter().enumerate() {
                    if index > 0 {
                        json.push(',');
                    }
                    write_compact_string(key, json);
                    json.push(':');
                    value.write_compact(json);
                }
                json.push('}');
            }
        }
    }
}

/// Appends a member, replacing the value of a key already present.
pub(crate) fn put(members: &mut Vec<(String, JsonValue)>, key: String, value: JsonValue) {
    match members.iter_mut().find(|(name, _)| *name == key) {
        Some(member) => member.1 = value,
        None => members.push((key, value)),
    }
}

/// Quotes a string for the compact form: `"`, `\` and the controls are escaped, the latter with capital hex digits.
fn write_compact_string(text: &str, json: &mut String) {
    json.push('"');
    for value in text.chars() {
        match value {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            '\u{8}' => json.push_str("\\b"),
            '\u{c}' => json.push_str("\\f"),
            value if value < ' ' => {
                let _ = write!(json, "\\u{:04X}", u32::from(value));
            }
            value => json.push(value),
        }
    }
    json.push('"');
}

/// A double in the form above; the non-finite values are `NaN`, `Infinity` and `-Infinity`.
fn compact_float(number: f64) -> String {
    if number.is_nan() {
        return "NaN".to_owned();
    }
    if number.is_infinite() {
        return if number > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if number == 0.0 {
        return if number.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let scientific = format!("{:e}", number.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|value| *value != '.').collect();
    let sign = if number < 0.0 { "-" } else { "" };
    if (-3..7).contains(&exponent) {
        let point = exponent + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat(usize::try_from(-point).unwrap_or(0)), digits)
        } else {
            let point = usize::try_from(point).unwrap_or(0);
            if digits.len() <= point {
                format!("{}{}.0", digits, "0".repeat(point - digits.len()))
            } else {
                format!("{}.{}", &digits[..point], &digits[point..])
            }
        };
        format!("{sign}{body}")
    } else {
        let fraction = if digits.len() > 1 { &digits[1..] } else { "0" };
        format!("{sign}{}.{fraction}E{exponent}", &digits[..1])
    }
}

struct Parser<'a> {
    text: &'a str,
    position: usize,
    /// A repeated object key ends the parse.
    distinct: bool,
    /// The lone surrogates of the string last read, as the byte offset in its text where each one was replaced
    /// and its code unit.
    lone: Vec<(usize, u16)>,
}

/// A key as the strict reading tells keys apart: its text and where lone surrogates stood in it, so that two
/// different lone surrogates, or one and a literal U+FFFD, are different keys.
type KeyIdentity = (String, Vec<(usize, u16)>);

/// The keys an object has had so far, for the strict reading: a short list while it is small, a set beyond.
#[derive(Default)]
struct SeenKeys {
    few: Vec<KeyIdentity>,
    many: HashSet<KeyIdentity>,
}

impl SeenKeys {
    /// Records `key`; false when the object already had it.
    fn insert(&mut self, key: KeyIdentity) -> bool {
        if !self.many.is_empty() {
            return self.many.insert(key);
        }
        if self.few.contains(&key) {
            return false;
        }
        if self.few.len() < LINEAR_KEY_LIMIT {
            self.few.push(key);
        } else {
            self.many = self.few.drain(..).collect();
            self.many.insert(key);
        }
        true
    }
}

impl Parser<'_> {
    fn rest(&self) -> &str {
        &self.text[self.position..]
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.position).copied()
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn literal(&mut self, word: &str, value: JsonValue) -> Option<JsonValue> {
        if self.rest().starts_with(word) {
            self.position += word.len();
            Some(value)
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<JsonValue> {
        match self.peek()? {
            b'n' => self.literal("null", JsonValue::Null),
            b't' => self.literal("true", JsonValue::Bool(true)),
            b'f' => self.literal("false", JsonValue::Bool(false)),
            b'"' => self.string().map(JsonValue::String),
            b'[' => self.array(depth),
            b'{' => self.object(depth),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn array(&mut self, depth: usize) -> Option<JsonValue> {
        if depth >= MAX_DEPTH {
            return None;
        }
        self.position += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek()? == b']' {
            self.position += 1;
            return Some(JsonValue::Array(items));
        }
        loop {
            self.skip_space();
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.peek()? {
                b',' => self.position += 1,
                b']' => {
                    self.position += 1;
                    return Some(JsonValue::Array(items));
                }
                _ => return None,
            }
        }
    }

    fn object(&mut self, depth: usize) -> Option<JsonValue> {
        if depth >= MAX_DEPTH {
            return None;
        }
        self.position += 1;
        let mut members = Vec::new();
        let mut seen = SeenKeys::default();
        self.skip_space();
        if self.peek()? == b'}' {
            self.position += 1;
            return Some(JsonValue::Object(members));
        }
        loop {
            self.skip_space();
            if self.peek()? != b'"' {
                return None;
            }
            let key = self.string()?;
            let lone = std::mem::take(&mut self.lone);
            if key.len() > 3 * MAX_NAME_UNITS || key.encode_utf16().count() > MAX_NAME_UNITS {
                return None;
            }
            self.skip_space();
            if self.peek()? != b':' {
                return None;
            }
            self.position += 1;
            self.skip_space();
            let value = self.value(depth + 1)?;
            if self.distinct {
                if !seen.insert((key.clone(), lone)) {
                    return None;
                }
                members.push((key, value));
            } else {
                put(&mut members, key, value);
            }
            self.skip_space();
            match self.peek()? {
                b',' => self.position += 1,
                b'}' => {
                    self.position += 1;
                    return Some(JsonValue::Object(members));
                }
                _ => return None,
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.position;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        self.position - start
    }

    fn number(&mut self) -> Option<JsonValue> {
        let start = self.position;
        let mut digit_count = 0;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek()? {
            b'0' => self.position += 1,
            b'1'..=b'9' => {
                digit_count += self.digits();
            }
            _ => return None,
        }
        if digit_count == 0 {
            digit_count = 1;
        }
        let mut integer = true;
        if self.peek() == Some(b'.') {
            integer = false;
            self.position += 1;
            let fraction = self.digits();
            if fraction == 0 {
                return None;
            }
            digit_count += fraction;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integer = false;
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            let exponent = self.digits();
            if exponent == 0 {
                return None;
            }
            digit_count += exponent;
        }
        if digit_count > MAX_NUMBER_DIGITS {
            return None;
        }
        let literal = &self.text[start..self.position];
        if integer {
            return JsonInteger::parse(literal).map(JsonValue::Integer);
        }
        literal.parse::<f64>().ok().map(JsonValue::Float)
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = self.rest().get(..4)?;
        if !digits.bytes().all(|value| value.is_ascii_hexdigit()) {
            return None;
        }
        let unit = u32::from_str_radix(digits, 16).ok()?;
        self.position += 4;
        Some(unit)
    }

    fn string(&mut self) -> Option<String> {
        self.position += 1;
        self.lone.clear();
        let mut text = String::new();
        loop {
            let value = self.rest().chars().next()?;
            self.position += value.len_utf8();
            match value {
                '"' => return Some(text),
                '\\' => {
                    let escape = self.rest().chars().next()?;
                    self.position += escape.len_utf8();
                    match escape {
                        '"' | '\\' | '/' => text.push(escape),
                        'b' => text.push('\u{8}'),
                        'f' => text.push('\u{c}'),
                        'n' => text.push('\n'),
                        'r' => text.push('\r'),
                        't' => text.push('\t'),
                        'u' => {
                            let unit = self.hex4()?;
                            let scalar = self.scalar(unit)?;
                            if scalar == '\u{fffd}' && (0xd800..0xe000).contains(&unit) {
                                self.lone.push((text.len(), unit as u16));
                            }
                            text.push(scalar);
                        }
                        _ => return None,
                    }
                }
                value if value < ' ' => return None,
                value => text.push(value),
            }
        }
    }

    /// The character of a `\u` escape; a high surrogate takes the low one that follows, and a surrogate left
    /// alone becomes U+FFFD.
    fn scalar(&mut self, unit: u32) -> Option<char> {
        if (0xd800..0xdc00).contains(&unit) && self.rest().starts_with("\\u") {
            let saved = self.position;
            self.position += 2;
            match self.hex4() {
                Some(low) if (0xdc00..0xe000).contains(&low) => {
                    return char::from_u32(0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00));
                }
                _ => self.position = saved,
            }
        }
        Some(char::from_u32(unit).unwrap_or('\u{fffd}'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_read_and_write_back_compactly() {
        let text = r#"{"b": [1, 2.5, "x\ny", null, true], "a": {"k": -0}, "b": 7}"#;
        let value = JsonValue::parse(text).unwrap();
        assert_eq!(value.to_compact_string(), r#"{"b":7,"a":{"k":0}}"#);
    }

    #[test]
    fn a_repeated_key_is_refused_by_the_distinct_reading_only() {
        let repeated = r#"{"a": 1, "b": {"c": 1, "c": 2}}"#;
        assert!(JsonValue::parse(repeated).is_some());
        assert!(JsonValue::parse_distinct(repeated).is_none());
        assert!(JsonValue::parse_distinct(r#"{"a": {"c": 1}, "b": {"c": 2}}"#).is_some());
        assert!(JsonValue::parse_distinct(r#"{"a":1} x"#).is_none());
    }

    #[test]
    fn distinct_lone_surrogate_keys_are_not_repeats() {
        let read = |text: &str| JsonValue::parse_distinct(text);
        assert!(read(r#"{"\ud800":1,"\ud801":2}"#).is_some());
        assert!(read(r#"{"\udc00":1,"\ud800":2}"#).is_some());
        assert!(read(r#"{"\ud800":1,"\ufffd":2}"#).is_some());
        assert!(read(r#"{"a\ud800":1,"a\ud801b":2}"#).is_some());
        assert!(read(r#"{"\ud800":1,"\ud800":2}"#).is_none());
        assert!(read(r#"{"\ud83d\ude00":1,"\ud83d\ude01":2}"#).is_some());
        assert!(read(r#"{"\ud83d\ude00":1,"\ud83d\ude00":2}"#).is_none());
    }

    #[test]
    fn a_repeat_is_found_among_many_keys() {
        let keys = |count: usize, last: &str| {
            let members: Vec<String> = (0..count).map(|index| format!(r#""k{index}":{index}"#)).collect();
            format!("{{{},{last}}}", members.join(","))
        };
        for count in [LINEAR_KEY_LIMIT - 1, LINEAR_KEY_LIMIT, LINEAR_KEY_LIMIT + 1, 5000] {
            assert!(
                JsonValue::parse_distinct(&keys(count, r#""other":0"#)).is_some(),
                "{count}"
            );
            assert!(
                JsonValue::parse_distinct(&keys(count, r#""k0":0"#)).is_none(),
                "{count}"
            );
            let last = format!(r#""k{}":0"#, count - 1);
            assert!(JsonValue::parse_distinct(&keys(count, &last)).is_none(), "{count}");
        }
    }

    #[test]
    fn invalid_documents_are_refused() {
        for text in [
            "",
            "3 4",
            "{\"k\": 1} x",
            "01",
            "1.",
            ".5",
            "+1",
            "[1,]",
            "{\"a\"}",
            "'a'",
            "NaN",
            "\"a\nb\"",
            "tru",
        ] {
            assert!(JsonValue::parse(text).is_none(), "{text:?}");
        }
    }

    #[test]
    fn deep_nesting_is_refused_instead_of_overflowing_the_stack() {
        let text = format!("{}{}", "[".repeat(50_000), "]".repeat(50_000));
        assert!(JsonValue::parse(&text).is_none());
    }

    #[test]
    fn floats_print_as_the_shortest_round_trip() {
        let cases = [
            (0.25, "0.25"),
            (3.0, "3.0"),
            (1.0e7, "1.0E7"),
            (9999999.5, "9999999.5"),
            (0.001, "0.001"),
            (0.0001, "1.0E-4"),
            (1.5e20, "1.5E20"),
            (-2.5e-7, "-2.5E-7"),
            (123456.789, "123456.789"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
        ];
        for (number, expected) in cases {
            assert_eq!(compact_float(number), expected);
        }
    }

    #[test]
    fn strings_escape_controls_in_capitals() {
        let value = JsonValue::String("a\u{1f}\"\\é\u{7f}".to_string());
        assert_eq!(value.to_compact_string(), "\"a\\u001F\\\"\\\\é\u{7f}\"");
    }

    #[test]
    fn escapes_decode_including_surrogate_pairs() {
        let value = JsonValue::parse(r#""\u00e9\ud83d\ude00\ud800x""#).unwrap();
        assert_eq!(value, JsonValue::String("é😀\u{fffd}x".to_string()));
    }

    #[test]
    fn huge_integers_keep_their_digits() {
        let value = JsonValue::parse("123456789012345678901234567890").unwrap();
        assert_eq!(value.to_compact_string(), "123456789012345678901234567890");
        assert!(matches!(JsonValue::parse("1e400"), Some(JsonValue::Float(number)) if number.is_infinite()));
    }

    #[test]
    fn param_text_spells_values_as_the_wire_format_does() {
        let text = |json: &str| JsonValue::parse(json).expect("json").param_text();
        assert_eq!(text("1e400"), "Infinity");
        assert_eq!(text("-1e400"), "-Infinity");
        assert_eq!(text("-0"), "0");
        assert_eq!(text("-0.0"), "-0.0");
        assert_eq!(text("[-0, 1.5, null, true]"), "[0, 1.5, null, true]");
        assert_eq!(text(r#"{"a":"x","b":[1,{"c":2.0}]}"#), "{a=x, b=[1, {c=2.0}]}");
        assert_eq!(JsonValue::Float(f64::NAN).param_text(), "NaN");
        assert_eq!(JsonValue::Float(f64::NEG_INFINITY).param_text(), "-Infinity");
    }

    #[test]
    fn python_floats_follow_repr() {
        let cases = [
            (1.0e-5, "1e-05"),
            (1.0e16, "1e+16"),
            (1.0e15, "1000000000000000.0"),
            (2.0, "2.0"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (-1.5e300, "-1.5e+300"),
            (f64::NAN, "NaN"),
            (f64::NEG_INFINITY, "-Infinity"),
            (-0.0, "-0.0"),
        ];
        for (number, expected) in cases {
            assert_eq!(python_float(number), expected);
        }
    }

    #[test]
    fn python_documents_use_python_separators_and_keep_non_ascii_text() {
        let value = JsonValue::parse(r#"{"a": [1, 2.50, "é\u0001\n"], "b": {"c": null, "d": true}, "e": {}, "f": []}"#)
            .unwrap();
        assert_eq!(
            value.to_python_json(),
            "{\"a\": [1, 2.5, \"é\\u0001\\n\"], \"b\": {\"c\": null, \"d\": true}, \"e\": {}, \"f\": []}"
        );
    }

    #[test]
    fn trailing_text_after_the_value_is_refused() {
        assert_eq!(JsonValue::parse("{} x"), None);
        assert_eq!(JsonValue::parse("1 2"), None);
        assert!(JsonValue::parse(" {\"a\":1} ").is_some());
    }

    fn nested(open: &str, close: &str, levels: usize) -> String {
        format!("{}{}", open.repeat(levels), close.repeat(levels))
    }

    #[test]
    fn nesting_of_five_hundred_levels_is_read_and_one_more_is_refused() {
        assert!(JsonValue::parse(&nested("[", "]", 500)).is_some());
        assert!(JsonValue::parse(&nested("[", "]", 501)).is_none());
        let objects = |levels: usize| format!("{}1{}", "{\"a\":".repeat(levels), "}".repeat(levels));
        assert!(JsonValue::parse(&objects(500)).is_some());
        assert!(JsonValue::parse(&objects(501)).is_none());
        let mixed = format!("{}0{}", "[{\"a\":".repeat(250), "}]".repeat(250));
        assert!(JsonValue::parse(&mixed).is_some());
        let mixed = format!("[{}0{}]", "[{\"a\":".repeat(250), "}]".repeat(250));
        assert!(JsonValue::parse(&mixed).is_none());
    }

    #[test]
    fn numbers_of_a_thousand_digits_are_read_and_one_more_is_refused() {
        let digits = |count: usize| "1".repeat(count);
        assert!(JsonValue::parse(&digits(1000)).is_some());
        assert!(JsonValue::parse(&digits(1001)).is_none());
        assert!(JsonValue::parse(&format!("-{}", digits(1000))).is_some());
        assert!(JsonValue::parse(&format!("-{}", digits(1001))).is_none());
        assert!(JsonValue::parse(&format!("{}.{}", digits(500), digits(500))).is_some());
        assert!(JsonValue::parse(&format!("{}.{}", digits(500), digits(501))).is_none());
        assert!(JsonValue::parse(&format!("-{}.{}", digits(500), digits(500))).is_some());
        assert!(JsonValue::parse(&format!("1.{}e-{}", digits(499), digits(500))).is_some());
        assert!(JsonValue::parse(&format!("1.{}e-{}", digits(499), digits(501))).is_none());
        assert!(JsonValue::parse(&format!("0.{}", digits(999))).is_some());
        assert!(JsonValue::parse(&format!("0.{}", digits(1000))).is_none());
    }

    #[test]
    fn object_keys_of_fifty_thousand_characters_are_read_and_one_more_is_refused() {
        let object = |key: String| format!("{{\"{key}\":1}}");
        assert!(JsonValue::parse(&object("k".repeat(50_000))).is_some());
        assert!(JsonValue::parse(&object("k".repeat(50_001))).is_none());
        assert!(JsonValue::parse(&object("\u{e9}".repeat(50_000))).is_some());
        assert!(JsonValue::parse(&object("\u{e9}".repeat(50_001))).is_none());
        // Three bytes each, one unit each; an astral character is two units.
        assert!(JsonValue::parse(&object("\u{20ac}".repeat(50_000))).is_some());
        assert!(JsonValue::parse(&object("\u{1f600}".repeat(25_000))).is_some());
        assert!(JsonValue::parse(&object("\u{1f600}".repeat(25_001))).is_none());
        let long_value = format!("[\"{}\"]", "v".repeat(200_000));
        assert!(JsonValue::parse(&long_value).is_some());
    }
}

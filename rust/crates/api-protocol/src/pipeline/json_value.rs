//! A JSON document as the pipeline reads and writes it: object members keep their insertion order, integers keep
//! their digits, and a document is parsed strictly (no trailing text, no comments, no non-finite literals).

use std::fmt::Write;

/// The deepest nesting a document may have.
const MAX_DEPTH: usize = 1000;

/// One JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    /// An integer of any size, as its decimal digits with a leading `-` when negative.
    Integer(String),
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    /// Members in insertion order; a repeated key replaces the earlier value in place.
    Object(Vec<(String, JsonValue)>),
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

    /// Reads one whole document; `None` when `text` is not exactly one JSON value.
    pub fn parse(text: &str) -> Option<JsonValue> {
        let mut parser = Parser { text, position: 0 };
        parser.skip_space();
        let value = parser.value(0)?;
        parser.skip_space();
        (parser.position == text.len()).then_some(value)
    }

    /// The compact form: no spaces, integers as written, other numbers in the shortest form that reads back
    /// (positional from `1e-3` up to `1e7`, otherwise a mantissa and a signed power of ten, `1.0E10`).
    pub fn to_compact_string(&self) -> String {
        let mut json = String::new();
        self.write_compact(&mut json);
        json
    }

    fn write_compact(&self, json: &mut String) {
        match self {
            JsonValue::Null => json.push_str("null"),
            JsonValue::Bool(flag) => json.push_str(if *flag { "true" } else { "false" }),
            JsonValue::Integer(digits) => json.push_str(digits),
            JsonValue::Float(number) => json.push_str(&compact_float(*number)),
            JsonValue::String(text) => write_string(text, json),
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
                    write_string(key, json);
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

/// Quotes a string, escaping `"`, `\` and the controls (short escapes where they exist, else `\u00XX`).
fn write_string(text: &str, json: &mut String) {
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

/// A finite double in the form above.
fn compact_float(number: f64) -> String {
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
        if depth > MAX_DEPTH {
            return None;
        }
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
        self.position += 1;
        let mut members = Vec::new();
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
            self.skip_space();
            if self.peek()? != b':' {
                return None;
            }
            self.position += 1;
            self.skip_space();
            let value = self.value(depth + 1)?;
            put(&mut members, key, value);
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
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek()? {
            b'0' => self.position += 1,
            b'1'..=b'9' => {
                self.digits();
            }
            _ => return None,
        }
        let mut integer = true;
        if self.peek() == Some(b'.') {
            integer = false;
            self.position += 1;
            if self.digits() == 0 {
                return None;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integer = false;
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if self.digits() == 0 {
                return None;
            }
        }
        let literal = &self.text[start..self.position];
        if integer {
            let digits = if literal == "-0" { "0" } else { literal };
            return Some(JsonValue::Integer(digits.to_string()));
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
                            text.push(self.scalar(unit)?);
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
        let text = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
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
}

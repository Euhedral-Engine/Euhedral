//! JSON values and their serialization as the Qwen chat template's `tojson` filter writes them: Python's
//! `json.dumps(value, ensure_ascii=False)`.

use std::fmt::Write;

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
pub(super) fn python_float(value: f64) -> String {
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

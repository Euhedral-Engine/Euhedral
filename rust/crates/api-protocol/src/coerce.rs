//! How a request's typed scalar fields read the JSON value they are given, with the conversions a lenient JSON
//! reader makes: numbers from numeric text and from fractional numbers (truncated), booleans from integers and
//! `true`/`false` text, strings from numbers and booleans. Text that is empty, blank or `null` reads as an absent
//! value. Anything else that does not fit is an unparsable body.
//!
//! Each function returns `Ok(None)` for a null or absent-like value, `Ok(Some(_))` for a value read, and the
//! unparsable-body error for a value that cannot be read.

use crate::error::ApiError;
use crate::json::JsonValue;

/// True for text that reads as no value: empty, all of at most U+0020, or `null` once trimmed.
fn reads_as_absent(text: &str) -> bool {
    text.chars().all(|character| character <= ' ') || text.trim() == "null"
}

/// The text without the leading and trailing characters of at most U+0020.
fn trimmed(text: &str) -> &str {
    text.trim_matches(|character: char| character <= ' ')
}

fn unreadable<T>() -> Result<Option<T>, ApiError> {
    Err(ApiError::unparsable_body())
}

/// A 64-bit integer: integer literals in range; fractional numbers truncated toward zero when in range; text of an
/// optionally signed run of digits, trimmed.
pub fn long(value: &JsonValue) -> Result<Option<i64>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::Integer(digits) => digits.as_str().parse::<i64>().map(Some).or_else(|_| unreadable()),
        JsonValue::Float(number) => {
            // 2^63 is the first double above i64::MAX.
            if number.is_finite() && *number >= -9_223_372_036_854_775_808.0 && *number < 9_223_372_036_854_775_808.0 {
                Ok(Some(number.trunc() as i64))
            } else {
                unreadable()
            }
        }
        JsonValue::String(text) => {
            if reads_as_absent(text) {
                return Ok(None);
            }
            let digits = trimmed(text);
            let digits = digits.strip_prefix('+').unwrap_or(digits);
            if digits
                .strip_prefix('-')
                .unwrap_or(digits)
                .bytes()
                .all(|byte| byte.is_ascii_digit())
            {
                digits.parse::<i64>().map(Some).or_else(|_| unreadable())
            } else {
                unreadable()
            }
        }
        _ => unreadable(),
    }
}

/// A 32-bit integer, read as [`long`] and then required to be in range.
pub fn integer(value: &JsonValue) -> Result<Option<i32>, ApiError> {
    match long(value)? {
        None => Ok(None),
        Some(number) => i32::try_from(number).map(Some).or_else(|_| unreadable()),
    }
}

/// A double: any number (an integer literal of any size, or a float, infinite when out of range); text of a decimal
/// number, optionally with a trailing `d`, `D`, `f` or `F`, or one of `NaN`, `Infinity`, `INF` with a sign, trimmed.
pub fn double(value: &JsonValue) -> Result<Option<f64>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::Integer(digits) => digits.as_str().parse::<f64>().map(Some).or_else(|_| unreadable()),
        JsonValue::Float(number) => Ok(Some(*number)),
        JsonValue::String(text) => {
            if reads_as_absent(text) {
                return Ok(None);
            }
            let text = trimmed(text);
            match text {
                "Infinity" | "INF" | "+Infinity" | "+INF" => return Ok(Some(f64::INFINITY)),
                "-Infinity" | "-INF" => return Ok(Some(f64::NEG_INFINITY)),
                "NaN" => return Ok(Some(f64::NAN)),
                _ => {}
            }
            let number = text.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(text);
            let plain = number
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'));
            match number.parse::<f64>() {
                Ok(parsed) if plain => Ok(Some(parsed)),
                _ => unreadable(),
            }
        }
        _ => unreadable(),
    }
}

/// A boolean: `true` and `false`; an integer literal (zero is false, any other true); text that is `true`, `True`,
/// `TRUE`, `false`, `False` or `FALSE`, trimmed. Fractional numbers are not read.
pub fn boolean(value: &JsonValue) -> Result<Option<bool>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::Bool(flag) => Ok(Some(*flag)),
        JsonValue::Integer(digits) => Ok(Some(digits.as_str() != "0")),
        JsonValue::String(text) => {
            if reads_as_absent(text) {
                return Ok(None);
            }
            match trimmed(text) {
                "true" | "True" | "TRUE" => Ok(Some(true)),
                "false" | "False" | "FALSE" => Ok(Some(false)),
                _ => unreadable(),
            }
        }
        _ => unreadable(),
    }
}

/// A string: text as it is; a number or boolean as its text. Arrays and objects are not read.
pub fn string(value: &JsonValue) -> Result<Option<String>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::String(text) => Ok(Some(text.clone())),
        JsonValue::Bool(_) | JsonValue::Integer(_) | JsonValue::Float(_) => Ok(Some(value.to_compact_string())),
        _ => unreadable(),
    }
}

/// An array of values; a single value in place of an array is not read.
pub fn array(value: &JsonValue) -> Result<Option<Vec<JsonValue>>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::Array(items) => Ok(Some(items.clone())),
        _ => unreadable(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonInteger;
    use std::fmt::Debug;

    fn json(text: &str) -> JsonValue {
        JsonValue::parse(text).expect("valid json")
    }

    /// Each case: a JSON value, and `Some(expected)` when it reads, `None` when it is unparsable. An expected
    /// `Ok(None)` is `Some(None)`.
    fn table<T: PartialEq + Debug>(
        read: fn(&JsonValue) -> Result<Option<T>, ApiError>,
        cases: Vec<(&str, Option<Option<T>>)>,
    ) {
        for (input, expected) in cases {
            let outcome = read(&json(input)).ok();
            assert_eq!(outcome, expected, "input {input}");
        }
    }

    #[test]
    fn integers() {
        table(
            integer,
            vec![
                ("5", Some(Some(5))),
                ("-5", Some(Some(-5))),
                ("5.0", Some(Some(5))),
                ("5.7", Some(Some(5))),
                ("-5.7", Some(Some(-5))),
                ("\"5\"", Some(Some(5))),
                ("\" 5 \"", Some(Some(5))),
                ("\"+5\"", Some(Some(5))),
                ("\"-5\"", Some(Some(-5))),
                ("\"\"", Some(None)),
                ("\"  \"", Some(None)),
                ("\"null\"", Some(None)),
                ("null", Some(None)),
                ("2147483647", Some(Some(i32::MAX))),
                ("2147483648", None),
                ("\"2147483648\"", None),
                ("1e10", None),
                ("1e400", None),
                ("99999999999999999999", None),
                ("\"5.0\"", None),
                ("\"five\"", None),
                ("\"Null\"", None),
                ("true", None),
                ("[5]", None),
                ("{}", None),
            ],
        );
    }

    #[test]
    fn longs() {
        table(
            long,
            vec![
                ("123456789012", Some(Some(123_456_789_012))),
                ("9223372036854775807", Some(Some(i64::MAX))),
                ("9223372036854775808", None),
                ("1.5e3", Some(Some(1500))),
                ("1e19", None),
                ("\" 42 \"", Some(Some(42))),
                ("\"\"", Some(None)),
                ("false", None),
            ],
        );
    }

    #[test]
    fn doubles() {
        table(
            double,
            vec![
                ("0.7", Some(Some(0.7))),
                ("2", Some(Some(2.0))),
                ("\" 0.7 \"", Some(Some(0.7))),
                ("\"1e-3\"", Some(Some(0.001))),
                ("\"2.5f\"", Some(Some(2.5))),
                ("\"7\"", Some(Some(7.0))),
                ("1e400", Some(Some(f64::INFINITY))),
                ("\"Infinity\"", Some(Some(f64::INFINITY))),
                ("\"-INF\"", Some(Some(f64::NEG_INFINITY))),
                ("\"\"", Some(None)),
                ("\"null\"", Some(None)),
                ("null", Some(None)),
                ("\"inf\"", None),
                ("\"nan\"", None),
                ("\"0x10\"", None),
                ("\"abc\"", None),
                ("true", None),
                ("[1.0]", None),
            ],
        );
        let huge = JsonValue::Integer(JsonInteger::parse("123456789012345678901234567890").expect("digits"));
        assert_eq!(double(&huge).ok().flatten(), Some(1.2345678901234568e29));
        assert!(double(&json("\"NaN\"")).ok().flatten().is_some_and(f64::is_nan));
    }

    #[test]
    fn booleans() {
        table(
            boolean,
            vec![
                ("true", Some(Some(true))),
                ("false", Some(Some(false))),
                ("1", Some(Some(true))),
                ("0", Some(Some(false))),
                ("7", Some(Some(true))),
                ("-0", Some(Some(false))),
                ("\"true\"", Some(Some(true))),
                ("\"True\"", Some(Some(true))),
                ("\"TRUE\"", Some(Some(true))),
                ("\" false \"", Some(Some(false))),
                ("\"False\"", Some(Some(false))),
                ("\"FALSE\"", Some(Some(false))),
                ("\"\"", Some(None)),
                ("\"null\"", Some(None)),
                ("null", Some(None)),
                ("\"tRUE\"", None),
                ("\"yes\"", None),
                ("\"1\"", None),
                ("1.0", None),
                ("0.0", None),
                ("[true]", None),
            ],
        );
    }

    #[test]
    fn strings() {
        table(
            string,
            vec![
                ("\"m\"", Some(Some("m".to_owned()))),
                ("\"\"", Some(Some(String::new()))),
                ("\"null\"", Some(Some("null".to_owned()))),
                ("123", Some(Some("123".to_owned()))),
                ("true", Some(Some("true".to_owned()))),
                ("2.5", Some(Some("2.5".to_owned()))),
                ("null", Some(None)),
                ("[]", None),
                ("{}", None),
            ],
        );
    }

    #[test]
    fn arrays() {
        assert_eq!(
            array(&json("[1,\"a\"]")).ok().flatten().map(|items| items.len()),
            Some(2)
        );
        assert_eq!(array(&json("null")).ok(), Some(None));
        assert!(array(&json("1")).is_err());
        assert!(array(&json("\"a\"")).is_err());
        assert!(array(&json("{}")).is_err());
    }
}

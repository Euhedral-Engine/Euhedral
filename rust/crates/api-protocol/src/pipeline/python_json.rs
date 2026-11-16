//! JSON text exactly as Python's `json.dumps(value, ensure_ascii=False)` writes it, which is what the checkpoint
//! template's `tojson` filter produces: `", "` and `": "` separators, insertion key order, non-ASCII text
//! unescaped, and floats in Python `repr` form (`1e-05`, `1e+16`, `2.0`).

use std::fmt::Write;

use crate::pipeline::json_value::JsonValue;

/// Serializes a value. Non-finite numbers are spelled `Infinity`, `-Infinity` and `NaN`.
pub fn dumps(value: &JsonValue) -> String {
    let mut json = String::new();
    write_value(value, &mut json);
    json
}

fn write_value(value: &JsonValue, json: &mut String) {
    match value {
        JsonValue::Null => json.push_str("null"),
        JsonValue::Bool(flag) => json.push_str(if *flag { "true" } else { "false" }),
        JsonValue::Integer(digits) => json.push_str(digits),
        JsonValue::Float(number) => json.push_str(&python_float(*number)),
        JsonValue::String(text) => write_string(text, json),
        JsonValue::Array(items) => {
            json.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    json.push_str(", ");
                }
                write_value(item, json);
            }
            json.push(']');
        }
        JsonValue::Object(members) => {
            json.push('{');
            for (index, (key, member)) in members.iter().enumerate() {
                if index > 0 {
                    json.push_str(", ");
                }
                write_string(key, json);
                json.push_str(": ");
                write_value(member, json);
            }
            json.push('}');
        }
    }
}

/// Python escapes only `"`, `\` and the C0 controls (four lowercase hex digits except the short escapes).
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
                let _ = write!(json, "\\u{:04x}", u32::from(value));
            }
            value => json.push(value),
        }
    }
    json.push('"');
}

/// Python `float.__repr__`: the shortest round-tripping digits, positional when the decimal exponent is in
/// [-4, 16), otherwise `d.ddde±XX`.
pub fn python_float(number: f64) -> String {
    if number.is_nan() {
        return "NaN".to_string();
    }
    if number.is_infinite() {
        return if number > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if number == 0.0 {
        return if number.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let scientific = format!("{:e}", number.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|value| *value != '.').collect();
    let sign = if number < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exponent) {
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
        return format!("{sign}{body}");
    }
    let mantissa = if digits.len() == 1 {
        digits
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let direction = if exponent < 0 { '-' } else { '+' };
    format!("{sign}{mantissa}e{direction}{:02}", exponent.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_follow_python_repr() {
        let cases = [
            (1.0e-5, "1e-05"),
            (1.0e16, "1e+16"),
            (1.0e15, "1000000000000000.0"),
            (2.0, "2.0"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (-1.5e300, "-1.5e+300"),
            (0.1, "0.1"),
            (123456789.125, "123456789.125"),
            (f64::NAN, "NaN"),
            (f64::NEG_INFINITY, "-Infinity"),
            (-0.0, "-0.0"),
        ];
        for (number, expected) in cases {
            assert_eq!(python_float(number), expected);
        }
    }

    #[test]
    fn documents_use_python_separators_and_keep_non_ascii_text() {
        let value = JsonValue::parse(r#"{"a": [1, 2.50, "é\u0001\n"], "b": {"c": null, "d": true}, "e": {}, "f": []}"#)
            .unwrap();
        assert_eq!(
            dumps(&value),
            "{\"a\": [1, 2.5, \"é\\u0001\\n\"], \"b\": {\"c\": null, \"d\": true}, \"e\": {}, \"f\": []}"
        );
    }
}

//! The text of the search's JSON files, in the layout the files have always had: two-space indentation for objects,
//! `"key" : value`, arrays on one line, `{ }` and `[ ]` when empty, and one newline at the end. Existing files and
//! new ones are then byte for byte the same.

use std::fmt::Write;

/// A JSON value to render.
pub(crate) enum Node {
    Object(Vec<(String, Node)>),
    Array(Vec<Node>),
    Text(String),
    Int(i64),
    Float(f64),
}

/// `node` as the text of a file.
pub(crate) fn render(node: &Node) -> String {
    let mut out = String::new();
    write_node(&mut out, node, 0);
    out.push('\n');
    out
}

fn write_node(out: &mut String, node: &Node, depth: usize) {
    match node {
        Node::Object(fields) if fields.is_empty() => out.push_str("{ }"),
        Node::Object(fields) => {
            out.push_str("{\n");
            for (index, (name, value)) in fields.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                indent(out, depth + 1);
                write_text(out, name);
                out.push_str(" : ");
                write_node(out, value, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push('}');
        }
        Node::Array(items) if items.is_empty() => out.push_str("[ ]"),
        Node::Array(items) => {
            out.push_str("[ ");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_node(out, item, depth);
            }
            out.push_str(" ]");
        }
        Node::Text(text) => write_text(out, text),
        Node::Int(value) => {
            let _ = write!(out, "{value}");
        }
        Node::Float(value) => out.push_str(&double_text(*value)),
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn write_text(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04X}", control as u32);
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// A double as the file format writes it: the shortest digits that read back the same, in plain notation from
/// 1e-3 up to (not including) 1e7 and in `d.dddE±n` notation outside it, always with a digit after the point. A value
/// that is not finite is written as a string, as the file's writer does.
pub(crate) fn double_text(value: f64) -> String {
    if value.is_nan() {
        return "\"NaN\"".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let mut scientific = format!("{:e}", value.abs());
    if value.abs() < f64::MIN_POSITIVE && !scientific.contains('.') {
        // A subnormal has so few digits of precision that one shortest digit is not the file's: it writes at least two,
        // the nearest to the value.
        scientific = format!("{:.1e}", value.abs());
    }
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value < 0.0 { "-" } else { "" };
    if (-3..7).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            let zeros = "0".repeat((-point) as usize);
            format!("{sign}0.{zeros}{digits}")
        } else if digits.len() as i32 <= point {
            let zeros = "0".repeat(point as usize - digits.len());
            format!("{sign}{digits}{zeros}.0")
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{sign}{whole}.{fraction}")
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { "0" } else { rest };
        format!("{sign}{first}.{rest}E{exponent}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_are_written_as_the_files_have_always_had_them() {
        let cases = [
            (0.0, "0.0"),
            (1.0, "1.0"),
            (51.2, "51.2"),
            (0.001, "0.001"),
            (f64::from_bits(1), "4.9E-324"),
            (f64::from_bits(2), "9.9E-324"),
            (f64::from_bits(4), "2.0E-323"),
            (0.0001, "1.0E-4"),
            (0.00012345, "1.2345E-4"),
            (1234567.0, "1234567.0"),
            (12345678.0, "1.2345678E7"),
            (1e7, "1.0E7"),
            (100.0, "100.0"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (9.999999e6, "9999999.0"),
        ];
        for (value, text) in cases {
            assert_eq!(double_text(value), text, "{value:e}");
        }
    }

    #[test]
    fn empty_containers_and_nesting_follow_the_layout() {
        let node = Node::Object(vec![
            ("a".to_string(), Node::Object(Vec::new())),
            ("b".to_string(), Node::Array(Vec::new())),
            (
                "c".to_string(),
                Node::Array(vec![
                    Node::Object(vec![("x".to_string(), Node::Int(1))]),
                    Node::Object(vec![("x".to_string(), Node::Text("q\"".to_string()))]),
                ]),
            ),
        ]);
        assert_eq!(
            render(&node),
            "{\n  \"a\" : { },\n  \"b\" : [ ],\n  \"c\" : [ {\n    \"x\" : 1\n  }, {\n    \"x\" : \"q\\\"\"\n  } ]\n}\n"
        );
    }
}

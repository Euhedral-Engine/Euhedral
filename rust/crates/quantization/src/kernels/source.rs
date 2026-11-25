//! The kernel sources the library compiles and the expansion of their shipped includes.
//!
//! The sources are embedded at build time from the files the repository keeps once, so the text compiled here is the
//! text every other build of the same files compiles.

use std::collections::HashSet;

/// `trellis/codebook.cuh`, the one header a quantization source includes from the repository.
const TRELLIS_CODEBOOK: &str = include_str!("../../../../../native/src/trellis/codebook.cuh");

/// The shipped headers: the include path (as the source writes it) and the text.
const HEADERS: [(&str, &str); 1] = [("trellis/codebook.cuh", TRELLIS_CODEBOOK)];

/// The text of the shipped header `path`, when there is one.
pub(super) fn header(path: &str) -> Option<&'static str> {
    HEADERS.iter().find(|(name, _)| *name == path).map(|(_, text)| *text)
}

/// Replaces every line that is only `#include "PATH"` (blanks allowed around it) by the text of the header `lookup`
/// finds for `PATH`, recursively and once per path; a later include of the same path becomes an empty line. A line
/// whose path `lookup` does not know is left for the compiler to resolve. The line break after the directive is kept.
pub fn expand_includes<'a>(source: &str, lookup: &impl Fn(&str) -> Option<&'a str>) -> String {
    let mut expanded = HashSet::new();
    expand(source, lookup, &mut expanded)
}

fn expand<'a>(source: &str, lookup: &impl Fn(&str) -> Option<&'a str>, expanded: &mut HashSet<String>) -> String {
    let mut result = String::with_capacity(source.len());
    for line in source.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let body = body.strip_suffix('\r').unwrap_or(body);
        let ending = &line[body.len()..];
        match include_path(body) {
            Some(path) => match lookup(path) {
                Some(text) => {
                    if expanded.insert(path.to_owned()) {
                        result.push_str(&expand(text, lookup, expanded));
                    }
                    result.push_str(ending);
                }
                None => result.push_str(line),
            },
            None => result.push_str(line),
        }
    }
    result
}

/// The path of a line that is only `#include "PATH"`, with blanks (spaces and tabs) around the directive.
fn include_path(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches([' ', '\t']).strip_prefix("#include")?;
    let after = rest.trim_start_matches([' ', '\t']);
    if after.len() == rest.len() {
        return None;
    }
    let quoted = after.strip_prefix('"')?;
    let end = quoted.find('"')?;
    if end == 0 || !quoted[end + 1..].trim_start_matches([' ', '\t']).is_empty() {
        return None;
    }
    Some(&quoted[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(path: &str) -> Option<&'static str> {
        match path {
            "a.cuh" => Some("int a;\n#include \"b.cuh\"\n"),
            "b.cuh" => Some("int b;\n"),
            _ => None,
        }
    }

    #[test]
    fn a_shipped_include_is_replaced_by_its_text() {
        assert_eq!(
            expand_includes("x\n  #include \"b.cuh\"  \ny\n", &headers),
            "x\nint b;\n\ny\n"
        );
    }

    #[test]
    fn includes_expand_recursively_and_once_per_path() {
        let text = expand_includes("#include \"a.cuh\"\n#include \"b.cuh\"\nend", &headers);
        assert_eq!(text, "int a;\nint b;\n\n\n\nend");
    }

    #[test]
    fn other_includes_are_left_to_the_compiler() {
        let source = "#include <cuda_bf16.h>\n#include \"unknown.h\"\n// #include \"b.cuh\"\n#include \"b.cuh\" // x\n";
        assert_eq!(expand_includes(source, &headers), source);
    }

    #[test]
    fn a_carriage_return_line_ending_is_kept() {
        assert_eq!(expand_includes("#include \"b.cuh\"\r\nz", &headers), "int b;\n\r\nz");
    }

    #[test]
    fn the_directive_needs_a_blank_and_a_path() {
        assert_eq!(include_path("#include\"b.cuh\""), None);
        assert_eq!(include_path("#include \"\""), None);
        assert_eq!(include_path("\t#include\t\"b.cuh\" "), Some("b.cuh"));
    }
}

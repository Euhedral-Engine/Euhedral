//! The two notions of whitespace the stages use.

/// Python's `str.isspace`: the Unicode white space, the four information separators and the separator spaces.
pub(crate) fn is_python_space(value: char) -> bool {
    value.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&value)
}

/// Whitespace between a call's tags and inside its block: the separator spaces except the non-breaking ones, the
/// line and paragraph separators, and the ASCII and information-separator controls.
pub(crate) fn is_block_space(value: char) -> bool {
    match value {
        '\t'..='\r' | '\u{1c}'..='\u{1f}' | ' ' | '\u{1680}' | '\u{2028}' | '\u{2029}' | '\u{205f}' | '\u{3000}' => {
            true
        }
        '\u{2000}'..='\u{200a}' => value != '\u{2007}',
        _ => false,
    }
}

/// The start of the python-space run that ends at byte `end` of `text`.
pub(crate) fn trailing_space_start(text: &str, end: usize) -> usize {
    let mut start = end;
    for (index, value) in text[..end].char_indices().rev() {
        if !is_python_space(value) {
            break;
        }
        start = index;
    }
    start
}

/// The byte length of the longest suffix of `text` that is a proper prefix of the ASCII `tag`.
pub(crate) fn held_tag_prefix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&length| text.as_bytes().ends_with(&tag.as_bytes()[..length]))
        .unwrap_or(0)
}

/// The start of `text` after its leading python space.
pub(crate) fn strip_leading_space(text: &str) -> &str {
    text.trim_start_matches(is_python_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_space_includes_the_information_separators_and_no_break_space() {
        for value in ['\u{1c}', '\u{85}', '\u{a0}', '\u{2007}', '\u{2028}', ' ', '\n'] {
            assert!(is_python_space(value), "{value:?}");
        }
        assert!(!is_python_space('a'));
    }

    #[test]
    fn block_space_leaves_out_the_no_break_spaces_and_next_line() {
        for value in ['\u{a0}', '\u{2007}', '\u{202f}', '\u{85}'] {
            assert!(!is_block_space(value), "{value:?}");
        }
        for value in ['\u{1c}', '\u{2003}', '\u{3000}', '\n'] {
            assert!(is_block_space(value), "{value:?}");
        }
    }

    #[test]
    fn a_held_prefix_is_the_longest_proper_one() {
        assert_eq!(held_tag_prefix("a</thi", "</think>"), 5);
        assert_eq!(held_tag_prefix("a</think", "</think>"), 7);
        assert_eq!(held_tag_prefix("a</think>", "</think>"), 0);
        assert_eq!(held_tag_prefix("é", "</think>"), 0);
    }
}

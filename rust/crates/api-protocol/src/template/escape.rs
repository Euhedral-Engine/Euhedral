//! Text handling the chat template needs: Python's `str.strip` and the protection of control-token spellings.

/// Whether Python's `str.strip` removes the character: Unicode white space plus the separators U+001C to U+001F.
pub(super) fn is_python_space(value: char) -> bool {
    value.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&value)
}

/// Python's `str.strip()`.
pub(super) fn trim(text: &str) -> &str {
    text.trim_matches(is_python_space)
}

/// Replaces the `<` that opens a spelling of a checkpoint control token (`<|...|>`, `<think>`, `</think>`,
/// `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<tts_...>`) with the six characters
/// `<`, so untrusted text cannot be encoded as a control token. Other angle brackets are unchanged.
pub(super) fn escape_control_tokens(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'<' && opens_control_token(&bytes[index + 1..]) {
            out.push_str(&text[copied..index]);
            out.push_str("\\u003c");
            copied = index + 1;
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Whether `rest`, the bytes after a `<`, continue as a control-token spelling.
fn opens_control_token(rest: &[u8]) -> bool {
    if rest.first() == Some(&b'|') {
        // `|` then one or more characters that are not `<`, `>` or white space, the last of them `|`, then `>`.
        let run = rest[1..]
            .iter()
            .take_while(|byte| !matches!(byte, b'<' | b'>' | b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r'))
            .count();
        if run >= 2 && rest[run] == b'|' && rest.get(run + 1) == Some(&b'>') {
            return true;
        }
    }
    let name = rest.strip_prefix(b"/").unwrap_or(rest);
    ["tool_call>", "tool_response>", "think>"]
        .iter()
        .any(|spelling| name.starts_with(spelling.as_bytes()))
        || name.strip_prefix(b"tts_").is_some_and(|tail| {
            let run = tail
                .iter()
                .take_while(|byte| byte.is_ascii_lowercase() || **byte == b'_')
                .count();
            run >= 1 && tail.get(run) == Some(&b'>')
        })
}

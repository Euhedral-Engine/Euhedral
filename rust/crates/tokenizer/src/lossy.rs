//! Lossy UTF-8 decoding with the replacement rules of Java's `new String(bytes, UTF_8)`.
//!
//! Rust's [`String::from_utf8_lossy`] replaces the maximal invalid prefix of a sequence, and so does the JDK, with one
//! difference: the JDK treats a surrogate encoding (lead byte `ED` followed by `A0..=BF`) as one malformed unit,
//! replacing the two bytes `ED xx`, or all three of `ED xx yy` when `yy` is a continuation byte, with a single U+FFFD.
//! Streamed text must match the Java engine character for character, so every lossy decode goes through here.

use std::str;

/// The U+FFFD the JDK emits for malformed input.
const REPLACEMENT: char = '\u{fffd}';

/// `bytes` as text, each malformed unit replaced by one U+FFFD, exactly as the JDK's UTF-8 decoder does.
pub(crate) fn decode_lossy(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len());
    let mut rest = bytes;
    while !rest.is_empty() {
        match str::from_utf8(rest) {
            Ok(valid) => {
                output.push_str(valid);
                break;
            }
            Err(error) => {
                let (valid, bad) = rest.split_at(error.valid_up_to());
                // The prefix is valid by the error's contract; the fallback keeps the library free of unwraps.
                output.push_str(str::from_utf8(valid).unwrap_or_default());
                output.push(REPLACEMENT);
                let consumed = match bad {
                    [0xed, 0xa0..=0xbf, 0x80..=0xbf, ..] => 3,
                    [0xed, 0xa0..=0xbf, ..] => 2,
                    _ => error.error_len().unwrap_or(bad.len()),
                };
                rest = &bad[consumed..];
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected text for each input, as printed by the JDK (`new String(bytes, UTF_8)`, Java 25) in jshell.
    const JDK: &[(&[u8], &str)] = &[
        (&[0xed, 0xa0, 0x80], "\u{fffd}"),
        (&[0xed, 0xa0, 0x41], "\u{fffd}A"),
        (&[0xed, 0xa0], "\u{fffd}"),
        (&[0xed, 0xbf, 0xbf], "\u{fffd}"),
        (&[0xed, 0xa0, 0x80, 0x80], "\u{fffd}\u{fffd}"),
        (&[0xed, 0x9f, 0xbf], "\u{d7ff}"),
        (&[0xc0, 0x80], "\u{fffd}\u{fffd}"),
        (&[0xc1, 0xbf], "\u{fffd}\u{fffd}"),
        (&[0xe0, 0x80, 0x80], "\u{fffd}\u{fffd}\u{fffd}"),
        (&[0xe0, 0x9f, 0x80], "\u{fffd}\u{fffd}\u{fffd}"),
        (&[0xf0, 0x80, 0x80, 0x80], "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"),
        (&[0xf0, 0x8f, 0xbf, 0xbf], "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"),
        (&[0xf4, 0x90, 0x80, 0x80], "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"),
        (&[0xf5, 0x80], "\u{fffd}\u{fffd}"),
        (
            &[0xf8, 0x80, 0x80, 0x80, 0x80],
            "\u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}",
        ),
        (&[0x80], "\u{fffd}"),
        (&[0xff], "\u{fffd}"),
        (&[0xe2, 0x82], "\u{fffd}"),
        (&[0xe2, 0x82, 0x41], "\u{fffd}A"),
        (&[0xf0, 0x9f, 0x98], "\u{fffd}"),
        (&[0xf0, 0x9f, 0x98, 0x41], "\u{fffd}A"),
        (&[0xf0, 0x9f, 0x41], "\u{fffd}A"),
        (&[0xf0, 0x9f, 0x98, 0x80], "\u{1f600}"),
        (&[], ""),
    ];

    #[test]
    fn matches_the_jdk_on_the_malformed_classes() {
        for (bytes, expected) in JDK {
            assert_eq!(decode_lossy(bytes), *expected, "{bytes:02x?}");
        }
    }

    #[test]
    fn valid_text_passes_through() {
        assert_eq!(decode_lossy("héllo €😀".as_bytes()), "héllo €😀");
    }
}

/// The byte-level BPE alphabet: every byte has one visible character, and the vocabulary spells tokens in them.
///
/// Bytes that print as themselves (`!` to `~`, `¡` to `¬`, `®` to `ÿ`) map to their own code point; the other 68
/// bytes map, in order, to the code points from 256 up.
const FIRST_EXTRA: u32 = 256;
const BYTES: usize = 256;

fn is_visible(byte: usize) -> bool {
    matches!(byte, 33..=126 | 161..=172 | 174..=255)
}

/// The byte a vocabulary character stands for.
#[derive(Debug, Clone)]
pub(crate) struct ByteLevel {
    /// Indexed by code point; `None` for a character outside the alphabet.
    byte_by_code_point: Vec<Option<u8>>,
}

impl ByteLevel {
    pub(crate) fn new() -> ByteLevel {
        let extras = (0..BYTES).filter(|byte| !is_visible(*byte)).count();
        let mut byte_by_code_point = vec![None; FIRST_EXTRA as usize + extras];
        let mut next_extra = FIRST_EXTRA as usize;
        for byte in 0..BYTES {
            let code_point = if is_visible(byte) {
                byte
            } else {
                next_extra += 1;
                next_extra - 1
            };
            // Every index is below 256 + 68, and every byte is below 256.
            byte_by_code_point[code_point] = u8::try_from(byte).ok();
        }
        ByteLevel { byte_by_code_point }
    }

    pub(crate) fn byte(&self, character: char) -> Option<u8> {
        self.byte_by_code_point
            .get(usize::try_from(u32::from(character)).ok()?)
            .copied()
            .flatten()
    }

    /// The bytes of a vocabulary token, or `None` when a character of it is outside the alphabet.
    pub(crate) fn token_bytes(&self, token: &str) -> Option<Vec<u8>> {
        token.chars().map(|character| self.byte(character)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_alphabet_matches_the_gpt2_table() {
        let alphabet = ByteLevel::new();
        assert_eq!(alphabet.byte('!'), Some(33));
        assert_eq!(alphabet.byte('~'), Some(126));
        assert_eq!(alphabet.byte('¡'), Some(161));
        assert_eq!(alphabet.byte('ÿ'), Some(255));
        // The first non-visible byte, 0, is the first extra; space (32) is the 33rd: `Ġ`, U+0120.
        assert_eq!(alphabet.byte('\u{100}'), Some(0));
        assert_eq!(alphabet.byte('Ġ'), Some(32));
        assert_eq!(alphabet.byte('Ċ'), Some(10));
        assert_eq!(alphabet.byte('\u{143}'), Some(173));
        assert_eq!(alphabet.byte('\u{144}'), None);
        assert_eq!(alphabet.byte('a'), Some(97));
        assert_eq!(alphabet.byte('€'), None);
    }

    #[test]
    fn every_byte_has_one_character() {
        let alphabet = ByteLevel::new();
        let mut seen = [false; BYTES];
        for code_point in 0..alphabet.byte_by_code_point.len() as u32 {
            if let Some(byte) = char::from_u32(code_point).and_then(|character| alphabet.byte(character)) {
                assert!(!seen[usize::from(byte)]);
                seen[usize::from(byte)] = true;
            }
        }
        assert!(seen.iter().all(|present| *present));
    }

    #[test]
    fn token_bytes_spells_multibyte_tokens() {
        let alphabet = ByteLevel::new();
        // `ðŁĺĢ` is the byte-level spelling of the emoji U+1F600.
        assert_eq!(alphabet.token_bytes("ðŁĺĢ"), Some("😀".as_bytes().to_vec()));
        assert_eq!(alphabet.token_bytes("a€"), None);
    }
}

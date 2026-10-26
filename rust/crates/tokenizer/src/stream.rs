use crate::error::TokenizerError;

/// What the vocabulary holds for one token ID.
#[derive(Debug, Clone)]
pub(crate) enum Piece {
    /// The ID is unused.
    Absent,
    /// An ordinary token: the bytes it stands for.
    Bytes(Box<[u8]>),
    /// An added token (special or not): decoded as its own spelling.
    Control(Box<str>),
}

/// Turns generated token IDs into text, holding back an incomplete UTF-8 suffix until the next token completes it.
///
/// Created by [`Tokenizer::decode_stream`](crate::Tokenizer::decode_stream). Everything a token adds is emitted as
/// soon as it is complete text; a control token first flushes the held bytes, so an invalid suffix before it becomes
/// U+FFFD rather than waiting. [`TokenStream::finish`] flushes the rest the same way.
#[derive(Debug)]
pub struct TokenStream<'a> {
    pieces: &'a [Piece],
    pending: Vec<u8>,
}

impl<'a> TokenStream<'a> {
    pub(crate) fn new(pieces: &'a [Piece]) -> TokenStream<'a> {
        TokenStream {
            pieces,
            pending: Vec::new(),
        }
    }

    /// Appends one token and returns the complete text it finishes (possibly empty).
    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        let piece = usize::try_from(id)
            .ok()
            .and_then(|index| self.pieces.get(index))
            .ok_or(TokenizerError::UnknownToken(id))?;
        match piece {
            Piece::Absent => Err(TokenizerError::UnknownToken(id)),
            Piece::Control(text) => {
                let mut output = self.flush();
                output.push_str(text);
                Ok(output)
            }
            Piece::Bytes(bytes) => {
                self.pending.extend_from_slice(bytes);
                Ok(self.emit_complete())
            }
        }
    }

    /// Appends a chunk of tokens and returns the text they complete.
    pub fn push_all(&mut self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut output = String::new();
        for id in ids {
            output.push_str(&self.push(*id)?);
        }
        Ok(output)
    }

    /// Ends the stream: a truncated UTF-8 suffix becomes U+FFFD.
    pub fn finish(mut self) -> String {
        self.flush()
    }

    fn flush(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }

    fn emit_complete(&mut self) -> String {
        let safe = complete_utf8_length(&self.pending);
        if safe == 0 {
            return String::new();
        }
        let text = String::from_utf8_lossy(&self.pending[..safe]).into_owned();
        self.pending.drain(..safe);
        text
    }
}

/// The length of the longest prefix of `bytes` that no later byte can change: whole characters, and bytes that can
/// never start or continue a character (those decode to U+FFFD at once). A suffix that may still become a character
/// is held back.
fn complete_utf8_length(bytes: &[u8]) -> usize {
    let mut offset = 0;
    while offset < bytes.len() {
        let expected = match bytes[offset] {
            0x00..=0x7f => 1,
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => 1,
        };
        if expected == 1 {
            offset += 1;
            continue;
        }
        let available = bytes.len() - offset;
        let mut continuations = 0;
        while continuations < expected - 1 && continuations + 1 < available {
            if !(0x80..=0xbf).contains(&bytes[offset + continuations + 1]) {
                break;
            }
            continuations += 1;
        }
        if continuations < expected - 1 {
            if continuations == available - 1 {
                return offset;
            }
            offset += 1;
            continue;
        }
        offset += expected;
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces() -> Vec<Piece> {
        let bytes = |bytes: &[u8]| Piece::Bytes(bytes.into());
        vec![
            bytes(b"A"),
            bytes(&[0xf0]),
            bytes(&[0x9f]),
            bytes(&[0x98]),
            bytes(&[0x80]),
            Piece::Control("<|im_end|>".into()),
            Piece::Absent,
            bytes(&[0xe2, 0x82]),
            bytes(&[0xac]),
        ]
    }

    fn stream_all(pieces: &[Piece], ids: &[u32]) -> String {
        let mut stream = TokenStream::new(pieces);
        let mut text = String::new();
        for id in ids {
            text.push_str(&stream.push(*id).unwrap());
        }
        text.push_str(&stream.finish());
        text
    }

    #[test]
    fn a_character_split_across_tokens_is_emitted_when_it_completes() {
        let pieces = pieces();
        let mut stream = TokenStream::new(&pieces);
        assert_eq!(stream.push(1).unwrap(), "");
        assert_eq!(stream.push(2).unwrap(), "");
        assert_eq!(stream.push(3).unwrap(), "");
        assert_eq!(stream.push(4).unwrap(), "😀");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn a_chunk_and_single_tokens_decode_alike() {
        let pieces = pieces();
        let mut chunked = TokenStream::new(&pieces);
        assert_eq!(chunked.push_all(&[0, 1, 2]).unwrap(), "A");
        assert_eq!(chunked.push_all(&[3, 4, 7, 8]).unwrap(), "😀€");
        assert_eq!(chunked.finish(), "");
        assert_eq!(stream_all(&pieces, &[0, 1, 2, 3, 4, 7, 8]), "A😀€");
    }

    #[test]
    fn a_truncated_suffix_becomes_a_replacement_character_at_the_end_or_before_a_control_token() {
        let pieces = pieces();
        assert_eq!(stream_all(&pieces, &[1]), "\u{fffd}");
        assert_eq!(stream_all(&pieces, &[1, 0]), "\u{fffd}A");
        assert_eq!(stream_all(&pieces, &[1, 5]), "\u{fffd}<|im_end|>");
        let mut stream = TokenStream::new(&pieces);
        assert_eq!(stream.push(1).unwrap(), "");
        assert_eq!(stream.push(5).unwrap(), "\u{fffd}<|im_end|>");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn a_continuation_byte_with_no_start_is_replaced_at_once() {
        let pieces = pieces();
        let mut stream = TokenStream::new(&pieces);
        assert_eq!(stream.push(4).unwrap(), "\u{fffd}");
        assert_eq!(stream.push(0).unwrap(), "A");
    }

    #[test]
    fn unknown_and_unused_ids_are_refused() {
        let pieces = pieces();
        let mut stream = TokenStream::new(&pieces);
        assert!(matches!(stream.push(6), Err(TokenizerError::UnknownToken(6))));
        assert!(matches!(stream.push(99), Err(TokenizerError::UnknownToken(99))));
    }

    #[test]
    fn complete_length_holds_back_only_a_suffix_that_can_still_complete() {
        assert_eq!(complete_utf8_length(b"ab"), 2);
        assert_eq!(complete_utf8_length(&[b'a', 0xe2, 0x82]), 1);
        assert_eq!(complete_utf8_length(&[0xe2, 0x82, 0xac]), 3);
        // A lead byte followed by a non-continuation is complete garbage, not a pending character.
        assert_eq!(complete_utf8_length(&[0xe2, b'a']), 2);
        // Bytes that can never start a character are not held back.
        assert_eq!(complete_utf8_length(&[0xc0]), 1);
        assert_eq!(complete_utf8_length(&[0xff, 0xf0]), 1);
    }
}

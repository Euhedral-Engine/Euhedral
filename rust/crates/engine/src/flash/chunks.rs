//! How a prompt is cut into the quanta of its prefill (the prompt loop): full chunks of the planner's
//! prefill chunk first, the remainder last. Only the last chunk scores its final row; the others run for their state
//! alone.

use super::FlashError;

const OP: &str = "prompt chunks";

/// One chunk of a prompt: where it starts in the prompt, how many rows it has and whether it scores its last row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptChunk {
    pub start: usize,
    pub rows: usize,
    pub scores: bool,
}

/// The chunks of a prompt of `tokens` tokens for chunks of at most `chunk` rows, in order. A prompt without a token,
/// or a chunk of no rows, is refused with the value.
pub fn prompt_chunks(tokens: usize, chunk: usize) -> Result<Vec<PromptChunk>, FlashError> {
    if tokens == 0 {
        return Err(FlashError::refused(OP, "a prompt of 0 tokens".to_owned()));
    }
    if chunk == 0 {
        return Err(FlashError::refused(OP, "a chunk of 0 rows".to_owned()));
    }
    let count = tokens.div_ceil(chunk);
    Ok((0..count)
        .map(|index| {
            let start = index * chunk;
            PromptChunk {
                start,
                rows: chunk.min(tokens - start),
                scores: index + 1 == count,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_chunks_come_first_and_the_remainder_last() {
        let chunks = prompt_chunks(1100, 512).unwrap();
        let shape: Vec<_> = chunks.iter().map(|c| (c.start, c.rows)).collect();
        assert_eq!(shape, vec![(0, 512), (512, 512), (1024, 76)]);
    }

    #[test]
    fn only_the_last_chunk_scores() {
        let chunks = prompt_chunks(1100, 512).unwrap();
        let scores: Vec<_> = chunks.iter().map(|c| c.scores).collect();
        assert_eq!(scores, vec![false, false, true]);
    }

    #[test]
    fn a_prompt_of_whole_chunks_has_no_remainder_and_one_row_is_one_chunk() {
        assert_eq!(prompt_chunks(1024, 512).unwrap().len(), 2);
        assert_eq!(
            prompt_chunks(1, 512).unwrap(),
            vec![PromptChunk {
                start: 0,
                rows: 1,
                scores: true
            }]
        );
    }

    #[test]
    fn the_chunks_cover_the_prompt_exactly_once() {
        let chunks = prompt_chunks(777, 100).unwrap();
        assert_eq!(chunks.iter().map(|c| c.rows).sum::<usize>(), 777);
        assert!(chunks.windows(2).all(|w| w[0].start + w[0].rows == w[1].start));
    }

    #[test]
    fn an_empty_prompt_and_an_empty_chunk_are_refused() {
        assert!(prompt_chunks(0, 512).is_err());
        assert!(prompt_chunks(5, 0).is_err());
    }
}

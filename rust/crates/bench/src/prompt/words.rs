//! The raw-text word prompt, generator `euhedral-words-v1` (Java `PromptMaterial`).
//!
//! The text is a space-separated sequence of words drawn from [`WORDS`] by `SplittableRandom(seed)`. The longest word
//! prefix whose token count does not exceed the target is kept; single punctuation suffixes are then tried to reach the
//! target exactly. The count is the one the session itself takes (model special tokens included), so it may stay below
//! the target when no suffix fits. Prompts are not chat-templated.

use euhedral_inference_tokenizer::TokenizerError;

use super::splittable_random::SplittableRandom;
use super::{Built, PromptError};

pub const GENERATOR: &str = "euhedral-words-v1";

/// The seed the Java benchmark uses unless its configuration names another.
pub const DEFAULT_SEED: u64 = 20_260_925;

pub const WORDS: [&str; 64] = [
    "the",
    "river",
    "carries",
    "stone",
    "across",
    "a",
    "quiet",
    "valley",
    "where",
    "engineers",
    "measure",
    "light",
    "and",
    "time",
    "while",
    "clocks",
    "record",
    "every",
    "small",
    "change",
    "in",
    "pressure",
    "under",
    "bright",
    "northern",
    "sky",
    "people",
    "build",
    "bridges",
    "from",
    "steel",
    "glass",
    "wood",
    "maps",
    "show",
    "old",
    "roads",
    "that",
    "follow",
    "water",
    "toward",
    "distant",
    "cities",
    "markets",
    "sell",
    "bread",
    "salt",
    "copper",
    "tools",
    "children",
    "learn",
    "numbers",
    "letters",
    "songs",
    "winter",
    "brings",
    "snow",
    "summer",
    "harvest",
    "machines",
    "turn",
    "slowly",
    "through",
    "night",
];

const SUFFIXES: [&str; 8] = [".", ",", ";", "!", "?", ":", " .", " ,"];

/// Builds the prompt text for `target_tokens`; `count` is the number of tokens a text encodes to.
pub fn build<C>(target_tokens: usize, seed: u64, mut count: C) -> Result<Built, PromptError>
where
    C: FnMut(&str) -> Result<usize, TokenizerError>,
{
    if target_tokens == 0 {
        return Err(PromptError::ZeroTarget);
    }
    let word_count = target_tokens * 2 + 8;
    let mut random = SplittableRandom::new(seed);
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        // The bound is the list's length, which is positive and fits an int.
        let index = random.next_int_bounded(WORDS.len() as i32).unwrap_or(0);
        words.push(WORDS[index as usize]);
    }

    let mut low = 1;
    let mut high = words.len();
    let shortest = count(&words[..low].join(" "))?;
    if shortest > target_tokens {
        return Err(PromptError::TargetTooSmall {
            target: target_tokens,
            shortest,
        });
    }
    while low < high {
        let middle = (low + high).div_ceil(2);
        if count(&words[..middle].join(" "))? <= target_tokens {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let mut text = words[..low].join(" ");
    let mut tokens = count(&text)?;
    while tokens < target_tokens {
        let mut extended = None;
        for suffix in SUFFIXES {
            let candidate = format!("{text}{suffix}");
            let candidate_tokens = count(&candidate)?;
            if candidate_tokens > tokens && candidate_tokens <= target_tokens {
                extended = Some((candidate, candidate_tokens));
                break;
            }
        }
        let Some((candidate, candidate_tokens)) = extended else {
            break;
        };
        text = candidate;
        tokens = candidate_tokens;
    }
    Ok(Built { text, tokens })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::sha256_hex;

    /// One token per word plus one for a trailing period.
    fn count(text: &str) -> Result<usize, TokenizerError> {
        Ok(text.split(' ').count() + usize::from(text.ends_with('.')))
    }

    #[test]
    fn reaches_the_target_deterministically() {
        let first = build(100, 5, count).unwrap();
        let second = build(100, 5, count).unwrap();
        assert_eq!(first.text, second.text);
        assert_eq!(first.tokens, 100);
        assert_ne!(sha256_hex(&first.text), sha256_hex(&build(100, 6, count).unwrap().text));
    }

    #[test]
    fn words_come_from_the_list_in_draw_order() {
        let built = build(8, DEFAULT_SEED, count).unwrap();
        // The first 16 draws for this seed are the indices in the splittable_random test.
        let expected: Vec<&str> = [36, 49, 6, 25, 30, 62, 47, 44]
            .iter()
            .map(|index| WORDS[*index])
            .collect();
        assert_eq!(built.text.split(' ').collect::<Vec<_>>(), expected);
    }

    #[test]
    fn reports_the_actual_count_when_the_target_cannot_be_hit_exactly() {
        // Every word costs two tokens, so an odd target is unreachable and must not be claimed.
        let built = build(7, 1, |text| Ok(text.split(' ').count() * 2)).unwrap();
        assert_eq!(built.tokens, 6);
        assert!(matches!(
            build(1, 1, |_| Ok(2)),
            Err(PromptError::TargetTooSmall { target: 1, shortest: 2 })
        ));
    }

    #[test]
    fn a_punctuation_suffix_fills_the_last_token() {
        // Two tokens per word, and a trailing comma adds one: a target of 7 takes three words and the comma, since the
        // period before it adds nothing.
        let built = build(7, 1, |text| {
            Ok(text.split(' ').count() * 2 + usize::from(text.ends_with(',')))
        })
        .unwrap();
        assert_eq!(built.tokens, 7);
        assert!(built.text.ends_with(','));
        assert_eq!(built.text.split(' ').count(), 3);
    }

    #[test]
    fn a_zero_target_is_refused() {
        assert!(matches!(build(0, 1, count), Err(PromptError::ZeroTarget)));
    }
}

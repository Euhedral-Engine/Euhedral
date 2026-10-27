//! The chat prompt corpus, generator `euhedral-chat-v2` (Java `ChatPromptCorpus`).
//!
//! Real instructions in the model's chat template with thinking disabled, so generation behaves like a served request.
//! A target gets one prompt per task. When it leaves room, a prompt quotes the longest paragraph-aligned prefix of a
//! frozen document that keeps it within the target, and its task refers to that document; otherwise it carries a
//! standalone task. The document is the Java benchmark's resource, a snapshot of repository documents and sources of
//! about 60K tokens, so 16K and 32K prompts are real long contexts. The token count is at most the target. Every task
//! asks for long-form output.

use euhedral_inference_tokenizer::TokenizerError;

use super::{Built, PromptError};

pub const GENERATOR: &str = "euhedral-chat-v2";

/// The least document a prompt quotes; shorter targets use standalone tasks.
const MIN_DOCUMENT_TOKENS: usize = 64;

/// The tasks of one target; iteration `i` takes task `i` modulo this.
pub const TASKS: usize = 4;

const DOCUMENT_TASKS: [&str; TASKS] = [
    "Summarize the document above in detail, section by section.",
    "Write a Java implementation of the main mechanism the document above describes, with comments explaining each \
     part.",
    "List the design decisions in the document above and explain the trade-off behind each one.",
    "Write a tutorial for a new engineer that explains the ideas in the document above, with examples.",
];

const STANDALONE_TASKS: [&str; TASKS] = [
    "Write a Java class that implements an LRU cache, with comments, and explain how it works.",
    "Explain in detail how a CPU executes an instruction, from fetch to write-back.",
    "Write a Python script that reads a CSV file of sales and prints totals per month, then explain it line by line.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
];

/// The frozen document, read from the Java benchmark's resources so the two engines quote the same text.
pub const DOCUMENT: &str = include_str!(
    "../../../../../benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/chat-corpus-v1-document.md"
);

/// Builds every task's prompt text for `target_tokens`, in task order, as Java's `ChatPromptCorpus.build` does: the
/// target is refused when any task's standalone prompt is above it, not only the one an iteration runs.
pub fn build_all<C>(target_tokens: usize, count: C) -> Result<Vec<Built>, PromptError>
where
    C: FnMut(&str) -> Result<usize, TokenizerError>,
{
    build_all_from(target_tokens, DOCUMENT, count)
}

/// As [`build_all`], quoting `document`.
pub fn build_all_from<C>(target_tokens: usize, document: &str, mut count: C) -> Result<Vec<Built>, PromptError>
where
    C: FnMut(&str) -> Result<usize, TokenizerError>,
{
    (0..TASKS)
        .map(|task| build_from(task, target_tokens, document, &mut count))
        .collect()
}

/// Builds task `task`'s prompt text for `target_tokens`; `count` is the number of tokens a text encodes to.
pub fn build<C>(task: usize, target_tokens: usize, count: C) -> Result<Built, PromptError>
where
    C: FnMut(&str) -> Result<usize, TokenizerError>,
{
    build_from(task, target_tokens, DOCUMENT, count)
}

/// As [`build`], quoting `document`. A checkout may hold the document with CRLF line endings; prompts are built from
/// its LF form so they, and their hashes, are the same on every platform.
pub fn build_from<C>(task: usize, target_tokens: usize, document: &str, mut count: C) -> Result<Built, PromptError>
where
    C: FnMut(&str) -> Result<usize, TokenizerError>,
{
    if target_tokens == 0 {
        return Err(PromptError::ZeroTarget);
    }
    let document = document.replace("\r\n", "\n");
    let paragraphs = paragraphs(&document);
    let task = task % TASKS;

    let standalone = chat(None, STANDALONE_TASKS[task]);
    let standalone_tokens = count(&standalone)?;
    if standalone_tokens > target_tokens {
        return Err(PromptError::TargetTooSmall {
            target: target_tokens,
            shortest: standalone_tokens,
        });
    }
    let instruction = DOCUMENT_TASKS[task];
    if count(&chat(Some(paragraphs[0]), instruction))? > target_tokens {
        return Ok(Built {
            text: standalone,
            tokens: standalone_tokens,
        });
    }
    let mut low = 1;
    let mut high = paragraphs.len();
    while low < high {
        let middle = (low + high).div_ceil(2);
        if count(&chat(Some(&paragraphs[..middle].join("\n\n")), instruction))? <= target_tokens {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let quoted = chat(Some(&paragraphs[..low].join("\n\n")), instruction);
    let quoted_tokens = count(&quoted)?;
    if quoted_tokens.saturating_sub(standalone_tokens) >= MIN_DOCUMENT_TOKENS {
        Ok(Built {
            text: quoted,
            tokens: quoted_tokens,
        })
    } else {
        Ok(Built {
            text: standalone,
            tokens: standalone_tokens,
        })
    }
}

/// The document's paragraphs as Java's `split("\n\n")` yields them: trailing empty ones are dropped.
fn paragraphs(document: &str) -> Vec<&str> {
    let mut paragraphs: Vec<&str> = document.split("\n\n").collect();
    while paragraphs.len() > 1 && paragraphs.last() == Some(&"") {
        paragraphs.pop();
    }
    paragraphs
}

fn chat(document: Option<&str>, task: &str) -> String {
    let user = match document {
        Some(document) => format!("Here is a document:\n\n{document}\n\n{task}"),
        None => task.to_string(),
    };
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per whitespace-separated word: enough to exercise sizing deterministically.
    fn count(text: &str) -> Result<usize, TokenizerError> {
        Ok(text.split_whitespace().count())
    }

    #[test]
    fn large_targets_quote_the_document_within_the_target() {
        for task in 0..TASKS {
            let built = build(task, 2000, count).unwrap();
            assert!(built.tokens <= 2000);
            assert!(built.tokens > 1500, "fills most of the target: {}", built.tokens);
            assert!(built.text.starts_with("<|im_start|>user\nHere is a document:"));
            assert!(built.text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
            assert_eq!(count(&built.text).unwrap(), built.tokens);
        }
    }

    #[test]
    fn short_targets_use_standalone_tasks() {
        for task in 0..TASKS {
            let built = build(task, 40, count).unwrap();
            assert!(!built.text.contains("Here is a document"));
            assert!(built.tokens <= 40);
        }
    }

    #[test]
    fn a_target_below_the_standalone_prompt_is_refused() {
        assert!(matches!(
            build(0, 5, count),
            Err(PromptError::TargetTooSmall { target: 5, .. })
        ));
        assert!(matches!(build(0, 0, count), Err(PromptError::ZeroTarget)));
    }

    #[test]
    fn a_target_between_the_shortest_and_the_longest_standalone_prompt_is_refused_whole() {
        // Java builds all four tasks and throws when any standalone prompt is above the target.
        let words: Vec<usize> = STANDALONE_TASKS
            .iter()
            .map(|task| count(&chat(None, task)).unwrap())
            .collect();
        let shortest = *words.iter().min().unwrap();
        let longest = *words.iter().max().unwrap();
        assert!(shortest < longest, "the tasks differ in length: {words:?}");
        let target = shortest;
        let shortest_task = words.iter().position(|&tokens| tokens == shortest).unwrap();
        // The shortest task alone fits, as the old per-iteration build allowed.
        assert!(build(shortest_task, target, count).is_ok());
        let refused = build_all(target, count).unwrap_err();
        assert!(matches!(refused, PromptError::TargetTooSmall { target: t, .. } if t == target));
        // At the longest standalone prompt every task fits.
        let built = build_all(longest, count).unwrap();
        assert_eq!(built.len(), TASKS);
        assert!(built.iter().all(|prompt| prompt.tokens <= longest));
    }

    #[test]
    fn building_is_deterministic_and_tasks_differ() {
        assert_eq!(build(1, 2000, count).unwrap(), build(1, 2000, count).unwrap());
        assert_ne!(build(0, 2000, count).unwrap().text, build(1, 2000, count).unwrap().text);
        assert_eq!(build(1, 2000, count).unwrap(), build(1 + TASKS, 2000, count).unwrap());
    }

    #[test]
    fn crlf_checkouts_build_the_same_prompts() {
        let lf = DOCUMENT.replace("\r\n", "\n");
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(
            build_from(2, 2000, &lf, count).unwrap(),
            build_from(2, 2000, &crlf, count).unwrap()
        );
        assert_eq!(build_from(2, 2000, &lf, count).unwrap(), build(2, 2000, count).unwrap());
    }

    #[test]
    fn trailing_empty_paragraphs_are_dropped_like_javas_split() {
        assert_eq!(paragraphs("a\n\nb\n\n\n\n"), ["a", "b"]);
        assert_eq!(paragraphs("a\n\n\n\nb"), ["a", "", "b"]);
        assert_eq!(paragraphs("a"), ["a"]);
    }

    #[test]
    fn a_document_that_adds_too_little_falls_back_to_the_standalone_task() {
        // Room for the first paragraph but fewer than 64 tokens beyond the standalone prompt.
        let built = build_from(0, 60, "tiny document\n\nsecond paragraph", count).unwrap();
        assert!(!built.text.contains("Here is a document"));
    }
}

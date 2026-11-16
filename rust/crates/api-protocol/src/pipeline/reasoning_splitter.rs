//! Splits output that opens inside the model's think block into reasoning and answer at `</think>`.

use crate::pipeline::text_space::{held_tag_prefix, strip_leading_space, trailing_space_start};

/// The end of the think block.
pub const THINK_END: &str = "</think>";

/// One part of the split output, in generation order: all reasoning, then all answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Split {
    Reasoning(String),
    Answer(String),
}

/// Inverts the template's `'<think>\n' + reasoning|trim + '\n</think>\n\n' + content`: the reasoning is trimmed and
/// the answer's leading whitespace dropped. Reasoning text that could still be part of `</think>`, or whitespace
/// that could still precede it, is held until the next chunk decides it, so reasoning streams in order with at
/// most that much delay. Without a `</think>` the block never closes: [`finish`](Self::finish) releases the held
/// reasoning and the answer stays empty.
#[derive(Debug, Default)]
pub struct ReasoningSplitter {
    held: String,
    answering: bool,
    reasoning_started: bool,
    answer_started: bool,
}

impl ReasoningSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once `</think>` has been generated.
    pub fn answering(&self) -> bool {
        self.answering
    }

    /// Takes the next chunk of output and appends the parts it settles to `out`.
    pub fn accept(&mut self, chunk: &str, out: &mut Vec<Split>) {
        if self.answering {
            self.answer(chunk, out);
            return;
        }
        self.held.push_str(chunk);
        if let Some(end) = self.held.find(THINK_END) {
            let reasoning = self.held[..trailing_space_start(&self.held, end)].to_string();
            self.reasoning(&reasoning, out);
            let rest = self.held[end + THINK_END.len()..].to_string();
            self.held.clear();
            self.answering = true;
            self.answer(&rest, out);
            return;
        }
        let limit = self.held.len() - held_tag_prefix(&self.held, THINK_END);
        let safe = trailing_space_start(&self.held, limit);
        let reasoning = self.held[..safe].to_string();
        self.held.drain(..safe);
        self.reasoning(&reasoning, out);
    }

    /// Generation ended: what is held is reasoning that never reached `</think>`, less its trailing whitespace.
    pub fn finish(&mut self, out: &mut Vec<Split>) {
        if !self.answering {
            let reasoning = self.held[..trailing_space_start(&self.held, self.held.len())].to_string();
            self.reasoning(&reasoning, out);
        }
        self.held.clear();
    }

    fn reasoning(&mut self, text: &str, out: &mut Vec<Split>) {
        let text = if self.reasoning_started {
            text
        } else {
            strip_leading_space(text)
        };
        if text.is_empty() {
            return;
        }
        self.reasoning_started = true;
        out.push(Split::Reasoning(text.to_string()));
    }

    fn answer(&mut self, text: &str, out: &mut Vec<Split>) {
        let text = if self.answer_started {
            text
        } else {
            strip_leading_space(text)
        };
        if text.is_empty() {
            return;
        }
        self.answer_started = true;
        out.push(Split::Answer(text.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasoning_of(parts: &[Split]) -> String {
        parts
            .iter()
            .filter_map(|part| {
                if let Split::Reasoning(text) = part {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    fn answer_of(parts: &[Split]) -> String {
        parts
            .iter()
            .filter_map(|part| {
                if let Split::Answer(text) = part {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Every way of cutting the output into two chunks splits it the same way.
    #[test]
    fn the_split_does_not_depend_on_chunk_boundaries() {
        let output = " \nFirst <b>step</b>, then </thin k.\n\n</think>\n\n Answer </think> stays.";
        for cut in 0..=output.len() {
            let mut splitter = ReasoningSplitter::new();
            let mut parts = Vec::new();
            splitter.accept(&output[..cut], &mut parts);
            splitter.accept(&output[cut..], &mut parts);
            splitter.finish(&mut parts);
            assert_eq!(reasoning_of(&parts), "First <b>step</b>, then </thin k.", "cut {cut}");
            assert_eq!(answer_of(&parts), "Answer </think> stays.", "cut {cut}");
            assert!(splitter.answering());
            let first_answer = parts.iter().position(|part| matches!(part, Split::Answer(_)));
            let last_reasoning = parts.iter().rposition(|part| matches!(part, Split::Reasoning(_)));
            assert!(
                last_reasoning < first_answer,
                "reasoning after the answer began, cut {cut}"
            );
        }
    }

    #[test]
    fn reasoning_streams_as_it_arrives_and_holds_only_a_possible_tag() {
        let mut splitter = ReasoningSplitter::new();
        let mut parts = Vec::new();
        splitter.accept("Thinking about it", &mut parts);
        assert_eq!(parts, [Split::Reasoning("Thinking about it".to_string())]);
        splitter.accept(" more\n</thi", &mut parts);
        assert_eq!(
            parts,
            [
                Split::Reasoning("Thinking about it".to_string()),
                Split::Reasoning(" more".to_string())
            ]
        );
        splitter.accept("s is text", &mut parts);
        assert_eq!(reasoning_of(&parts), "Thinking about it more\n</this is text");
        assert!(!splitter.answering());
    }

    #[test]
    fn reasoning_that_never_ends_is_trimmed_at_the_end() {
        let mut splitter = ReasoningSplitter::new();
        let mut parts = Vec::new();
        splitter.accept("Unfinished  \n", &mut parts);
        splitter.finish(&mut parts);
        assert_eq!(reasoning_of(&parts), "Unfinished");
        assert_eq!(answer_of(&parts), "");
    }

    #[test]
    fn multi_byte_text_is_held_and_released_whole() {
        let mut splitter = ReasoningSplitter::new();
        let mut parts = Vec::new();
        splitter.accept("é😀 ", &mut parts);
        splitter.accept("</think>日本", &mut parts);
        assert_eq!(
            parts,
            [Split::Reasoning("é😀".to_string()), Split::Answer("日本".to_string())]
        );
    }
}

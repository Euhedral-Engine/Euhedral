//! Applies `stop` strings to decoded text before it reaches the client.

use std::mem::take;

/// Text that could still become a stop string is held back, so a match spanning chunks is never partially emitted.
/// After a match, the stop string and everything after it are discarded.
#[derive(Debug)]
pub struct StopSequenceFilter {
    stops: Vec<String>,
    pending: String,
    matched: bool,
    matched_stop: Option<String>,
}

impl StopSequenceFilter {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.to_vec(),
            pending: String::new(),
            matched: false,
            matched_stop: None,
        }
    }

    /// Returns the text that is now safe to emit; empty when all of it may still begin a stop string.
    pub fn accept(&mut self, chunk: &str) -> String {
        if self.matched {
            return String::new();
        }
        if self.stops.is_empty() {
            return chunk.to_string();
        }
        self.pending.push_str(chunk);
        match self.earliest_match() {
            Some((position, stop)) => {
                // A stop that starts at or before this one and could still complete would win over it.
                if self.hold_start().is_some_and(|start| start <= position) {
                    return self.release_before_hold();
                }
                self.settle(position, stop)
            }
            None => self.release_before_hold(),
        }
    }

    /// Releases held-back text at the natural end of generation; a complete stop still waiting for a longer one to
    /// finish is decided now, as nothing more can arrive.
    pub fn finish(&mut self) -> String {
        match self.earliest_match() {
            Some((position, stop)) => self.settle(position, stop),
            None => take(&mut self.pending),
        }
    }

    pub fn matched(&self) -> bool {
        self.matched
    }

    /// The stop string that matched: of those matching at the earliest position, the longest, if any.
    pub fn matched_stop(&self) -> Option<&str> {
        self.matched_stop.as_deref()
    }

    fn settle(&mut self, position: usize, stop: usize) -> String {
        self.matched_stop = Some(self.stops[stop].clone());
        self.matched = true;
        let emitted = self.pending[..position].to_string();
        self.pending.clear();
        emitted
    }

    /// Emits the text before the earliest point where a stop could still begin; the rest stays held.
    fn release_before_hold(&mut self) -> String {
        let safe = self.hold_start().unwrap_or(self.pending.len());
        self.pending.drain(..safe).collect()
    }

    /// The earliest complete stop in the pending text: its position and index; the longest at that position.
    fn earliest_match(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (index, stop) in self.stops.iter().enumerate() {
            let Some(position) = self.pending.find(stop.as_str()) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((at, held)) => position < at || (position == at && stop.len() > self.stops[held].len()),
            };
            if better {
                best = Some((position, index));
            }
        }
        best
    }

    /// The start of the earliest suffix of the pending text that is a proper prefix of some stop string. It is
    /// shorter than the longest stop, so the hold never exceeds that.
    fn hold_start(&self) -> Option<usize> {
        self.pending.char_indices().map(|(start, _)| start).find(|&start| {
            let suffix = &self.pending[start..];
            self.stops
                .iter()
                .any(|stop| stop.len() > suffix.len() && stop.starts_with(suffix))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(stops: &[&str]) -> StopSequenceFilter {
        let stops: Vec<String> = stops.iter().map(|stop| (*stop).to_string()).collect();
        StopSequenceFilter::new(&stops)
    }

    #[test]
    fn passes_text_through_without_stops() {
        let mut filter = filter(&[]);
        assert_eq!(filter.accept("abc"), "abc");
        assert_eq!(filter.finish(), "");
        assert!(!filter.matched());
    }

    #[test]
    fn holds_back_a_possible_stop_prefix_until_it_resolves() {
        let mut filter = filter(&["END"]);
        assert_eq!(filter.accept("hello E"), "hello ");
        assert_eq!(filter.accept("N"), "");
        assert_eq!(filter.accept("x"), "ENx", "a broken prefix is released");
        assert!(!filter.matched());
    }

    #[test]
    fn match_spanning_chunks_truncates_at_the_stop() {
        let mut filter = filter(&["END"]);
        assert_eq!(filter.accept("a E"), "a ");
        assert_eq!(filter.accept("ND tail"), "");
        assert!(filter.matched());
        assert_eq!(filter.matched_stop(), Some("END"));
        assert_eq!(filter.accept("more"), "", "nothing is emitted after a match");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        let mut filter = filter(&["zz", "\n\n"]);
        assert_eq!(filter.accept("one\n\ntwo zz"), "one");
        assert!(filter.matched());
        assert_eq!(filter.matched_stop(), Some("\n\n"));
    }

    #[test]
    fn finish_releases_held_text() {
        let mut filter = filter(&["STOP"]);
        assert_eq!(filter.accept("xST"), "x");
        assert_eq!(filter.finish(), "ST");
    }

    #[test]
    fn the_longest_of_the_stops_matching_at_the_earliest_position_is_reported() {
        let mut filter = filter(&["ab", "abc"]);
        assert_eq!(filter.accept("xabc"), "x");
        assert_eq!(filter.matched_stop(), Some("abc"));
    }

    #[test]
    fn a_stop_with_multi_byte_characters_is_held_at_character_boundaries() {
        let mut filter = filter(&["日本語"]);
        assert_eq!(filter.accept("a日"), "a");
        assert_eq!(filter.accept("本"), "");
        assert_eq!(filter.accept("x"), "日本x");
        let mut filter = self::filter(&["日本語"]);
        assert_eq!(filter.accept("日本語!"), "");
        assert!(filter.matched());
    }

    #[test]
    fn a_shorter_stop_waits_for_a_longer_one_that_starts_at_the_same_place() {
        let mut cut = filter(&["abc", "ab"]);
        assert_eq!(cut.accept("xab"), "x");
        assert!(!cut.matched());
        assert_eq!(cut.accept("c"), "");
        assert_eq!(cut.matched_stop(), Some("abc"));
        let mut broken = filter(&["abc", "ab"]);
        assert_eq!(broken.accept("xab"), "x");
        assert_eq!(broken.accept("y"), "");
        assert_eq!(broken.matched_stop(), Some("ab"));
    }

    #[test]
    fn a_stop_waits_for_a_longer_one_that_starts_earlier() {
        let mut whole = filter(&["abcd", "c"]);
        assert_eq!(whole.accept("abcd"), "");
        assert_eq!(whole.matched_stop(), Some("abcd"));
        let mut cut = filter(&["abcd", "c"]);
        assert_eq!(cut.accept("abc"), "");
        assert!(!cut.matched());
        assert_eq!(cut.accept("d"), "");
        assert_eq!(cut.matched_stop(), Some("abcd"));
        let mut broken = filter(&["abcd", "c"]);
        assert_eq!(broken.accept("abc"), "");
        assert_eq!(broken.accept("x"), "ab");
        assert_eq!(broken.matched_stop(), Some("c"));
    }

    #[test]
    fn a_stop_prefix_that_never_completes_is_released_at_finish() {
        let mut filter = filter(&["abcd", "c"]);
        assert_eq!(filter.accept("xab"), "x");
        assert_eq!(filter.finish(), "ab");
        assert!(!filter.matched());
    }

    #[test]
    fn a_complete_stop_still_waiting_is_decided_at_finish() {
        let mut filter = filter(&["abcd", "c"]);
        assert_eq!(filter.accept("xabc"), "x");
        assert_eq!(filter.finish(), "ab");
        assert_eq!(filter.matched_stop(), Some("c"));
    }

    /// Feeds the chunks, checking after each that the hold is shorter than the longest stop; returns the emitted
    /// text, the matched stop and whether it matched.
    fn feed(stops: &[&str], chunks: &[&str]) -> (String, Option<String>) {
        let mut filter = filter(stops);
        let longest = stops.iter().map(|stop| stop.len()).max().unwrap();
        let mut emitted = String::new();
        for chunk in chunks {
            emitted.push_str(&filter.accept(chunk));
            assert!(
                filter.pending.len() < longest,
                "held {:?} for {stops:?}",
                filter.pending
            );
        }
        emitted.push_str(&filter.finish());
        (emitted, filter.matched_stop().map(str::to_string))
    }

    #[test]
    fn the_hold_never_exceeds_the_longest_stop() {
        let mut filter = filter(&["abcd", "c", "xyz"]);
        for chunk in ["ab", "ab", "a", "xy", "xy", "ab", "x"] {
            filter.accept(chunk);
            assert!(!filter.matched(), "{chunk:?} completed a stop");
            assert!(filter.pending.len() < 4, "held {:?}", filter.pending);
        }
    }

    #[test]
    fn every_cut_gives_the_same_result_and_a_bounded_hold() {
        let sets: [&[&str]; 5] = [&["abc", "ab"], &["ab", "abc"], &["abcd", "c"], &["bc", "abcd"], &["aa"]];
        let inputs = ["xabc", "xaby", "abcd", "abcx", "aaa", "xaaab", "ababcab", "abab"];
        for stops in sets {
            for input in inputs {
                let whole = feed(stops, &[input]);
                let bounds: Vec<usize> = input
                    .char_indices()
                    .map(|(index, _)| index)
                    .chain([input.len()])
                    .collect();
                for &first in &bounds {
                    for &second in bounds.iter().filter(|&&index| index >= first) {
                        let chunks = [&input[..first], &input[first..second], &input[second..]];
                        assert_eq!(
                            feed(stops, &chunks),
                            whole,
                            "{stops:?} on {input:?}, cuts {first}, {second}"
                        );
                    }
                }
            }
        }
    }
}

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
        if let Some(position) = self.earliest_match() {
            self.matched_stop = self
                .stops
                .iter()
                .find(|stop| self.pending[position..].starts_with(stop.as_str()))
                .cloned();
            self.matched = true;
            let emitted = self.pending[..position].to_string();
            self.pending.clear();
            return emitted;
        }
        let safe = self.pending.len() - self.longest_stop_prefix_suffix();
        self.pending.drain(..safe).collect()
    }

    /// Releases held-back text at the natural end of generation.
    pub fn finish(&mut self) -> String {
        take(&mut self.pending)
    }

    pub fn matched(&self) -> bool {
        self.matched
    }

    /// The stop string that matched (the first listed of those matching at the earliest position), if any.
    pub fn matched_stop(&self) -> Option<&str> {
        self.matched_stop.as_deref()
    }

    fn earliest_match(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop.as_str()))
            .min()
    }

    /// Byte length of the longest suffix of the pending text that is a proper prefix of some stop string.
    fn longest_stop_prefix_suffix(&self) -> usize {
        let mut longest = 0;
        for stop in &self.stops {
            let prefix = stop
                .char_indices()
                .map(|(index, _)| index)
                .rfind(|&length| length > longest && length > 0 && self.pending.ends_with(&stop[..length]));
            if let Some(length) = prefix {
                longest = length;
            }
        }
        longest
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
    fn the_first_listed_of_the_stops_matching_at_the_earliest_position_is_reported() {
        let mut filter = filter(&["ab", "abc"]);
        assert_eq!(filter.accept("xabc"), "x");
        assert_eq!(filter.matched_stop(), Some("ab"));
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
}

package io.euhedral_execution.inference.api.chat;

import java.util.List;

/// Applies OpenAI `stop` strings to decoded text before it reaches the client.
///
/// Text that could still become a stop string is held back, so a match spanning decoder chunks is never
/// partially emitted. After a match, the stop string and everything after it are discarded.
/// Confined to the generation thread.
final class StopSequenceFilter {
    private final List<String> stops;
    private final StringBuilder pending = new StringBuilder();
    private boolean matched;
    private String matchedStop;

    StopSequenceFilter(List<String> stops) {
        this.stops = List.copyOf(stops);
    }

    /// Returns the text that is now safe to emit; empty when all of it may still begin a stop string.
    String accept(String chunk) {
        if (this.matched) return "";
        if (this.stops.isEmpty()) return chunk;
        this.pending.append(chunk);
        int match = earliestMatch();
        if (match >= 0) {
            for (String stop : this.stops)
                if (this.matchedStop == null && this.pending.indexOf(stop, match) == match) this.matchedStop = stop;
            this.matched = true;
            String emitted = this.pending.substring(0, match);
            this.pending.setLength(0);
            return emitted;
        }
        int safe = this.pending.length() - longestStopPrefixSuffix();
        String emitted = this.pending.substring(0, safe);
        this.pending.delete(0, safe);
        return emitted;
    }

    /// Releases held-back text at the natural end of generation.
    String finish() {
        String remaining = this.pending.toString();
        this.pending.setLength(0);
        return remaining;
    }

    boolean matched() {
        return this.matched;
    }

    /// The stop string that matched (the first listed of those matching at the earliest position), or null.
    String matchedStop() {
        return this.matchedStop;
    }

    private int earliestMatch() {
        int earliest = -1;
        for (String stop : this.stops) {
            int index = this.pending.indexOf(stop);
            if (index >= 0 && (earliest < 0 || index < earliest)) earliest = index;
        }
        return earliest;
    }

    private int longestStopPrefixSuffix() {
        int longest = 0;
        for (String stop : this.stops) {
            int limit = Math.min(stop.length() - 1, this.pending.length());
            for (int length = limit; length > longest; length--) {
                if (pendingEndsWith(stop, length)) {
                    longest = length;
                    break;
                }
            }
        }
        return longest;
    }

    private boolean pendingEndsWith(String stop, int prefixLength) {
        int offset = this.pending.length() - prefixLength;
        for (int index = 0; index < prefixLength; index++) {
            if (this.pending.charAt(offset + index) != stop.charAt(index)) return false;
        }
        return true;
    }
}

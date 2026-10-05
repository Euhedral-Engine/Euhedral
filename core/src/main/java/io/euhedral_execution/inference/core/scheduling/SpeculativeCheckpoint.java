package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.prefix.PrefixNode;
import java.util.List;
import java.util.function.ToLongFunction;

/// The state a speculative decoding strategy keeps beside a sequence's base state (its KV pages, GDN state and
/// position), as prefix checkpoints store it. The prefix cache stores, pins, evicts, finds and copies it; the
/// strategy that owns it defines what it holds and where each piece goes.
///
/// A node's extent holds the base state first and this state after it, at the offset the cache passes in. A
/// node holds speculative state of one kind only, and a restore reads a chain whose nodes all hold its kind.
public interface SpeculativeCheckpoint {

    /// Names the kind of state; nodes of different kinds never stand in for one another.
    String kind();

    /// Bytes this state adds to the extent of a node covering positions `[startPosition, position)`.
    long extentBytes(int startPosition, int position);

    /// Whether `sequence` holds this state for a checkpoint at `position`; when not, the node is stored without it.
    boolean holds(QwenSequenceState sequence, int position);

    /// The copies that save this state of `sequence` into `node`'s extent, from host offset `at`.
    List<PrefixLayout.Copy> captureCopies(PrefixNode node, long at, QwenSequenceState sequence);

    /// Readies `sequence` (its state allocated, held under its lease) to receive this state at the leaf's
    /// position, and returns the copies that load it from `chain`; `at` gives each node's offset of this state.
    List<PrefixLayout.Copy> restoreCopies(
            List<PrefixNode> chain, ToLongFunction<PrefixNode> at, QwenSequenceState sequence);

    /// Publishes the restored state once the copies ran.
    void restored(QwenSequenceState sequence, int position);
}

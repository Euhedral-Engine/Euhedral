package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// Receives the answer to an [ExpertCache#acquireAsync] request. `tag` is whatever the requester passed, so one
/// listener object serves many requests without a closure each.
///
/// Called exactly once per request: with the lease (the requester now owns it and must close it) and a null
/// failure, or with a null lease and the reason the expert is not available ([ExpertTransferException] for a failed
/// load, [IllegalStateException] for a closed cache). It may run on the thread that made the request (a hit, or a
/// failure that is known at once) or later on a lattice worker; it must not block.
@FunctionalInterface
public interface ExpertListener {
    void ready(int tag, ExpertLease lease, Throwable failure);
}

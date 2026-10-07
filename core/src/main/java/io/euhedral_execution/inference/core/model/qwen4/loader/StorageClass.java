package io.euhedral_execution.inference.core.model.qwen4.loader;

/// Where an object lives at runtime and how execution reaches it. Placement is decided by the planner from
/// the device's free memory, the requested context and the artifact's object sizes, never by the user.
public enum StorageClass {
    /// Permanently in device memory.
    DEVICE_RESIDENT,
    /// In pinned host memory that kernels read in place over the bus (gathers: the token embedding).
    HOST_MAPPED,
    /// Held on the host; execution stages the object (a fixed projection) or the rows it needs (n-gram
    /// embeddings) to the device on use.
    HOST_STAGED,
    /// Brought into a bounded device cache on demand and evicted when its slot is needed (routed experts).
    /// Never a permanent resident: a device address is only valid while a lease on it is held.
    DEVICE_CACHED,
    /// Present in the artifact and validated, but not loaded by the selected execution mode.
    DEFERRED
}

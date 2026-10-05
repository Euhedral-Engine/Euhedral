package io.euhedral_execution.inference.core.qwen4;

/// A weight as an operator uses it: where its bytes are on the device when the operator launches.
///
/// A device-resident or host-mapped tensor is always at the same address. A host-staged one is copied to a
/// slot of the staging ring when it is asked for, on the stream the operator is submitting to, so the copy is
/// ordered before the operator's kernels and after the earlier use of the slot. [#address] is therefore called
/// exactly when the weight is about to be used, never ahead of time: asking for several staged weights before
/// launching any of them could reuse a ring slot that a pending kernel still reads.
public interface Qwen4Weight {

    /// The device address at which the next launch reads the weight; stages it first when it is host-staged.
    long address();

    /// Bytes of the weight's payload.
    long bytes();

    /// A weight that is always at `address`.
    static Qwen4Weight of(long address, long bytes) {
        return new Qwen4Weight() {
            @Override
            public long address() {
                return address;
            }

            @Override
            public long bytes() {
                return bytes;
            }
        };
    }
}

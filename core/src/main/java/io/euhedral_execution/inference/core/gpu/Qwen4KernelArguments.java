package io.euhedral_execution.inference.core.gpu;

import java.util.Arrays;

/// The arguments of one [Qwen4Kernel] launch, as the native launcher takes them: each a 64-bit word and the byte
/// size (4 or 8) the kernel declares for it. Reusable: [#clear] rewinds it, so a hot path keeps one per thread.
public final class Qwen4KernelArguments {

    /// The launcher's argument limit.
    public static final int MAX = 32;

    private final long[] words = new long[MAX];
    private final byte[] sizes = new byte[MAX];
    private int count;

    public Qwen4KernelArguments clear() {
        this.count = 0;
        return this;
    }

    /// A device address or another 64-bit value.
    public Qwen4KernelArguments pointer(long address) {
        return add(address, 8);
    }

    public Qwen4KernelArguments int32(int value) {
        return add(value & 0xffffffffL, 4);
    }

    public Qwen4KernelArguments float32(float value) {
        return add(Float.floatToRawIntBits(value) & 0xffffffffL, 4);
    }

    private Qwen4KernelArguments add(long word, int size) {
        if (this.count == MAX) throw new IllegalStateException("too many kernel arguments");
        this.words[this.count] = word;
        this.sizes[this.count] = (byte) size;
        this.count++;
        return this;
    }

    public int count() {
        return this.count;
    }

    public long[] words() {
        return this.words;
    }

    public byte[] sizes() {
        return this.sizes;
    }

    @Override
    public String toString() {
        return "args" + Arrays.toString(Arrays.copyOf(this.words, this.count));
    }
}

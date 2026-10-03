package io.euhedral_execution.inference.core.scheduling;

/// Mixes the device addresses and sizes a quantum's submissions depend on into one 64-bit value, part of
/// its capture key: a quantum whose state was reallocated keys differently from the one that was captured.
final class CaptureFingerprint {
    private long value = 0x9e3779b97f4a7c15L;

    CaptureFingerprint add(long word) {
        long x = this.value ^ (word + 0x9e3779b97f4a7c15L + (this.value << 6) + (this.value >>> 2));
        x ^= x >>> 30;
        x *= 0xbf58476d1ce4e5b9L;
        x ^= x >>> 27;
        x *= 0x94d049bb133111ebL;
        this.value = x ^ (x >>> 31);
        return this;
    }

    long value() {
        return this.value;
    }
}

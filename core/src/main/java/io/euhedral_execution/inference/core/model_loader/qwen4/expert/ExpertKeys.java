package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// The flat index of every `(bank, expert)` pair of an expert bank array: `firstKey[bank] + expert`, so a
/// directory over all experts is one primitive array and two experts of different banks can never share an
/// index. Every lookup checks both coordinates against their own bank, which is what rules out aliasing
/// (the last expert of one bank is never read as the first of the next).
final class ExpertKeys {
    private final int[] firstKey;

    ExpertKeys(ExpertBank[] banks) {
        if (banks.length == 0) throw new IllegalArgumentException("at least one expert bank is required");
        this.firstKey = new int[banks.length + 1];
        long total = 0;
        for (int bank = 0; bank < banks.length; bank++) {
            this.firstKey[bank] = (int) total;
            total += banks[bank].expertCount();
            if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("too many experts for one directory");
        }
        this.firstKey[banks.length] = (int) total;
    }

    int bankCount() {
        return this.firstKey.length - 1;
    }

    int keyCount() {
        return this.firstKey[this.firstKey.length - 1];
    }

    int expertCount(int bank) {
        return this.firstKey[checkedBank(bank) + 1] - this.firstKey[bank];
    }

    /// The index of `expert` in `bank`.
    ///
    /// @throws IndexOutOfBoundsException when either is outside its range
    int key(int bank, int expert) {
        int count = expertCount(bank);
        if (expert < 0 || expert >= count)
            throw new IndexOutOfBoundsException("expert " + expert + " of bank " + bank + " (" + count + ")");
        return this.firstKey[bank] + expert;
    }

    /// The bank that owns `key`.
    int bankOf(int key) {
        if (key < 0 || key >= keyCount()) throw new IndexOutOfBoundsException("key " + key);
        int low = 0;
        int high = bankCount() - 1;
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            if (this.firstKey[middle] <= key) low = middle;
            else high = middle - 1;
        }
        return low;
    }

    int expertOf(int key, int bank) {
        return key - this.firstKey[bank];
    }

    private int checkedBank(int bank) {
        if (bank < 0 || bank >= bankCount())
            throw new IndexOutOfBoundsException("bank " + bank + " (" + bankCount() + ")");
        return bank;
    }
}

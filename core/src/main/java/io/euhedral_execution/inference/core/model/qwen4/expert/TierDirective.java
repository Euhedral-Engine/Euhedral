package io.euhedral_execution.inference.core.model.qwen4.expert;

/// What the host tier asks of one load, decided by the tier's owner before the load runs and read by the
/// frames that do it. A load owns one: the owner writes it, then the load's frames are published and read it.
public final class TierDirective {

    /// What the load does with the tier.
    public enum Mode {
        /// There is no tier: read the artifact into the staging slot.
        NONE,
        /// The record is in the tier's slot: copy it from there into the staging slot.
        HIT,
        /// The record is not in the tier and has a slot reserved: read the artifact into the slot, then
        /// copy it into the staging slot.
        FILL,
        /// The record is not in the tier and no slot may be taken: read the artifact into the staging
        /// slot, as without a tier.
        BYPASS
    }

    private Mode mode = Mode.NONE;
    private int slot = -1;
    private long address;

    public Mode mode() {
        return this.mode;
    }

    /// The tier slot (global to the tier) the mode refers to, or -1.
    public int slot() {
        return this.slot;
    }

    /// The address of the tier slot's memory, or 0.
    public long address() {
        return this.address;
    }

    void set(Mode mode, int slot, long address) {
        this.mode = mode;
        this.slot = slot;
        this.address = address;
    }

    public void clear() {
        set(Mode.NONE, -1, 0);
    }

    @Override
    public String toString() {
        return this.mode + (this.slot < 0 ? "" : " slot " + this.slot);
    }
}

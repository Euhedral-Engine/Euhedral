package io.euhedral_execution.inference.core.model_loader.qwen4;

/// The size of the routed-expert cache a budget allows: whole slots, each able to hold the largest expert record.
public record ExpertCacheGeometry(long slotBytes, int slotCount, long totalExperts, int minimumSlots) {

    public ExpertCacheGeometry {
        if (slotBytes <= 0 || slotCount < 0 || totalExperts <= 0 || minimumSlots <= 0)
            throw new IllegalArgumentException("invalid expert cache geometry");
    }

    /// The device bytes the cache holds.
    public long bytes() {
        return slotBytes * slotCount;
    }

    /// Slots a budget of `budgetBytes` buys, capped at one slot per expert: more would never be used.
    public static ExpertCacheGeometry derive(long budgetBytes, long slotBytes, long totalExperts, int minimumSlots) {
        long affordable = budgetBytes <= 0 ? 0 : budgetBytes / slotBytes;
        return new ExpertCacheGeometry(slotBytes, (int) Math.min(affordable, totalExperts), totalExperts, minimumSlots);
    }

    public boolean viable() {
        return slotCount >= minimumSlots;
    }
}

#pragma once
// Programmatic dependent launch. A kernel launched with the programmatic-stream-serialization
// attribute may start while its predecessor drains; griddepcontrol.wait blocks until that predecessor
// has completed and its memory is visible. Every kernel launched with the attribute MUST call this
// before reading or writing anything a preceding kernel touches. Without the attribute both
// instructions are no-ops, so the same kernel serves ordinary launches.
__device__ __forceinline__ void euhedral_pdl_begin() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

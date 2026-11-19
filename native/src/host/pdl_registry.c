#include "pdl_registry.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

/* Slots (a power of two); at most three quarters are filled so a probe sequence always reaches an empty slot. */
#define SLOTS 1024u
#define LIMIT (SLOTS / 4u * 3u)

static _Atomic(uintptr_t) slots[SLOTS];
static atomic_uint used;
static atomic_flag reported = ATOMIC_FLAG_INIT;

static unsigned int home(uintptr_t key) {
    return (unsigned int)(((uint64_t)key * 0x9E3779B97F4A7C15ull) >> 54) & (SLOTS - 1u);
}

int euhedral_pdl_registry_add(const void* function) {
    if (function == NULL) return 1;
    const uintptr_t key = (uintptr_t)function;
    for (unsigned int slot = home(key), probes = 0; probes < SLOTS; slot = (slot + 1u) & (SLOTS - 1u), probes++) {
        uintptr_t current = atomic_load(&slots[slot]);
        if (current == key) return 1;
        if (current != 0) continue;
        if (atomic_load(&used) >= LIMIT) break;
        if (atomic_compare_exchange_strong(&slots[slot], &current, key)) {
            atomic_fetch_add(&used, 1u);
            return 1;
        }
        if (current == key) return 1;
        /* Another thread took the slot with a different function: continue probing. */
    }
    if (!atomic_flag_test_and_set(&reported))
        fprintf(stderr, "euhedral: the programmatic-launch registry is full (%u functions); later kernels are not registered\n", LIMIT);
    return 0;
}

int euhedral_pdl_registry_contains(const void* function) {
    const uintptr_t key = (uintptr_t)function;
    if (key == 0) return 0;
    for (unsigned int slot = home(key), probes = 0; probes < SLOTS; slot = (slot + 1u) & (SLOTS - 1u), probes++) {
        const uintptr_t current = atomic_load(&slots[slot]);
        if (current == key) return 1;
        if (current == 0) return 0;
    }
    return 0;
}

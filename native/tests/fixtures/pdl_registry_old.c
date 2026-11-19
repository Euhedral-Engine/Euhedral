/* The former registry: 64 slots filled in registration order, later registrations dropped, a linear scan.
 * Kept as the harness's negative control, behind the current registry's interface. */
#include <stdatomic.h>
#include <stdint.h>

#define PDL_CAPACITY 64
static _Atomic(const void*) pdl_functions[PDL_CAPACITY];
static atomic_uint pdl_count;
#define LIMIT PDL_CAPACITY
#define used pdl_count

int euhedral_pdl_registry_add(const void* function) {
    if (function == 0) return 1;
    unsigned int slot = atomic_fetch_add(&pdl_count, 1u);
    if (slot < PDL_CAPACITY) {
        atomic_store(&pdl_functions[slot], function);
        return 1;
    }
    return 0;
}

int euhedral_pdl_registry_contains(const void* function) {
    unsigned int count = atomic_load(&pdl_count);
    if (count > PDL_CAPACITY) count = PDL_CAPACITY;
    for (unsigned int index = 0; index < count; index++)
        if (atomic_load(&pdl_functions[index]) == function) return 1;
    return 0;
}

#ifndef EUHEDRAL_PDL_REGISTRY_H
#define EUHEDRAL_PDL_REGISTRY_H

#include "euhedral_cuda.h"

/* The kernels that may overlap their predecessor (programmatic dependent launch): a lock-free open-addressing
 * set of function handles. Registration may run from any thread; lookup runs on every kernel launch and
 * touches one slot when the function is absent. */

/* Adds the function (idempotent). Returns 1, or 0 when the set is full: the refusal is also reported once on
 * standard error, never dropped silently. NULL is ignored and returns 1. */
EUHEDRAL_CUDA_EXPORT int euhedral_pdl_registry_add(const void* function);

/* Whether the function was added. */
EUHEDRAL_CUDA_EXPORT int euhedral_pdl_registry_contains(const void* function);

#endif

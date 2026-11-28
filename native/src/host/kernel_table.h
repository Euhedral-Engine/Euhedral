#ifndef EUHEDRAL_KERNEL_TABLE_H
#define EUHEDRAL_KERNEL_TABLE_H

#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
#include <cuda.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* A table of kernels the library can report on: their names, and whether each is available on this device. The
 * Rust engine loads and launches the kernels itself; the library compiles a table's modules only to answer
 * availability. A table's entry order is the order its names are reported in.
 *
 * Entries name the module root they live in (<domain>/kernels.cu), so one table may span modules; each distinct module
 * is compiled and loaded once, on the first query of any of the table's kernels. */
typedef struct {
    const char* name;
    unsigned int max_dynamic_shared_bytes;
    const char* module; /* the module root the kernel is compiled from */
    /* Nonzero: the module uses architecture-specific features (sm_12xa). On another device its kernels are
     * unavailable (euhedral_kernel_table_available) and the rest of the table still loads. Every entry of a module
     * gives the same value. */
    int architecture_specific;
} EuhedralKernelEntry;

typedef struct {
    const EuhedralKernelEntry* entries;
    int count;
    /* Filled by euhedral_kernel_table_initialize, which runs once per table. */
    CUfunction* functions;
    int init_status;
} EuhedralKernelTable;

#define EUHEDRAL_KERNEL_TABLE_INIT(entries_array) \
    { (entries_array), (int)(sizeof(entries_array) / sizeof((entries_array)[0])), NULL, \
      EUHEDRAL_CUDA_KERNEL_UNAVAILABLE }

/* Loads every module, resolves every function and sets the dynamic shared limits. Call it only through a ready
 * function made by EUHEDRAL_KERNEL_TABLE_READY. */
void euhedral_kernel_table_initialize(EuhedralKernelTable* table);

/* 1 when kernel `kernel` is loaded, 0 when its module is architecture-specific and does not apply to this device,
 * or a negative status. */
int euhedral_kernel_table_available(EuhedralKernelTable* table, int (*ready)(void), int32_t kernel);

/* Defines `static int function(void)`: initialize `table` once and return its status. */
#ifdef _WIN32
#define EUHEDRAL_KERNEL_TABLE_READY(function, table) \
    static INIT_ONCE function##_once = INIT_ONCE_STATIC_INIT; \
    static BOOL CALLBACK function##_run(PINIT_ONCE state, PVOID parameter, PVOID* context) { \
        (void)state; (void)parameter; (void)context; \
        euhedral_kernel_table_initialize(&(table)); \
        return TRUE; \
    } \
    static int function(void) { \
        if (!InitOnceExecuteOnce(&function##_once, function##_run, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; \
        return (table).init_status; \
    }
#else
#define EUHEDRAL_KERNEL_TABLE_READY(function, table) \
    static pthread_once_t function##_once = PTHREAD_ONCE_INIT; \
    static void function##_run(void) { euhedral_kernel_table_initialize(&(table)); } \
    static int function(void) { \
        if (pthread_once(&function##_once, function##_run) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; \
        return (table).init_status; \
    }
#endif

#endif

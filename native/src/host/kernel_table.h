#ifndef EUHEDRAL_KERNEL_TABLE_H
#define EUHEDRAL_KERNEL_TABLE_H

#include "cuda_kernel_loader.h"
#include "submission.h"
#include "euhedral_cuda.h"
#include <cuda.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* A table of kernels behind one launcher. A launch passes the kernel's index, its geometry and its arguments as 8-byte
 * words with the byte size (4 or 8) each argument has in the kernel's signature. The launcher checks the argument count
 * and every size against the compiled kernel (cuFuncGetParamInfo), so a mismatch between the Java caller and the CUDA
 * source fails the call instead of corrupting it. A table's entry order is its ABI: the Java enum of the same table
 * lists the same names in the same order and a test compares them.
 *
 * Entries name the module root they live in (<domain>/kernels.cu), so one table may span modules; each distinct module
 * is compiled and loaded once, on the first launch of any of the table's kernels. */
#define EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS 32

typedef struct {
    const char* name;
    unsigned int max_dynamic_shared_bytes;
    const char* module; /* the module root the kernel is compiled from */
} EuhedralKernelEntry;

typedef struct {
    const EuhedralKernelEntry* entries;
    int count;
    /* Filled by euhedral_kernel_table_initialize, which runs once per table. */
    CUfunction* functions;
    int* parameter_count;
    uint8_t* parameter_size; /* count x EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS */
    int init_status;
} EuhedralKernelTable;

#define EUHEDRAL_KERNEL_TABLE_INIT(entries_array) \
    { (entries_array), (int)(sizeof(entries_array) / sizeof((entries_array)[0])), NULL, NULL, NULL, \
      EUHEDRAL_CUDA_KERNEL_UNAVAILABLE }

/* Loads every module, resolves every function, sets the dynamic shared limits and records every parameter size. Call
 * it only through a ready function made by EUHEDRAL_KERNEL_TABLE_READY. */
void euhedral_kernel_table_initialize(EuhedralKernelTable* table);

/* Checks the arguments and launches kernel `kernel` on the submission stream (or synchronously when there is none).
 * `ready` returns the table's initialization status, running the initialization on its first call. */
int euhedral_kernel_table_launch(EuhedralKernelTable* table, int (*ready)(void), int32_t kernel,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
        uint32_t shared_bytes, const uint64_t* words, const uint8_t* sizes, int32_t count);

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

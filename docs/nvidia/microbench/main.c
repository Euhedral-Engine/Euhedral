#include "common.h"

int cmd_attrs(int, char **);
int cmd_mem(int, char **);
int cmd_mma(int, char **);
int cmd_alu(int, char **);
int cmd_tex(int, char **);
int cmd_pcie(int, char **);
int cmd_l2(int, char **);
int cmd_misc(int, char **);

static const struct {
    const char *name;
    int (*fn)(int, char **);
    const char *what;
} k_cmds[] = {
    {"attrs", cmd_attrs, "device attributes and feature flags"},
    {"mem", cmd_mem, "DRAM/L2 bandwidth, bytes in flight, TMA bulk copy, latency"},
    {"mma", cmd_mma, "tensor-core throughput per mma.sync kind"},
    {"alu", cmd_alu, "FP32/INT32/FP16/SFU/convert throughput per SM per clock"},
    {"tex", cmd_tex, "texture path: TEX vs LDG, block-compressed (BC) decode"},
    {"pcie", cmd_pcie, "host<->device copies, copy engines, zero-copy reads"},
    {"l2", cmd_l2, "persisting L2 window, eviction hints, compressible memory"},
    {"misc", cmd_misc, "clusters/DSMEM, launch and graph overheads"},
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <command> [args]\n", argv[0]);
        for (size_t i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
            fprintf(stderr, "  %-6s %s\n", k_cmds[i].name, k_cmds[i].what);
        return 2;
    }
    bench_init();
    for (size_t i = 0; i < sizeof k_cmds / sizeof k_cmds[0]; i++)
        if (!strcmp(argv[1], k_cmds[i].name)) return k_cmds[i].fn(argc - 1, argv + 1);
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}

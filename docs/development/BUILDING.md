# Building and testing

Run the commands below from the repository root. For running a server, start with the
[quick start](../../README.md#quick-start).

## Toolchain

[mise.toml](../../mise.toml) pins the development tools:

| Tool | Version |
|---|---|
| Rust | 1.99.1 |
| Zig | 0.16.0 |
| cargo-zigbuild | 0.23.4 |

With mise installed, run `mise install` to install the toolchain. The native build resolves the pinned CUDA
runtime, NVRTC, and CUDA headers automatically. The Rust workspace is in [rust/](../../rust/README.md); it links
llguidance for structured output as a crate.

## Build

```bash
cd rust
cargo xtask native --product linux-x64     # the CUDA library and kernel sources, into build/native/linux-x64
cargo xtask bundle --product linux-x64     # the server bundle, into build/bundle
cargo xtask native --product windows-x64   # the Windows product
cargo xtask bundle --product windows-x64   # the Windows bundle (on Windows, or cross-built with cargo zigbuild)
```

A bundle holds `bin/euhedral-server` and the native libraries it loads, so serving requires only the NVIDIA driver.
See the [native module guide](../../native/README.md) for native build details and
[rust/README.md](../../rust/README.md#server-bundle) for the bundle layout.

## Test

```bash
mise run rust-check        # layer check, formatting, clippy and the CPU Rust tests
mise run native-test       # the Python kernel tests
```

The first task checks the Rust layers, formatting and lints, then runs the CPU tests (`mise run rust-test` runs
only the tests). The Python kernel tests require a GPU and NumPy. The device tests of the Rust crates need model
artifacts; stop any serving container before running them.

See [testing](../evaluation/TESTING.md) for test groups, artifact setup, and selecting individual device tests.
Use the [benchmarking guide](../evaluation/BENCHMARKING.md) to measure performance.

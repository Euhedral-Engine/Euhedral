# Rust workspace

The Rust crates of Euhedral-Inference. `mise run rust-check` runs the layer check, formatting, clippy (`-D warnings`)
and the CPU tests; CI runs the same on Linux and Windows (the layer check on Linux).

## Architecture rules

The rules are enforced by the compiler toolchain, not by review. Each is checked in one of three places:

- **clippy ban**: `clippy-no-wait.toml` lists the banned calls and types. Every library crate has a `clippy.toml` that
  is byte-identical to it. A library crate is every crate except `xtask` and the binaries (`bench`), whose main thread
  may block. `test-support` counts as a library: it keeps the same file and excepts itself in its source.
- **lint**: `[workspace.lints]` in `Cargo.toml`, inherited by every crate through `[lints] workspace = true`.
- **layer check**: `scripts/check-layers.sh` reads `cargo metadata` (no build). `--self-test` plants a violation of each
  kind and fails unless the check catches it. Normal and build dependencies count in every rule, in every target
  section; dev-dependencies are ignored. The crate table at the top of the script is the single place the layers
  are written down; a crate that is not in it fails the check until it is placed.

| Rule | Enforced by |
| --- | --- |
| Threads belong to the lattice: no `thread::spawn`, `Builder::spawn`, `scope`, `spawn_scoped`, `JoinHandle::join` | clippy ban |
| Nothing waits or spins: no `sleep`, `park`, `park_timeout`, `yield_now`, `spin_loop`, `Condvar`, `Barrier`, `Child::wait`, `Child::wait_with_output`, `Command::output`, `Command::status`, and RustEE's `spin_wait::await_while` and `await_termination` | clippy ban |
| No channels: no `mpsc` (`channel`, `sync_channel`, `Sender`, `SyncSender`, `Receiver`) | clippy ban |
| No locks: no `Mutex` or `RwLock`; state is owned by a sink or a source and polled one frame at a time | clippy ban |
| No blocking I/O: no `std::fs::{read, read_to_string, write, copy, read_dir, metadata}`, `File::open`, `File::create`, `OpenOptions::open`, `io::stdin`, `TcpStream`, `TcpListener::accept` | clippy ban |
| No waiting CUDA calls: the public wrappers (`CudaLibrary::synchronize`, `CudaLane::synchronize`, `DeviceBuffer::copy_from_host` and `copy_to_host`, `PinnedPool::trim`, `ops::copy_host_to_device` and `copy_device_to_host`) in every crate that links cuda, and inside cuda the generated `EuhedralCuda` methods behind them (synchronize, synchronous copies, `cudaFree`-style frees). A call through a generated function-pointer field is invisible to clippy: give it the same scoped allow | clippy ban (`allow-invalid`: active where the cuda crate is linked) |
| The environment is read in one place: no `std::env::var`, `var_os`, `vars`, `vars_os` outside the settings crate and a binary's `main` | clippy ban |
| No fully qualified names: import the type (`use std::mem;` then `mem::swap(&mut a, &mut b)`, not `std::mem::swap(..)`); a path of more than two segments from a crate root is flagged | `absolute_paths` lint, `absolute-paths-max-segments = 2` in `clippy.toml` (`use` items are not checked) |
| Libraries do not print | `print_stdout`, `print_stderr` lints |
| No `dbg!`, `todo!`, `unwrap()` or `expect()` in library code (tests may unwrap and expect) | `dbg_macro`, `todo`, `unwrap_used`, `expect_used` lints; `allow-unwrap-in-tests`, `allow-expect-in-tests` |
| Every `unsafe` block states why it is sound | `undocumented_unsafe_blocks` lint |
| Dependencies point one way (settings, edrl, host-gemv, test-support: none; tokenizer: edrl; cuda: none; engine: all of them; bench: engine and the leaves) | layer check |
| Configuration is parsed in one place: only the settings crate depends on `serde_yaml_ng` | layer check |
| Only the engine, bench and cuda depend on `euhedral-core` or `euhedral-graph`; cuda depends on nothing of ours and externally only on `euhedral-graph`, `libloading` and `thiserror` | layer check |
| No crate depends on `level-zero*`, `opencl3`, `ocl` (hardware agnosticism) or `core_affinity` (worker CPUs are never pinned) | layer check |
| `test-support` is a dev-dependency only | layer check |
| Every library crate's `clippy.toml` equals `clippy-no-wait.toml` | layer check |

### Making a justified exception

An exception is a scoped `#[allow]` on the smallest item that needs it, with a `reason` that names the rule:

```rust
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: the configuration file is read once at startup, before the lattice exists"
)]
fn read_file(file: &Path) { /* ... */ }
```

- A call that has no non-blocking form (startup, shutdown, a test fixture) gets the allow where it is called, never a
  crate-wide one. A test file may carry a file-level allow for its fixtures.
- Never edit one crate's `clippy.toml` to loosen it: the layer check fails on any difference. To change a ban, change
  `clippy-no-wait.toml` and copy it to every library crate.
- A new library crate gets a copy of `clippy-no-wait.toml` as its `clippy.toml` and `[lints] workspace = true`, and a
  row in the table of `scripts/check-layers.sh`. A new binary is added to `BINARIES` there.
- A layer or external-dependency exception is a change to the table in `scripts/check-layers.sh`, reviewed as a change
  to the architecture.

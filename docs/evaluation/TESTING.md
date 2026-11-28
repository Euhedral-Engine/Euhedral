# Testing

For toolchain setup and build commands, see [building and testing](../development/BUILDING.md).

## Host tests

`mise run rust-test` (`cargo test --workspace` in `rust/`) runs the CPU tests of every crate. They need no GPU.
`mise run rust-check` adds the layer check, formatting and clippy. A test that reads a file makes it with `tempfile`:
CI runs every test under Bazel's sandbox, where a crate's own files are not on disk.

## Device tests

A test that needs the GPU, the native library or a model is marked `#[ignore]` and skips with a message when its
setting is missing. Run one binary or one test by name with `--ignored`:

```bash
cd rust
cargo test --release -p euhedral-inference-engine --test device -- --ignored --nocapture <test name>
```

Settings are looked up in this order: `EUHEDRAL_TEST_<KEY>` in the environment (`euhedral.qwen.rco-artifact` is
`EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`), then `~/.config/euhedral/test.properties` (or the file named by
`EUHEDRAL_TEST_CONFIG`), then `gradle.properties` in the Gradle user home (`~/.gradle/gradle.properties`), kept as a fallback for existing setups. The native library is named by `euhedral.cuda.library` (`EUHEDRAL_TEST_CUDA_LIBRARY`), normally the `libeuhedral_cuda.so` that `cargo xtask native` builds under `build/native/linux-x64/lib/`.

CI builds the device tests once and runs them in groups with `scripts/ci-device.py` (`pack` writes the archive,
`run GROUP` runs one group, `check` fails unless every ignored test is in exactly one group). The groups keep the tests
that load a model together, so an artifact loads once per group, and keep the Flash-Next groups apart, because their
whole-model tests pin host memory sized from what the cgroup has left.

Stop any serving container before running a device test.

### Keeping the suite fast

- Load once, then reuse; an artifact load is several seconds.
- Time before cutting: a scalar CPU reference over the whole model, or a sweep over many row counts, costs far more
  than the check it supports. Keep a representative case for each boundary the test is about, and say in a comment
  why the reduced case still covers it.
- A deadline in a test is hang detection: about ten times the expected runtime.

# Building and testing

Run the commands below from the repository root. For running a server, start with the
[quick start](../../README.md#quick-start).

## Toolchain

[mise.toml](../../mise.toml) pins the development tools:

| Tool | Version |
|---|---|
| Java | 25.0.2 |
| Gradle | 9.6.1; wrapper included |
| Zig | 0.16.0 |
| Rust | 1.95.0 |
| cargo-zigbuild | 0.23.4 |

With mise installed, run `mise install` to install the toolchain. The build resolves the pinned CUDA
runtime, NVRTC, and CUDA headers automatically. Rust and cargo-zigbuild build llguidance for structured output.

## Build

```bash
./gradlew :api:bootJar             # JAR for the current platform
./gradlew :api:bootJarWindowsX64   # Windows x64 JAR
./gradlew build                   # full build, formatting checks, CPU tests, native products
```

The server JARs appear in `api/build/libs/`. They package their native libraries, so serving requires
Java and the NVIDIA driver. See the [native module guide](../../native/README.md) for native build details.

## Test

```bash
./gradlew test
python3 -m unittest discover -s native/tests -p 'test_*.py'
./gradlew :core:cudaTest :api:cudaTest
```

The first command runs CPU tests. The Python kernel tests require a GPU and NumPy. The CUDA suites
also need model artifacts; stop any serving container before running them.

See [testing](../evaluation/TESTING.md) for test groups, artifact setup, and selecting individual CUDA tests.
Use the [benchmarking guide](../evaluation/BENCHMARKING.md) to measure performance.

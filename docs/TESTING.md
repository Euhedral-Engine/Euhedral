# Testing

## Host tests

`./gradlew test` runs the CPU tests of every module. They need no GPU. Tests that touch process-wide state
(the Euhedral lattice, system properties, ports) are serialized or run in a JVM of their own; the rest run in
parallel.

## CUDA tests

`./gradlew :core:cudaTest :api:cudaTest` runs every CUDA test, in three steps:

1. **Light** (`cudaLightTest`): classes that load no artifact and use little device memory. Several JVMs run them at
   once (`-PcudaForks=N`, default 4).
2. **Model groups** (`cudaModelTest`): classes that load an artifact, directly or through an engine. All of them run
   in **one JVM**, ordered by group, and each artifact is loaded **once**: the first class that needs it loads it,
   the rest find it ready, and the next group closes it before loading its own (two models do not fit the GPU).
3. **Own JVM** (`cudaOwnJvmTest`): classes that cannot share a process, one JVM each.

A class tells which kind it is with an annotation from `ModelGroup` (core test sources, `core/testing`), which is also
a JUnit tag:

| Annotation | The class |
| --- | --- |
| none | needs no artifact; runs in the light step |
| `@ModelGroup.CompactQ3`, `@ModelGroup.Nvfp4`, `@ModelGroup.FlashNext` | works on that artifact's model directly (`SharedQwen38.q3()`, `nvfp4()`) |
| `@ModelGroup.CompactQ3Engine`, `Nvfp4Engine`, `FlashNextEngine` | starts an `InferenceEngine` over that artifact (`SharedEngines`) |
| `@ModelGroup.OwnJvm` | needs a fresh process; say why in a comment |

The `api` module has no access to core's test classes: its CUDA classes carry the same names as plain
`@Tag("engine-q3")` and so on.

### Sharing state

`Shared` keeps JVM-wide values (the device handle, a model, an engine), creates each on first use and closes them
after the last test. Asking for a value of another group closes the previous group's values first. A test:

- takes what it needs from the shared fixtures instead of loading in each method,
- frees what it allocates and restores any toggle in a `finally`,
- reads counters as differences (an engine is not fresh when the test starts),
- never closes a shared value.

Tests of one class that need different engines are ordered by engine (`@TestMethodOrder`) so the engine changes
once.

### Running one class

`./gradlew :core:cudaIntegrationTest --tests '*FullModelCudaIntegrationTest'` runs the selected classes in one JVM,
whatever their kind.

### Keeping the suite fast

- Load once, then reuse; an artifact load is several seconds.
- Time before cutting: a scalar CPU reference over the whole model, or a sweep over many row counts, costs far more
  than the check it supports. Keep a representative case for each boundary the test is about, and say in a comment
  why the reduced case still covers it.
- A `@Timeout` is hang detection: about ten times the expected runtime.

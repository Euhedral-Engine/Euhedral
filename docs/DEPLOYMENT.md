# Docker deployment

Run these commands from the repository root. For a Java deployment, use the
[quick start](../README.md#quick-start).

## Build and run

The container takes its settings from a `.env` file: copy [.env.example](../.env.example) to `.env` (git ignores it) and
set the artifact path, model name, and port. Adjust `EUHEDRAL_HOST_CPUS` for your machine, or remove it to let
the server select its workers.

```bash
docker build -t euhedral-inference:local .
set -a; . ./.env; set +a
docker run --rm --gpus all --env-file .env -p "$PORT:$PORT" --stop-timeout 45 \
  --mount type=bind,src="$EUHEDRAL_ARTIFACT_FILE",dst=/models/model.edrl,readonly \
  euhedral-inference:local
```

Install the NVIDIA Container Toolkit and a compatible host driver to use `--gpus all`. The model file is mounted
from the host.

## Hosts without the NVIDIA Container Toolkit

You can pass the GPU device nodes and driver libraries explicitly:

```text
--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-modeset \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/opt/euhedral/lib/libcuda.so.1,readonly \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1,dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly
```

## Deploy from main

`mise run deploy` reads the same `.env`. It pulls `origin/main`, builds an image from exactly that commit, switches the
`euhedral-inference-serve` container (published on `127.0.0.1:$PORT`) after checking Docker health, `/health`,
`/v1/models` and a one-token completion, and restores the previous container if the candidate fails. With
`EUHEDRAL_CUDA_DRIVER_FILE` and `EUHEDRAL_PTX_JIT_FILE` set it passes the device nodes and those libraries instead of
`--gpus all`.

The [operations guide](OPERATIONS.md) covers environment overrides, health checks, metrics, and shutdown.

# Docker deployment

Run deployment from a clean, committed `main` checkout. Pass one host YAML or Docker-format `.env` file:

```bash
mise deploy euhedral.yaml
# or
mise deploy .env
```

The task reads everything it needs from that file. It pulls `origin/main`, builds an image from that commit,
mounts the model and configuration, and starts the service on `127.0.0.1` at the configured port. No separate
image build or mount arguments are needed. Mise manages the Python runner and YAML parser dependencies.
The host needs a compatible NVIDIA driver. Deploy checks for a Docker NVIDIA runtime or NVIDIA CDI devices;
when neither is available, it discovers the host driver libraries and mounts them with the GPU device nodes.
This also works with YAML, without additional configuration or NVIDIA Container Toolkit installation.

## YAML configuration

Use [the example configuration](../config/euhedral.example.yaml). `model.artifact` is the model file's **host path**;
relative paths resolve from the YAML file's directory. For example:

```yaml
model:
  artifact: /path/to/model.edrl
  name: qwen
server:
  port: 1738
```

Deploy mounts the YAML read-only at `/etc/euhedral/custom.yaml` and the model at `/models/model.edrl`, then
passes `--config /etc/euhedral/custom.yaml` to the server. It overrides the artifact path inside the container
without editing your YAML. `model.name` supplies the public model ID used to verify the deployment.
`server.port` selects the listening, published, and health-check port (default `1738`). All other engine settings
come from the mounted YAML. The Java application continues to read YAML and environment variables.

## Environment file

Copy [.env.example](../.env.example) to `.env` and fill it in. Set `EUHEDRAL_ARTIFACT_FILE` (or
`EUHEDRAL_MODEL_ARTIFACT`) to the host model path and `EUHEDRAL_MODEL_NAME` to the public model name.
Relative host paths resolve from the `.env` file's directory. `EUHEDRAL_SERVER_PORT` takes precedence over
`PORT`, defaulting to `1738`. `EUHEDRAL_HOST_CPUS` is optional; omit it to select workers automatically.

Deploy passes the file to Docker using its literal `KEY=VALUE` format; it never sources it as a shell script.
It mounts the model and sets the artifact path and health-check port inside the container automatically.

To override automatic driver discovery, set both `EUHEDRAL_CUDA_DRIVER_FILE` and `EUHEDRAL_PTX_JIT_FILE`
in `.env`. Deploy mounts those libraries and passes the NVIDIA device nodes instead of using `--gpus all`.

## Cutover and rollback

The existing service stays up while the image builds. Deploy switches the `euhedral-inference-serve` container
only after the build succeeds, then checks Docker health, `/health`, `/v1/models`, and a chat completion.
It verifies the public model ID against the configuration and restores the previous container if the candidate
fails. Old Euhedral deployment containers and images are removed after the new service passes those checks.

The [operations guide](OPERATIONS.md) covers environment overrides, health checks, metrics, and shutdown.

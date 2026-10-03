#!/usr/bin/env python3
"""Pull, build, and deploy main; clean up only after a healthy cutover.

Run on the deployment host from any directory. Override EUHEDRAL_DEPLOY_MODEL,
EUHEDRAL_DEPLOY_TOKENIZER, EUHEDRAL_DEPLOY_DRIVER, EUHEDRAL_DEPLOY_PTX,
and EUHEDRAL_DEPLOY_PORT if this host's paths or port change.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from contextlib import contextmanager


SERVICE = "euhedral-inference-serve"
IMAGE_REPOSITORY = "euhedral-inference"
REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_ID = os.environ.get("EUHEDRAL_DEPLOY_MODEL_ID", "qwen3.8-27b-q3")


def command(*args):
    print("+", " ".join(str(arg) for arg in args), flush=True)
    result = subprocess.run(args, cwd=REPOSITORY, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    if args[:2] in (("git", "pull"), ("docker", "build")):
        print(result.stdout[-3000:], flush=True)
    return result.stdout.strip()


def get_json(url, data=None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30 if data is not None else 5) as response:
        return json.load(response)


def verify_ready(port):
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            health = command("docker", "inspect", "--format", "{{.State.Health.Status}}", SERVICE)
            status = get_json(base + "/health")
            models = get_json(base + "/v1/models").get("data", [])
            if health == "healthy" and status.get("status") == "up" and status.get("engine") == "ready" and models:
                model = models[0]["id"]
                response = get_json(
                    base + "/v1/chat/completions",
                    json.dumps({"model": model, "messages": [{"role": "user", "content": "Say READY"}], "max_tokens": 1, "temperature": 0}).encode(),
                )
                choices = response.get("choices") or []
                usage = response.get("usage") or {}
                if usage.get("completion_tokens", 0) > 0 and any(
                    isinstance(choice.get("message", {}).get("content"), str)
                    and choice["message"]["content"].strip() for choice in choices
                ):
                    return model
                last_error = "chat smoke response had no generated text"
            else:
                last_error = f"Docker={health}, HTTP={status}, models={len(models)}"
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            last_error = str(error)
        time.sleep(3)
    raise RuntimeError(f"{SERVICE} did not become ready: {last_error}")


@contextmanager
def build_context(repository):
    """Build exactly the pulled commit, never the untracked benchmark archive."""
    with tempfile.TemporaryDirectory(prefix="euhedral-deploy-", dir=os.environ.get("TMPDIR", str(repository / "benchmark-results"))) as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as output:
            subprocess.run(("git", "archive", "--format=tar", "HEAD"), cwd=repository, stdout=output, check=True)
        context = Path(temporary) / "source"
        context.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(context, filter="data")
        archive.unlink()
        yield context


def deploy(repository, model, tokenizer, driver, ptx, port):
    if command("git", "branch", "--show-current") != "main":
        raise RuntimeError("Deploy only from main; switch to main before invoking this script")
    if command("git", "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked working-tree changes must be committed or moved before deployment")
    for path in (model, driver, ptx):
        if not path.is_file():
            raise RuntimeError(f"Missing required host file: {path}")
    if not tokenizer.is_dir():
        raise RuntimeError(f"Missing tokenizer directory: {tokenizer}")
    for device in ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-modeset"):
        if not Path("/dev", device).exists():
            raise RuntimeError(f"Missing NVIDIA device: /dev/{device}")
    if not 1 <= port <= 65535:
        raise RuntimeError("Invalid HTTP port")

    command("git", "pull", "--ff-only", "origin", "main")
    revision = command("git", "rev-parse", "HEAD")
    if revision != command("git", "rev-parse", "origin/main"):
        raise RuntimeError("Local main does not match origin/main")
    image = f"{IMAGE_REPOSITORY}:{revision[:7]}"
    names = command("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    previous = SERVICE in names
    previous_name = None
    previous_image_id = None
    if previous:
        previous_id = command("docker", "inspect", "--format", "{{.Id}}", SERVICE)
        previous_image_id = command("docker", "inspect", "--format", "{{.Image}}", SERVICE)
        previous_name = f"{SERVICE}-previous-{previous_id[:12]}"
        if previous_name in names:
            raise RuntimeError(f"Rollback container already exists: {previous_name}")
    print(f"Building {image} from {revision}; the current service stays up until the build succeeds", flush=True)
    with build_context(repository) as source:
        command("docker", "build", "--label", f"org.opencontainers.image.revision={revision}", "-t", image, str(source))
    new_image_id = command("docker", "image", "inspect", "--format", "{{.Id}}", image)
    old_tags = [
        tag for tag in command("docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}").splitlines()
        if tag.startswith(IMAGE_REPOSITORY + ":") and tag != image
    ]
    old_image_ids = {command("docker", "image", "inspect", "--format", "{{.Id}}", tag) for tag in old_tags}
    if previous_image_id:
        old_image_ids.add(previous_image_id)
    old_stopped = False
    old_renamed = False
    try:
        if previous:
            command("docker", "stop", SERVICE)
            old_stopped = True
            command("docker", "rename", SERVICE, previous_name)
            old_renamed = True
        command(
            "docker", "run", "-d", "--name", SERVICE, "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{port}:{port}", "-e", f"PORT={port}",
            "-e", f"EUHEDRAL_INFERENCE_MODEL_ID={MODEL_ID}",
            "--device", "/dev/nvidia0", "--device", "/dev/nvidiactl",
            "--device", "/dev/nvidia-uvm", "--device", "/dev/nvidia-modeset",
            "--mount", f"type=bind,src={driver},dst=/opt/euhedral/lib/libcuda.so.1,readonly",
            "--mount", f"type=bind,src={ptx},dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly",
            "--mount", f"type=bind,src={model},dst=/models/model.edrl,readonly",
            "--mount", f"type=bind,src={tokenizer},dst=/tokenizer,readonly",
            image,
        )
        model_id = verify_ready(port)
        if model_id != MODEL_ID:
            raise RuntimeError(f"Unexpected public model ID {model_id!r}, expected {MODEL_ID!r}")
        if command("docker", "image", "inspect", "--format", "{{.Id}}", image) != new_image_id:
            raise RuntimeError("Deployment image changed before cleanup")
        if command("docker", "inspect", "--format", "{{.State.Health.Status}}", SERVICE) != "healthy":
            raise RuntimeError("Deployed container became unhealthy before cleanup")
    except BaseException as error:
        if old_stopped:
            print(f"Deployment failed ({error}); restoring the previous service", file=sys.stderr, flush=True)
            try:
                if old_renamed:
                    try:
                        command("docker", "rm", "-f", SERVICE)
                    except subprocess.CalledProcessError:
                        pass  # Candidate never reached container creation.
                    command("docker", "rename", previous_name, SERVICE)
                command("docker", "start", SERVICE)
                verify_ready(port)
                print("Previous service restored and verified", file=sys.stderr, flush=True)
            except BaseException as rollback_error:
                raise RuntimeError(f"Deployment failed: {error}; ROLLBACK FAILED: {rollback_error}") from error
        raise

    print(f"Healthy candidate {revision} serving {model_id}; cleaning old Euhedral containers/images", flush=True)
    failures = []
    for name in command("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        stale_deployment = name.startswith(SERVICE + "-")
        experimental_probe = name.startswith("euhedral-") and name.endswith("-probe")
        owned_probe = experimental_probe and command("docker", "inspect", "--format", "{{.Image}}", name) in old_image_ids
        if stale_deployment or owned_probe:
            try:
                command("docker", "rm", name)  # Never force-stop an unexpected running container.
            except subprocess.CalledProcessError as error:
                failures.append(f"container {name}: {error.output[-300:]}")
    for tag in old_tags:
        try:
            command("docker", "image", "rm", tag)
        except subprocess.CalledProcessError as error:
            failures.append(f"image {tag}: {error.output[-300:]}")
    if previous_image_id and previous_image_id != new_image_id:
        try:
            command("docker", "image", "inspect", "--format", "{{.Id}}", previous_image_id)
        except subprocess.CalledProcessError:
            pass  # Deleting its last tag already removed the old image.
        else:
            try:
                command("docker", "image", "rm", previous_image_id)
            except subprocess.CalledProcessError as error:
                failures.append(f"previous image {previous_image_id}: {error.output[-300:]}")

    if failures:
        raise RuntimeError("Deployed service is healthy, but cleanup was incomplete: " + "; ".join(failures))
    print(f"Deployed {revision} ({model_id}) and removed old Euhedral deployments and experiments", flush=True)
    return revision


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    model_path = Path(os.environ.get("EUHEDRAL_DEPLOY_MODEL", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl"))
    tokenizer_path = Path(os.environ.get("EUHEDRAL_DEPLOY_TOKENIZER", "/mnt/shared/qwen38-quant/source/qwen"))
    driver_path = Path(os.environ.get("EUHEDRAL_DEPLOY_DRIVER", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"))
    ptx_path = Path(os.environ.get("EUHEDRAL_DEPLOY_PTX", "/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"))
    try:
        deploy(root, model_path, tokenizer_path, driver_path, ptx_path, int(os.environ.get("EUHEDRAL_DEPLOY_PORT", "18080")))
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.output[-4000:], file=sys.stderr)
        sys.exit(1)

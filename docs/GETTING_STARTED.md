# Getting started

The [quick start](../README.md#quick-start) shows how to build and run the server with an `.edrl` model file.
This guide covers preparing that file, connecting clients, and using Windows.

## Convert a checkpoint

Start with a Hugging Face checkpoint of Qwen3.8-27B or Swift-1.5 (`ukisai/Swift-1.5-Qwen3.8-27b`).
The converter needs Python and NumPy; GPU conversion also needs PyTorch with CUDA support.

To create a Q3 artifact:

```bash
python3 tools/convert_checkpoint.py --model /path/to/Qwen3.8-27B --quantization q3 \
    --ranking /path/to/token-ranking.i64 --out /models/qwen-q3.edrl
```

`--ranking` takes a token-frequency file used to prepare the draft head. If you already have an artifact,
you can reuse its token selection with `--draft-ids-from /models/existing.edrl` instead.
The [converter guide](../tools/README.md) covers the input files, compressed formats, and calibrated conversion.

The resulting `.edrl` file includes the tokenizer and chat template. Use its path as `model.artifact`
in your server configuration. Older artifacts may need their tokenizer added; see
[upgrading an artifact](OPERATIONS.md#artifacts-and-their-tokenizer).

For model sizes and measured tradeoffs, see [choosing an artifact](../README.md#choosing-an-artifact).

## Connect a client

The examples below use the model name `qwen`, matching the quick-start configuration.
API keys are not checked; clients that require one can use `unused`.

### OpenAI

Use `http://localhost:1738/v1` as the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1738/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences."}],
    stream=True,
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

The [Responses API guide](RESPONSES_API.md) has examples for `client.responses.create`.

### Anthropic

Use `http://localhost:1738` as the base URL:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:1738", api_key="unused")
message = client.messages.create(
    model="qwen", max_tokens=512, messages=[{"role": "user", "content": "Name three prime numbers."}]
)
print(message.content[-1].text)
```

See the [Messages API guide](ANTHROPIC_API.md) for streaming and request options.

## Windows

Build the Windows JAR with:

```bash
./gradlew :api:bootJarWindowsX64
```

On Windows, set `model.artifact` to the artifact's Windows path in `euhedral.yaml`, then run:

```powershell
java --enable-native-access=ALL-UNNAMED -jar api/build/libs/euhedral-inference-windows-x64.jar --config euhedral.yaml
```

Both platform JARs include the native libraries they need. Install Java 25 and a current NVIDIA driver
on the serving machine. Build tool requirements are in [building and testing](development/BUILDING.md).

## Configure the server

The server listens on port 1738 by default. Start with [the example YAML](../config/euhedral.example.yaml)
to change the port, sampling defaults, context length, or concurrent requests. Settings can also be
overridden through environment variables; the [operations guide](OPERATIONS.md) explains the names and defaults.

To run in a container, follow [Docker deployment](DEPLOYMENT.md).

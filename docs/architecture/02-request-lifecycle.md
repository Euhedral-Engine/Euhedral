# Request lifecycle

This diagram follows an API generation from HTTP admission to response completion. The engine steps shown are the dense Qwen3.8 path. Planning occurs before the service reserves a generation slot.

```mermaid
sequenceDiagram
    participant C as Client
    participant HTTP as Controller
    participant P as ConversationPlanner on workers
    participant S as GenerationService
    participant B as EngineInferenceBackend
    participant G as Session / generation frames
    participant X as Execution / CUDA
    participant R as ChatGeneration / responder
    C->>HTTP: Chat, Responses or Messages request
    HTTP->>P: Map, render, encode and validate
    HTTP-->>C: Deferred response pending
    P->>S: Planned request
    alt Active slot available
        S->>B: openGeneration
    else Queue has room
        S->>S: Wait until a running request releases a slot
        S->>B: openGeneration
    else Capacity exhausted or shutting down
        S-->>C: HTTP 503
    end
    Note over B,R: Remaining flow applies to an admitted generation
    B->>G: generateAsync with encoded prompt
    opt Streaming request
        S-->>C: Publish SSE emitter after admission
    end
    opt Prefix cache hit
        G->>X: Restore prompt checkpoint
    end
    loop Remaining prompt chunks
        G->>X: PREFILL quantum
        X-->>G: Retired quantum / Select continuation
    end
    loop Until EOS, limit, cancellation or failure
        alt Speculation enabled
            G->>X: Draft then VERIFY
            X-->>G: Accepted outputs and committed state
        else Ordinary decoding
            G->>X: DECODE
            X-->>G: Logits for selection
        end
        G->>R: Generated text callback
        R-->>C: SSE events, or collect JSON output
    end
    G-->>R: Generation result after Finish
    R-->>C: Final response or terminal stream events
    R->>S: Close generation and release slot
    S->>S: Start next queued request
```

`GenerationService` bounds active requests by `maxActiveGenerations()` and waiting requests by `maxQueuedGenerations`. Validation failures reach the HTTP error handler before an emitter is published. Client disconnects, request timeouts and shutdown stop generation or remove the waiting admission; completion releases the session and slot.

The core generation chain is `Admit → quantum → Select → Admit/Finish`. A session can skip a quantum when a step needs no device work. The first output can come from prompt logits; the diagram groups that selection with generation rather than implying an extra decode is always required.

## Implementation anchors

- [Controllers](../../api/src/main/java/io/euhedral_execution/inference/api/web/OpenAiController.java) and [AnthropicController](../../api/src/main/java/io/euhedral_execution/inference/api/web/AnthropicController.java).
- [ConversationPlanner](../../api/src/main/java/io/euhedral_execution/inference/api/chat/ConversationPlanner.java), [GenerationService](../../api/src/main/java/io/euhedral_execution/inference/api/chat/GenerationService.java) and [ChatGeneration](../../api/src/main/java/io/euhedral_execution/inference/api/chat/ChatGeneration.java).
- [EngineInferenceBackend](../../api/src/main/java/io/euhedral_execution/inference/api/engine/EngineInferenceBackend.java): constraints, session creation and callbacks.
- [Generation](../../core/src/main/java/io/euhedral_execution/inference/core/generation/Generation.java) and [Session](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/Session.java).

This diagram omits protocol-specific event schemas, tool parsing and individual execution stages. See [execution engine](03-execution-engine.md) and [speculative decoding](05-speculative-decoding.md).

# Reasoning

Qwen3.8 thinks before it answers: the assistant turn opens a `<think>` block, the model writes its reasoning, closes the
block with `</think>` and then writes the answer. The checkpoint's chat template decides whether that block is open for
generation and what it tells the model about effort. The server exposes exactly those controls and reports the reasoning
apart from the answer.

## Controls

| `reasoning_effort` | Template variables | What the model is told |
|---|---|---|
| omitted | `enable_thinking` undefined, `reasoning_effort` undefined (template default `xhigh`) | the `xhigh` instructions below |
| `none` | `enable_thinking=false` | nothing; the generation prompt closes the think block (`<think>\n\n</think>\n\n`) |
| `low` | `reasoning_effort=low` | "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration." |
| `medium` | `reasoning_effort=medium` | no instructions; thinking is on |
| `high` or `xhigh` | `reasoning_effort=xhigh` | "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer." |

The instructions open the system message, before the tool definitions and the client's own system prompt, exactly as
the template renders them. `minimal` has no counterpart in the template and is refused, as is any other value.

The template's variables are also accepted directly, as other servers for this model family accept them:
`chat_template_kwargs: {"enable_thinking": false}`, `{"reasoning_effort": "low"}` (the template's spellings: `low`,
`medium`, `xhigh`), and `{"preserve_thinking": false}`, which drops the think blocks of assistant turns before the last
user message. Settings that contradict each other (`reasoning_effort: "low"` with `enable_thinking: false`) are refused.
No other template variable is accepted.

Because the controls only change the rendered prompt, they change the prompt's tokens and therefore its prefix-cache
identity: requests that differ in effort share cached state only up to the first token where their prompts differ (the
system message, for any effort that has instructions).

## What the levels do

The levels are instructions, not budgets: the template has no token limit for reasoning, and the model decides how long
to think. Measured on `q3`, greedy (MTP speculative decoding), one run each, every answer correct:

| Prompt | `low` | `medium` | `high` |
|---|---|---|---|
| bat and ball | 104 | 199 | 111 |
| integers below 1000 divisible by 7 but not 11 | 286 | 332 | 340 |
| weekday 100 days from Wednesday | 125 | 129 | 128 |
| longest palindromic substring in Python | 125 | 573 | 501 |
| 3^2024 mod 1000 | 3618 | 5656 | 1762 |
| ordered pairs with lcm 2^4·3^3·5^2 | 1065 | 852 | 483 |

(reasoning tokens). The instructions change how the model reasons, but they do not order its length: on the two hardest
prompts `high` reasoned least. The reliable control is on or off. Reasoning length is bounded only by `max_tokens`, which
counts reasoning and answer together; a generation that spends it during reasoning ends with `finish_reason: "length"`
and no answer.

## Output

Chat Completions returns the reasoning as `message.reasoning_content` and, in a stream, as `delta.reasoning_content`
chunks, which all come before the first `content` chunk. The reasoning is the text between `<think>` and `</think>`
with surrounding whitespace removed (the inverse of how the template renders it back); the answer is what follows,
without the blank line that separates it. `usage.completion_tokens_details.reasoning_tokens` counts the reasoning
tokens, which are included in `completion_tokens`. With thinking off neither field appears.

A stream holds back only text that could still be the start of `</think>` or whitespace before it, so reasoning streams
token by token.

- **Stop sequences** apply to the answer only: a stop string inside the reasoning does not end the generation.
- **Tools**: the reasoning is free text. After `</think>` the answer is constrained to the JSON tool-call envelope
  ([TOOL_CALLING.md](TOOL_CALLING.md)); a few whitespace bytes may separate the two, and a generation terminator cannot
  end the reasoning before an answer exists.
- **Speculative decoding** is unaffected: free reasoning needs no sampling constraint, so a greedy request still drafts
  with MTP.

## Replaying a conversation

An assistant message may carry the `reasoning_content` it was returned with. The template keeps the think block of every
earlier assistant turn (`preserve_thinking` defaults to true), so replayed reasoning is rendered there, and a message
without it renders an empty block, as the template does. `reasoning_content` is accepted only on assistant messages.

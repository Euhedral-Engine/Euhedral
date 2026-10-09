#!/usr/bin/env python3
"""Builds the prompt set for MTP drafting measurements (MtpDraftConfidenceCudaIntegrationTest) as JSON lines.

Usage: mtp_draft_prompts.py CHAT_TEMPLATE.jinja OUTPUT.jsonl [--revision REV]   (requires jinja2 >= 3.1)

Three categories of eight prompts, rendered with the checkpoint's chat template (add_generation_prompt=True):

  chat     the benchmark's chat corpus tasks: four over a ~4K-token prefix of its frozen document, four standalone;
  code     coding requests, half over real repository files (tests, review, edits that reproduce a whole file);
  agentic  coding-agent transcripts with tool definitions, tool calls and real tool results (file reads, greps,
           test output), most with thinking enabled, ending where the agent's next turn starts.

With --context TOKENS the set is long-context instead: the four chat document tasks over a document prefix of about
TOKENS, and four agentic transcripts whose earlier turns read repository files until the prompt is about that long.

Repository files are read at REV (default main), so the prompts do not depend on the working tree. Each line is
{"name", "category", "thinking", "text"}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_qwen_chat_template_golden import compile_template  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = "benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/chat-corpus-v1-document.md"
SPECULATIVE = "core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/speculative/"
HOST_LOGITS = "core/src/main/java/io/euhedral_execution/inference/core/generation/HostLogits.java"

# The chat corpus's tasks (ChatPromptCorpus).
DOCUMENT_TASKS = [
    "Summarize the document above in detail, section by section.",
    "Write a Java implementation of the main mechanism the document above describes, with comments explaining each part.",
    "List the design decisions in the document above and explain the trade-off behind each one.",
    "Write a tutorial for a new engineer that explains the ideas in the document above, with examples.",
]
STANDALONE_TASKS = [
    "Write a Java class that implements an LRU cache, with comments, and explain how it works.",
    "Explain in detail how a CPU executes an instruction, from fetch to write-back.",
    "Write a Python script that reads a CSV file of sales and prints totals per month, then explain it line by line.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]
# About 4K tokens of the document (English prose and code run 3.5-4 characters per token).
DOCUMENT_CHARACTERS = 15000

AGENT_SYSTEM = (
    "You are a coding agent working in the user's repository, a Java 25 and CUDA LLM inference engine built with "
    "Gradle. Use the tools to inspect files, search, edit and run commands. Read a file before editing it. Make the "
    "smallest change that does what the user asked, match the surrounding code's style, and run the relevant tests "
    "when you are done. Report what you changed briefly."
)


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    tool("read_file", "Read a file from the repository.", {"path": {"type": "string", "description": "Repository-relative path."}}, ["path"]),
    tool(
        "edit_file",
        "Replace one exact occurrence of old_string with new_string in a file.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    tool("write_file", "Create or overwrite a file.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    tool(
        "grep",
        "Search files with a regular expression; prints path:line:text.",
        {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Directory to search."}},
        ["pattern"],
    ),
    tool(
        "run_command",
        "Run a shell command in the repository root and return its output.",
        {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
        ["command"],
    ),
]


def show(revision: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{path}"], check=True, capture_output=True, text=True
    ).stdout


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def turn(calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": calls}


def result(identifier, content):
    return {"role": "tool", "tool_call_id": identifier, "content": content}


def user(content):
    return {"role": "user", "content": content}


def prompts(revision: str):
    document = show(revision, CORPUS).replace("\r\n", "\n")
    paragraphs, quoted = document.split("\n\n"), []
    for paragraph in paragraphs:
        if sum(len(p) + 2 for p in quoted) + len(paragraph) > DOCUMENT_CHARACTERS:
            break
        quoted.append(paragraph)
    prefix = "\n\n".join(quoted)
    acceptance = show(revision, SPECULATIVE + "SpeculativeAcceptance.java")
    decoder = show(revision, SPECULATIVE + "MtpDecoder.java")
    host_logits = show(revision, HOST_LOGITS)
    quality = show(revision, "tools/dflash2_quality.py")
    checkpoint = show(revision, SPECULATIVE + "SpeculativeCheckpoint.java")

    for index, task in enumerate(DOCUMENT_TASKS):
        yield f"chat-document-{index}", "chat", False, None, [user("Here is a document:\n\n" + prefix + "\n\n" + task)]
    for index, task in enumerate(STANDALONE_TASKS):
        yield f"chat-standalone-{index}", "chat", False, None, [user(task)]

    def with_file(path, text, request):
        return f"{request}\n\n`{path}`:\n```java\n{text}```" if path.endswith(".java") else f"{request}\n\n`{path}`:\n```python\n{text}```"

    yield "code-tests", "code", False, None, [user(with_file(SPECULATIVE + "SpeculativeAcceptance.java", acceptance,
        "Write JUnit 5 tests for this class covering partial, full and zero acceptance, end-of-generation cuts and the output budget."))]
    yield "code-review", "code", False, None, [user(with_file(HOST_LOGITS, host_logits,
        "Review this class. Explain what it does, then list any bugs or risky spots with the line they are on."))]
    yield "code-edit-whole-file", "code", False, None, [user(with_file("tools/dflash2_quality.py", quality,
        "Add a --csv PATH option that also writes the per-position agreement table as CSV. Show the complete updated script."))]
    yield "code-refactor-long", "code", False, None, [user(with_file(SPECULATIVE + "MtpDecoder.java", decoder,
        "Move the catch-up state of Run (the catchUp* fields and the catchUpPiece port) into a nested CatchUp class. "
        "Show the new class and every method that changes."))]
    yield "code-java-queue", "code", False, None, [user(
        "Implement a thread-safe bounded blocking queue in Java with put, take and offer with a timeout, using one "
        "ReentrantLock and two Conditions. Include JUnit 5 tests.")]
    yield "code-rust-duration", "code", False, None, [user(
        "Write a Rust function that parses ISO-8601 durations like P3DT4H12M30.5S into a std::time::Duration, "
        "with an error enum and unit tests.")]
    yield "code-typescript-hook", "code", False, None, [user(
        "Write a React hook in TypeScript, useDebouncedSearch(query, delayMs), that debounces the query, fetches "
        "/api/search?q=..., cancels stale requests with AbortController and returns {results, loading, error}.")]
    yield "code-thinking-queens", "code", True, None, [user(
        "Write a Python function that counts the solutions of the N-Queens problem with bitmasks, and explain its complexity.")]

    read_decoder = call("call_1", "read_file", {"path": SPECULATIVE + "MtpDecoder.java"})
    read_acceptance = call("call_2", "read_file", {"path": SPECULATIVE + "SpeculativeAcceptance.java"})
    yield "agent-edit-statistics", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("MtpDecoder.Statistics should also count verifications whose drafts were all accepted, and print it in toString. Please add that."),
        turn([read_decoder]),
        result("call_1", decoder),
    ]
    yield "agent-edit-statistics-no-thinking", "agentic", False, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("MtpDecoder.Statistics should also count verifications whose drafts were all accepted, and print it in toString. Please add that."),
        turn([read_decoder]),
        result("call_1", decoder),
    ]
    yield "agent-two-reads-then-answer", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("How does the MTP verifier decide how many drafts to accept, and what happens to the state of rejected rows?"),
        turn([read_acceptance], "I'll start with the acceptance rule."),
        result("call_2", acceptance),
        turn([read_decoder]),
        result("call_1", decoder),
    ]
    grep_output = "\n".join([
        SPECULATIVE + "MtpDecoder.java:157:    /// The draft head's token for each of its rows (`text/draft_head_token_ids`).",
        SPECULATIVE + "MtpDecoder.java:159:        TensorHandle ids = plan.weights().runtimeObjects().get(\"text/draft_head_token_ids\");",
        "tools/convert_checkpoint.py:1412:    writer.add(\"text/draft_head_token_ids\", shortlist_ids.astype(np.int32))",
        "docs/MTP_CONTRACT.md:97:  - **Choice: draft with the shortlist** (argmax over 131,072 rows, then map through",
    ])
    yield "agent-grep-then-explain", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("Where does the draft head's shortlist come from, and how does the decoder map a draft row back to a token?"),
        turn([call("call_1", "grep", {"pattern": "draft_head_token_ids", "path": "."})]),
        result("call_1", grep_output),
    ]
    failure = (
        "> Task :core:test FAILED\n\nSpeculativeAcceptanceTest > budgetCutsOutputs() FAILED\n"
        "    org.opentest4j.AssertionFailedError: array lengths differ, expected: <2> but was: <3>\n"
        "        at org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)\n"
        "        at org.junit.jupiter.api.AssertArrayEquals.failArraysNotEqual(AssertArrayEquals.java:440)\n"
        "        at io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeAcceptanceTest.budgetCutsOutputs(SpeculativeAcceptanceTest.java:41)\n\n"
        "12 tests completed, 1 failed\n\nFAILURE: Build failed with an exception.\n"
    )
    yield "agent-test-failure", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("Run the speculative acceptance tests and fix whatever fails."),
        turn([call("call_1", "run_command", {"command": "./gradlew :core:test --tests '*SpeculativeAcceptanceTest'", "timeout_seconds": 600})]),
        result("call_1", failure),
        turn([read_acceptance]),
        result("call_2", acceptance),
    ]
    yield "agent-edit-python-tool", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("tools/dflash2_quality.py should accept any number of candidate reports and compare each with the reference. Update it."),
        turn([call("call_1", "read_file", {"path": "tools/dflash2_quality.py"})]),
        result("call_1", quality),
    ]
    yield "agent-new-file", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("Add a small Java record next to SpeculativeCheckpoint that bundles a checkpoint with its position and token count, with validation in the compact constructor and a test."),
        turn([call("call_1", "read_file", {"path": SPECULATIVE + "SpeculativeCheckpoint.java"})]),
        result("call_1", checkpoint),
    ]
    yield "agent-after-edit-summary", "agentic", True, TOOLS, [
        {"role": "system", "content": AGENT_SYSTEM},
        user("Make HostLogits.close() idempotent and add a test for it."),
        turn([call("call_1", "read_file", {"path": HOST_LOGITS})]),
        result("call_1", host_logits),
        turn([call("call_2", "edit_file", {
            "path": HOST_LOGITS,
            "old_string": "    public void close() {\n        if (this.closed) return;",
            "new_string": "    public void close() {\n        // Idempotent: a second close finds nothing to release.\n        if (this.closed) return;",
        })]),
        result("call_2", "Edited " + HOST_LOGITS + " (1 replacement)."),
        turn([call("call_3", "run_command", {"command": "./gradlew :core:test --tests '*HostLogits*'"})]),
        result("call_3", "> Task :core:test\n\nHostLogitsTest > closeTwice() PASSED\nHostLogitsTest > selectsOnDevice() PASSED\n\nBUILD SUCCESSFUL in 41s\n"),
    ]


# Files an agent reads before the task, in order, to fill a long context.
CONTEXT_FILES = [
    "docs/MTP_CONTRACT.md",
    "docs/MTP_VERIFIER.md",
    SPECULATIVE + "SpeculativeDecoding.java",
    SPECULATIVE + "MtpCheckpoint.java",
    "docs/CUDA_GRAPHS.md",
    "docs/DFLASH2.md",
    SPECULATIVE + "DFlash2Decoder.java",
    "docs/ATTENTION_DECODE.md",
    "docs/NVFP4_NATIVE.md",
    "docs/FRAME_MODEL.md",
    "docs/PREFIX_CACHE.md",
    "docs/NVFP4_RESIDENCY.md",
]
# English prose and code in this repository run about 3.6 characters per token.
CHARACTERS_PER_TOKEN = 3.6


def long_prompts(revision: str, tokens: int):
    budget = int(tokens * CHARACTERS_PER_TOKEN)
    document = show(revision, CORPUS).replace("\r\n", "\n")
    quoted = []
    for paragraph in document.split("\n\n"):
        if sum(len(p) + 2 for p in quoted) + len(paragraph) > budget - 600:
            break
        quoted.append(paragraph)
    prefix = "\n\n".join(quoted)
    for index, task in enumerate(DOCUMENT_TASKS):
        yield f"chat-document-{index}-{tokens}", "chat", False, None, [user("Here is a document:\n\n" + prefix + "\n\n" + task)]

    decoder = show(revision, SPECULATIVE + "MtpDecoder.java")
    acceptance = show(revision, SPECULATIVE + "SpeculativeAcceptance.java")
    host_logits = show(revision, HOST_LOGITS)
    tasks = [
        ("agent-edit-statistics", "MtpDecoder.Statistics should also count verifications whose drafts were all accepted, "
         "and print it in toString. Please add that.", SPECULATIVE + "MtpDecoder.java", decoder),
        ("agent-explain-acceptance", "How does the MTP verifier decide how many drafts to accept, and what happens to the "
         "state of rejected rows? Answer from the code.", SPECULATIVE + "SpeculativeAcceptance.java", acceptance),
        ("agent-close-idempotent", "Make HostLogits.close() idempotent and add a test for it.", HOST_LOGITS, host_logits),
        ("agent-tests-acceptance", "Write JUnit 5 tests for SpeculativeAcceptance covering partial, full and zero "
         "acceptance and the output budget. Create the test file.", SPECULATIVE + "SpeculativeAcceptance.java", acceptance),
    ]
    for name, request, path, text in tasks:
        messages = [{"role": "system", "content": AGENT_SYSTEM}, user(request)]
        used, call_id = len(AGENT_SYSTEM) + len(request) + len(text) + 3000, 0
        for context_path in CONTEXT_FILES:
            content = show(revision, context_path)
            if used + len(content) > budget:
                continue
            call_id += 1
            messages += [turn([call(f"call_{call_id}", "read_file", {"path": context_path})]), result(f"call_{call_id}", content)]
            used += len(content) + 200
        call_id += 1
        messages += [turn([call(f"call_{call_id}", "read_file", {"path": path})]), result(f"call_{call_id}", text)]
        yield f"{name}-{tokens}", "agentic", True, TOOLS, messages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("template")
    parser.add_argument("output")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--context", type=int, help="build the long-context set at about this many tokens")
    args = parser.parse_args(argv)
    template = compile_template(Path(args.template).read_text(encoding="utf-8"))
    with open(args.output, "w", encoding="utf-8") as out:
        entries = long_prompts(args.revision, args.context) if args.context else prompts(args.revision)
        for name, category, thinking, tools, messages in entries:
            text = template.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=thinking)
            out.write(json.dumps({"name": name, "category": category, "thinking": thinking, "text": text}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

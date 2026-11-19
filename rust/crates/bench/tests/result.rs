//! The result row: the fields the comparison tool reads, the output digest, and a decode rate measured from the first
//! token to the last.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test reads a fixture and writes a result file"
)]

use std::collections::BTreeMap;
use std::fs;
use std::time::{Duration, Instant, UNIX_EPOCH};

use euhedral_inference_bench::config::{Scenario, ScenarioKind};
use euhedral_inference_bench::metrics::{Ending, Phase, QuantumRecord, Recorder, Step, evaluate};
use euhedral_inference_bench::result::{
    DraftPhase, Fork, GpuMemory, Provenance, ResultWriter, Row, SUCCESS, ScenarioRecord, Speculative, Throughput,
    Timings, Work, civil, iso_instant, output_sha256, summary,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn sample_row(speculative: Option<Speculative>) -> Row {
    Row {
        schema: "euhedral-inference.benchmark-result".to_string(),
        schema_version: 1,
        implementation: "rust".to_string(),
        provenance: Provenance {
            kind: "measured".to_string(),
            tool: "euhedral-inference-benchmark".to_string(),
            command_line: "run c.json".to_string(),
            source: None,
            source_sha256: None,
            note: None,
        },
        run_id: "run-1".to_string(),
        fork: Fork {
            id: None,
            source: "unspecified".to_string(),
            pid: Some(42),
            jvm_started_at: Some("2026-09-25T00:00:00Z".to_string()),
        },
        recorded_at: "2026-09-25T12:00:00Z".to_string(),
        scenario: ScenarioRecord {
            name: "decode-2-3".to_string(),
            kind: "decode".to_string(),
            target_prompt_tokens: 2,
            requested_new_tokens: 3,
            prompt_generator: "euhedral-words-v1".to_string(),
            prompt_seed: 20_260_925,
            prompt_sha256: "00".to_string(),
        },
        warmup: false,
        iteration: 1,
        status: SUCCESS.to_string(),
        status_reason: None,
        work: Work {
            prompt_tokens: Some(2),
            prefill_quanta: Some(1),
            prefill_tokens: Some(2),
            generated_tokens: Some(3),
            decode_sampled_tokens: Some(2),
            final_commit_quanta: Some(0),
            eos_observed: Some(false),
            speculative,
            output_sha256: Some("ab".to_string()),
            prefix_restored_tokens: None,
        },
        timings: Some(Timings {
            tokenization: Some(10),
            prefill: Some(10),
            first_token_sample: Some(10),
            time_to_first_token: Some(40),
            decode: Some(50),
            decode_quanta_sum: Some(40),
            final_commit: None,
            time_to_last_token: Some(100),
            end_to_end: Some(130),
            prefix_restore: None,
        }),
        throughput: Some(Throughput {
            prefill_tokens_per_second: Some(2.0e8),
            decode_tokens_per_second: Some(4.0e7),
            end_to_end_output_tokens_per_second: Some(2.3e7),
        }),
        engine: json!({"configuration": {"artifact": "a.edrl"}}),
        gpu_memory: Some(GpuMemory {
            before_free_bytes: Some(95),
            after_free_bytes: Some(94),
            total_bytes: Some(1000),
            allocated_before_bytes: None,
            peak_allocated_bytes: None,
            allocated_after_bytes: None,
            retained_workspace_bytes: None,
        }),
    }
}

/// The fields `tools/rust_gate.py` reads from a row, each with the kind of value it takes there. A row written without
/// one of them, or with it of another kind, breaks the comparison.
#[test]
fn a_row_carries_every_field_the_comparison_tool_reads() {
    let speculative = Speculative {
        verifications: 3,
        accepted_drafts: vec![1, 0, 2, 0],
        committed_tokens: 7,
        verify_nanos: 300,
        draft_nanos: 120,
        rejections_in_shortlist: 1,
        rejections_outside_shortlist: 0,
        draft_phases: BTreeMap::from([("catchup".to_string(), DraftPhase { quanta: 3, nanos: 90 })]),
    };
    for row in [sample_row(None), sample_row(Some(speculative))] {
        let speculated = row.work.speculative.is_some();
        let value: Value = serde_json::from_str(&row.to_json()).unwrap();
        assert!(value["warmup"].is_boolean());
        assert!(value["status"].is_string());
        assert!(value["iteration"].is_u64());
        assert!(value["scenario"]["name"].is_string());
        assert!(value["scenario"]["promptSha256"].is_string());
        let work = &value["work"];
        for key in ["promptTokens", "generatedTokens", "decodeSampledTokens"] {
            assert!(work[key].is_u64(), "work.{key}");
        }
        assert!(work["outputSha256"].is_string());
        // The histogram is a list of counts when the row speculated, and absent or null when it did not.
        if speculated {
            let drafts = work["speculative"]["acceptedDrafts"].as_array().unwrap();
            assert!(!drafts.is_empty() && drafts.iter().all(Value::is_u64));
        } else {
            assert!(work["speculative"].is_null());
        }
        assert!(value["timings"]["timeToFirstToken"].is_u64());
        assert!(value["timings"]["timeToLastToken"].is_u64());
        assert!(value["throughput"]["prefillTokensPerSecond"].is_number());
    }
}

#[test]
fn output_sha_is_comma_joined_decimal() {
    // SHA-256 of "1,2,3", of "151645" and of nothing.
    assert_eq!(
        output_sha256(&[1, 2, 3]),
        "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d"
    );
    assert_eq!(
        output_sha256(&[151_645]),
        "b4778fde720738498ace5dd4aa258dcfc1f4cdafbbb666da52e5ae2f5c95d7f8"
    );
    assert_eq!(
        output_sha256(&[]),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn instants_print_as_java_prints_them() {
    assert_eq!(civil(1_790_424_000), (2026, 9, 26, 12, 0, 0));
    assert_eq!(civil(0), (1970, 1, 1, 0, 0, 0));
    assert_eq!(civil(951_782_400), (2000, 2, 29, 0, 0, 0));
    assert_eq!(
        iso_instant(UNIX_EPOCH + Duration::from_secs(1_790_424_000)),
        "2026-09-26T12:00:00Z"
    );
    assert_eq!(
        iso_instant(UNIX_EPOCH + Duration::new(1_790_424_000, 120_000_000)),
        "2026-09-26T12:00:00.120Z"
    );
    assert_eq!(
        iso_instant(UNIX_EPOCH + Duration::new(1_790_424_000, 120_500_000)),
        "2026-09-26T12:00:00.120500Z"
    );
    // 700 ns: a system clock with 100 ns ticks (Windows) holds it exactly.
    assert_eq!(
        iso_instant(UNIX_EPOCH + Duration::new(1_790_424_000, 700)),
        "2026-09-26T12:00:00.000000700Z"
    );
}

/// A generation of `tokens` tokens: a prompt of 512 tokens prefilled in one quantum, the first token 20 ms after the
/// entry, then one token every `interval`, ending `tail` after the last token.
fn generation(tokens: usize, interval: Duration, tail: Duration) -> (Recorder, Instant) {
    let entry = Instant::now();
    let mut recorder = Recorder::new(entry);
    recorder.encoded = Some((entry + Duration::from_millis(2), 512));
    recorder.quanta.push(QuantumRecord {
        phase: Phase::Prefill,
        rows: 512,
        admitted: entry + Duration::from_millis(3),
        retired: entry + Duration::from_millis(19),
    });
    let first = entry + Duration::from_millis(20);
    for index in 0..tokens {
        let at = first + interval * index as u32;
        recorder.tokens.push((100 + index as u32, at));
        if index > 0 {
            recorder.quanta.push(QuantumRecord {
                phase: Phase::Decode,
                rows: 1,
                admitted: at - interval,
                retired: at,
            });
        }
    }
    let last = first + interval * (tokens as u32 - 1);
    recorder.finished = Some((last + tail, Ending::Completed));
    (recorder, entry)
}

#[test]
fn decode_tps_excludes_first_token_and_final_commit() {
    let scenario = Scenario::new(ScenarioKind::SustainedDecode, 512, 5).unwrap();
    // Five tokens, 10 ms apart: four decode tokens over 40 ms, whatever follows the last token. The first token (found
    // at the end of prefill) is outside the window, and so is the 25 ms the request runs on after the last token.
    for tail in [Duration::ZERO, Duration::from_millis(25)] {
        let (recorder, _) = generation(5, Duration::from_millis(10), tail);
        let outcome = evaluate(&scenario, &recorder, |_| false);
        assert_eq!(outcome.status, SUCCESS, "{:?}", outcome.reason);
        let timings = outcome.timings.unwrap();
        let throughput = outcome.throughput.unwrap();
        assert_eq!(outcome.work.generated_tokens, Some(5));
        assert_eq!(outcome.work.decode_sampled_tokens, Some(4));
        assert_eq!(outcome.work.final_commit_quanta, Some(0));
        assert_eq!(timings.decode, Some(40_000_000));
        assert_eq!(timings.final_commit, None);
        let tps = throughput.decode_tokens_per_second.unwrap();
        assert!(
            (tps - 100.0).abs() < 1e-6,
            "four tokens in 40 ms is 100 tok/s, got {tps}"
        );
        // The other windows keep their own boundaries.
        assert_eq!(timings.tokenization, Some(2_000_000));
        assert_eq!(timings.prefill, Some(16_000_000));
        assert_eq!(timings.first_token_sample, Some(1_000_000));
        assert_eq!(timings.time_to_first_token, Some(20_000_000));
        assert_eq!(timings.time_to_last_token, Some(60_000_000));
        assert_eq!(timings.end_to_end, Some(60_000_000 + tail.as_nanos() as i64));
        let prefill_tps = throughput.prefill_tokens_per_second.unwrap();
        assert!((prefill_tps - 512.0 / 0.016).abs() < 1e-3);
    }
}

#[test]
fn a_decode_quantum_after_the_last_token_is_not_in_the_window() {
    let scenario = Scenario::new(ScenarioKind::SustainedDecode, 512, 5).unwrap();
    let (mut recorder, _) = generation(5, Duration::from_millis(10), Duration::from_millis(30));
    // A decode quantum that retires after the last token was found (the engine does not run one, but a listener that
    // saw one must not stretch the window or the count).
    let last = recorder.tokens.last().unwrap().1;
    recorder.quanta.push(QuantumRecord {
        phase: Phase::Decode,
        rows: 1,
        admitted: last,
        retired: last + Duration::from_millis(25),
    });
    let outcome = evaluate(&scenario, &recorder, |_| false);
    assert_eq!(outcome.timings.unwrap().decode, Some(40_000_000));
    assert_eq!(outcome.work.decode_sampled_tokens, Some(4));
    assert!((outcome.throughput.unwrap().decode_tokens_per_second.unwrap() - 100.0).abs() < 1e-6);
}

#[test]
fn statuses_follow_the_javas_rules() {
    let decode = Scenario::new(ScenarioKind::SustainedDecode, 512, 5).unwrap();
    // A terminator makes the row ineligible; fewer tokens than asked as well.
    let (recorder, _) = generation(3, Duration::from_millis(10), Duration::ZERO);
    let outcome = evaluate(&decode, &recorder, |id| id == 102);
    assert_eq!(
        (outcome.status, outcome.reason.as_deref()),
        ("ineligible", Some("eos_after_3_of_5_tokens"))
    );
    assert_eq!(outcome.work.eos_observed, Some(true));
    assert!(outcome.timings.is_some(), "an ineligible row keeps its timings");
    let outcome = evaluate(&decode, &recorder, |_| false);
    assert_eq!(
        (outcome.status, outcome.reason.as_deref()),
        ("ineligible", Some("stopped_after_3_of_5_tokens"))
    );

    // A failure keeps the work done and reports no timings or rates.
    let (mut failed, _) = generation(2, Duration::from_millis(10), Duration::ZERO);
    failed.finished = Some((Instant::now(), Ending::Failed("a quantum failed".to_string())));
    let outcome = evaluate(&decode, &failed, |_| false);
    assert_eq!(
        (outcome.status, outcome.reason.as_deref()),
        ("failed", Some("a quantum failed"))
    );
    assert!(outcome.timings.is_none() && outcome.throughput.is_none());
    assert_eq!(outcome.work.prefill_tokens, Some(512));
    assert_eq!(outcome.work.generated_tokens, None);

    // A prompt that was not prefilled whole is incomplete.
    let (mut short, _) = generation(5, Duration::from_millis(10), Duration::ZERO);
    short.quanta[0].rows = 500;
    assert_eq!(
        evaluate(&decode, &short, |_| false).reason.as_deref(),
        Some("incomplete_prefill")
    );

    // Prefill alone: no token, no decode window, no decode rate.
    let prefill = Scenario::new(ScenarioKind::Prefill, 512, 0).unwrap();
    let mut only = Recorder::new(Instant::now());
    only.encoded = Some((only.entry + Duration::from_millis(1), 512));
    only.quanta.push(QuantumRecord {
        phase: Phase::Prefill,
        rows: 512,
        admitted: only.entry + Duration::from_millis(2),
        retired: only.entry + Duration::from_millis(12),
    });
    only.finished = Some((only.entry + Duration::from_millis(12), Ending::Completed));
    let outcome = evaluate(&prefill, &only, |_| false);
    assert_eq!(outcome.status, SUCCESS);
    assert_eq!(outcome.timings.unwrap().prefill, Some(10_000_000));
    assert_eq!(outcome.throughput.unwrap().decode_tokens_per_second, None);
    assert_eq!(outcome.work.output_sha256.as_deref(), Some(output_sha256(&[]).as_str()));

    // First token: exactly one token and no decode.
    let first = Scenario::new(ScenarioKind::FirstToken, 512, 1).unwrap();
    let (one, _) = generation(1, Duration::from_millis(10), Duration::ZERO);
    let outcome = evaluate(&first, &one, |_| false);
    assert_eq!(outcome.status, SUCCESS);
    assert_eq!(outcome.work.decode_sampled_tokens, Some(0));
    assert_eq!(outcome.timings.unwrap().decode, None);
}

#[test]
fn a_restored_prefix_is_not_prefilled_and_is_reported() {
    let prefix = Scenario::new(ScenarioKind::PrefixRepeat, 512, 1).unwrap();
    let (mut recorder, _) = generation(1, Duration::from_millis(10), Duration::ZERO);
    // The engine restored 256 of the 512 prompt tokens and prefilled the other 256.
    recorder.quanta[0].rows = 256;
    assert_eq!(
        evaluate(&prefix, &recorder, |_| false).reason.as_deref(),
        Some("incomplete_prefill"),
        "without a restore, 256 prefilled tokens are not the prompt"
    );
    recorder.restored = 256;
    let outcome = evaluate(&prefix, &recorder, |_| false);
    assert_eq!(outcome.status, SUCCESS);
    assert_eq!(outcome.work.prefix_restored_tokens, Some(256));
    assert_eq!(outcome.work.prefill_tokens, Some(256));
    assert!(outcome.timings.unwrap().time_to_first_token.is_some());

    // A miss reports no restore.
    let (miss, _) = generation(1, Duration::from_millis(10), Duration::ZERO);
    assert_eq!(evaluate(&prefix, &miss, |_| false).work.prefix_restored_tokens, None);
}

#[test]
fn speculative_steps_fill_the_block_and_set_the_window() {
    let scenario = Scenario::new(ScenarioKind::SustainedDecode, 512, 9).unwrap();
    let (mut recorder, _) = generation(9, Duration::from_millis(10), Duration::ZERO);
    let first = recorder.tokens[0].1;
    let ms = Duration::from_millis;
    // Step 1 starts 4 ms after the first token and takes 6 ms; it accepts 2 drafts and commits 3 tokens. Step 2 starts
    // 2 ms later, takes 8 ms, accepts 0 (the base token was in the shortlist) and commits 1.
    recorder.steps = vec![
        Step {
            start: first + ms(4),
            executed: first + ms(10),
            outputs: 3,
            accepted: 2,
            rejection: -1,
        },
        Step {
            start: first + ms(12),
            executed: first + ms(20),
            outputs: 1,
            accepted: 0,
            rejection: 0,
        },
    ];
    let outcome = evaluate(&scenario, &recorder, |_| false);
    let speculative = outcome.work.speculative.unwrap();
    assert_eq!(speculative.verifications, 2);
    assert_eq!(speculative.accepted_drafts, vec![1, 0, 1, 0]);
    assert_eq!(speculative.committed_tokens, 4);
    assert_eq!(speculative.verify_nanos, 14_000_000);
    assert_eq!(speculative.draft_nanos, 6_000_000);
    assert_eq!(
        (
            speculative.rejections_in_shortlist,
            speculative.rejections_outside_shortlist
        ),
        (1, 0)
    );
    assert_eq!(outcome.work.decode_sampled_tokens, Some(4));
    assert_eq!(outcome.timings.unwrap().decode, Some(20_000_000));
    assert_eq!(outcome.timings.unwrap().decode_quanta_sum, Some(14_000_000));

    // Without steps the block is null.
    let (plain, _) = generation(9, Duration::from_millis(10), Duration::ZERO);
    assert!(evaluate(&scenario, &plain, |_| false).work.speculative.is_none());
}

#[test]
fn writer_writes_jsonl_per_row_and_a_document_on_close() {
    let directory = tempdir().unwrap();
    let lines = directory.path().join("nested/rows.jsonl");
    let mut writer = ResultWriter::create(&lines, false, false, false).unwrap();
    writer.write(sample_row(None)).unwrap();
    // The row is on disk before the writer closes.
    assert_eq!(fs::read_to_string(&lines).unwrap().lines().count(), 1);
    writer.write(sample_row(None)).unwrap();
    writer.close().unwrap();
    for line in fs::read_to_string(&lines).unwrap().lines() {
        assert_eq!(serde_json::from_str::<Value>(line).unwrap()["implementation"], "rust");
    }
    // An existing output is refused unless overwrite or append is asked for; append adds to it.
    assert!(ResultWriter::create(&lines, false, false, false).is_err());
    let mut again = ResultWriter::create(&lines, false, false, true).unwrap();
    again.write(sample_row(None)).unwrap();
    again.close().unwrap();
    assert_eq!(fs::read_to_string(&lines).unwrap().lines().count(), 3);
    drop(ResultWriter::create(&lines, false, true, false).unwrap());
    assert_eq!(
        fs::read_to_string(&lines).unwrap().lines().count(),
        0,
        "overwrite truncates"
    );

    let document = directory.path().join("rows.json");
    let mut writer = ResultWriter::create(&document, true, false, false).unwrap();
    writer.write(sample_row(None)).unwrap();
    writer.close().unwrap();
    let value: Value = serde_json::from_str(&fs::read_to_string(&document).unwrap()).unwrap();
    assert_eq!(value["schema"], "euhedral-inference.benchmark-results");
    assert_eq!(value["schemaVersion"], 1);
    assert_eq!(value["results"].as_array().unwrap().len(), 1);
}

#[test]
fn the_summary_reports_medians_of_measured_successes() {
    let mut rows = Vec::new();
    for (index, decode) in [100.0, 300.0, 200.0].into_iter().enumerate() {
        let mut row = sample_row(None);
        row.iteration = index;
        row.throughput = Some(Throughput {
            prefill_tokens_per_second: Some(1000.0),
            decode_tokens_per_second: Some(decode),
            end_to_end_output_tokens_per_second: Some(1.0),
        });
        rows.push(row);
    }
    let mut warmup = sample_row(None);
    warmup.warmup = true;
    warmup.throughput = Some(Throughput {
        prefill_tokens_per_second: Some(1.0),
        decode_tokens_per_second: Some(1.0),
        end_to_end_output_tokens_per_second: Some(1.0),
    });
    rows.push(warmup);
    let text = summary(&rows);
    let line = text
        .lines()
        .find(|line| line.starts_with("rust | a.edrl | decode-2-3"))
        .expect(&text);
    assert!(line.contains("200.0"), "the median decode rate: {line}");
    assert!(line.contains("1000.0"), "the median prefill rate: {line}");
}

#[test]
fn a_prefix_row_is_eligible_only_if_it_measured_what_its_name_says() {
    use euhedral_inference_bench::metrics::{Pass, prefix_ineligibility};
    assert_eq!(
        prefix_ineligibility(Pass::Cold, 512, 2048),
        Some("cold_request_restored_a_prefix")
    );
    assert_eq!(prefix_ineligibility(Pass::Cold, 0, 2048), None);
    assert_eq!(
        prefix_ineligibility(Pass::Repeat, 1024, 2048),
        Some("repeat_did_not_restore_the_shared_prompt")
    );
    assert_eq!(prefix_ineligibility(Pass::Repeat, 2048, 2048), None);
    assert_eq!(prefix_ineligibility(Pass::Only, 0, 2048), None);
    assert_eq!(prefix_ineligibility(Pass::Only, 7, 2048), None);
}

#[test]
fn only_a_prefix_scenario_counts_a_restore_as_part_of_the_prompt() {
    let (mut recorder, _) = generation(1, Duration::from_millis(10), Duration::ZERO);
    recorder.quanta[0].rows = 256;
    recorder.restored = 256;
    let first = Scenario::new(ScenarioKind::FirstToken, 512, 1).unwrap();
    assert_eq!(
        evaluate(&first, &recorder, |_| false).reason.as_deref(),
        Some("incomplete_prefill")
    );
}

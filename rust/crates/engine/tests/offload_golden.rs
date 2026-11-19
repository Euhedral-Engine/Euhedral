//! The plans the reference planner makes when the model does not fit the device, row for row.
//!
//! `fixtures/offload_plans.tsv` holds, for the RCO artifact's tensor table, the reference planner's plan for a grid of
//! device deficits, host budgets, sequence counts and contexts: which objects it holds in host memory and which it
//! leaves in the file (in selection order), whether the plan fits, which memory a refusal names, how many sequences
//! `auto` resolves to and how many cache pages are on the device and on the host. Every one of them is compared. The
//! byte columns are informational: the device need of this engine differs from the reference's by the alignment of
//! the arena and the workspace's records, so each row's free memory is this engine's resident need minus the row's
//! deficit.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::residency::footprint::PAGE_TOKENS;
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_closest, plan_with, sequences_for};

mod common;

use common::{config, rco_table, settings};

const AUTO_CAP: usize = 102;

struct Row {
    line: usize,
    deficit: u64,
    host_budget: u64,
    sequences: Option<usize>,
    context: u32,
    kv_device_min_tokens: u32,
    fits: bool,
    refusal: String,
    host_set: Vec<String>,
    disk_set: Vec<String>,
    resolved: usize,
    device_pages: usize,
    host_pages: usize,
}

fn names(field: &str) -> Vec<String> {
    field
        .split(',')
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

fn rows() -> Vec<Row> {
    let text = include_str!("fixtures/offload_plans.tsv");
    let mut lines = text.lines().enumerate().filter(|(_, line)| !line.starts_with('#'));
    let (_, header) = lines.next().unwrap();
    let columns: Vec<&str> = header.split('\t').collect();
    let at = |name: &str| columns.iter().position(|column| *column == name).unwrap();
    let (deficit, budget, sequences, context, minimum, fits, refusal, host, disk, resolved, device_pages, host_pages) = (
        at("deficit"),
        at("host_budget"),
        at("sequences"),
        at("context"),
        at("kv_device_min_tokens"),
        at("fits"),
        at("refusal"),
        at("host_set"),
        at("disk_set"),
        at("resolved_sequences"),
        at("device_kv_pages"),
        at("host_kv_pages"),
    );
    lines
        .map(|(line, text)| {
            let fields: Vec<&str> = text.split('\t').collect();
            Row {
                line: line + 1,
                deficit: fields[deficit].parse().unwrap(),
                host_budget: match fields[budget] {
                    "unlimited" => u64::MAX,
                    figure => figure.parse().unwrap(),
                },
                sequences: match fields[sequences] {
                    "auto" => None,
                    count => Some(count.parse().unwrap()),
                },
                context: fields[context].parse().unwrap(),
                kv_device_min_tokens: fields[minimum].parse().unwrap(),
                fits: fields[fits] == "true",
                refusal: fields[refusal].to_owned(),
                host_set: names(fields[host]),
                disk_set: names(fields[disk]),
                resolved: fields[resolved].parse().unwrap(),
                device_pages: fields[device_pages].parse().unwrap(),
                host_pages: fields[host_pages].parse().unwrap(),
            }
        })
        .collect()
}

#[test]
fn the_plans_of_the_reference_planner_are_reproduced_row_for_row() {
    let (config, table) = (config(), rco_table());
    let rows = rows();
    assert_eq!(rows.len(), 316);
    let mut mismatches = Vec::new();
    let mut kinds = [0usize; 3];
    for row in &rows {
        let mut settings = settings(row.context);
        settings.context.kv_cache_on_gpu_tokens = row.kv_device_min_tokens;
        let at = |device_free: u64, sequences: usize| PlanInputs {
            device_free,
            multiprocessors: Some(70),
            host_pinned_bytes: Some(row.host_budget),
            sequences,
            module_bytes: 0,
        };
        // This engine's resident need for the row's sequence count (one for `auto`): everything on the device.
        let need_for = row.sequences.unwrap_or(1);
        let need = plan_with(
            &config,
            &table,
            &settings,
            PlanInputs {
                host_pinned_bytes: Some(u64::MAX),
                ..at(u64::MAX, need_for)
            },
        )
        .unwrap();
        assert!(need.offload.is_empty() && need.kv_host_tokens == 0);
        let free = need.device_bytes - row.deficit;
        let resolved = match row.sequences {
            Some(count) => count,
            None => match sequences_for(&config, &table, &settings, at(free, 1), AUTO_CAP) {
                Ok(count) => count,
                Err(PlanError::HostTooSmall { .. } | PlanError::DeviceTooSmallWithOffload { .. }) => 1,
                Err(other) => panic!("line {}: {other}", row.line),
            },
        };
        let (plan, refusal) = plan_closest(&config, &table, &settings, at(free, resolved)).unwrap();
        let kind = match &refusal {
            None => "none",
            Some(PlanError::HostTooSmall { .. }) => "host",
            Some(PlanError::DeviceTooSmallWithOffload { .. }) => "device",
            Some(other) => panic!("line {}: {other}", row.line),
        };
        kinds[["none", "host", "device"]
            .iter()
            .position(|name| *name == kind)
            .unwrap()] += 1;
        let mut wrong = Vec::new();
        if plan.offload.host != row.host_set {
            wrong.push(format!(
                "host set {:?} not {:?}",
                plan.offload.host.len(),
                row.host_set.len()
            ));
        }
        if plan.offload.disk != row.disk_set {
            wrong.push(format!(
                "disk set {:?} not {:?}",
                plan.offload.disk.len(),
                row.disk_set.len()
            ));
        }
        if refusal.is_none() != row.fits {
            wrong.push(format!("fits {} not {}", refusal.is_none(), row.fits));
        }
        if kind != row.refusal {
            wrong.push(format!("refusal {kind} not {}", row.refusal));
        }
        if resolved != row.resolved {
            wrong.push(format!("sequences {resolved} not {}", row.resolved));
        }
        let page = PAGE_TOKENS as usize;
        if plan.kv_device_tokens / page != row.device_pages || plan.kv_host_tokens / page != row.host_pages {
            wrong.push(format!(
                "pages {}+{} not {}+{}",
                plan.kv_device_tokens / page,
                plan.kv_host_tokens / page,
                row.device_pages,
                row.host_pages
            ));
        }
        if !wrong.is_empty() {
            mismatches.push(format!(
                "line {} (deficit {}, host {}, sequences {:?}, context {}, kv {}): {}",
                row.line,
                row.deficit,
                row.host_budget,
                row.sequences,
                row.context,
                row.kv_device_min_tokens,
                wrong.join("; ")
            ));
        }
    }
    assert_eq!(
        kinds,
        [250, 61, 5],
        "the reference's mix of fits, host and device refusals"
    );
    assert!(
        mismatches.is_empty(),
        "{} of {} rows differ:\n{}",
        mismatches.len(),
        rows.len(),
        mismatches.join("\n")
    );
}

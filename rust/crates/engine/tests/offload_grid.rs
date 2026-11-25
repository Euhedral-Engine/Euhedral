//! The plans the planner makes when the model does not fit the device, over a grid of device deficits, host budgets,
//! sequence counts and contexts, held to the rules the plan is made by:
//! - an admitted plan fits both memories, counted again from the placements it reports;
//! - a plan is refused only when no selection fits: the device refusal when even the most the plan may offload leaves
//!   the device short, the host refusal when the device fits and the host is short by its own count;
//! - more host budget never turns an admitted plan into a refusal;
//! - `auto` sequences is the largest count that offloads no more than one sequence does.
//!
//! Each row's free memory is this engine's resident need minus the row's deficit, so the grid reaches the same edges
//! whatever the sizes of the arena and the workspace.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::HashSet;

use euhedral_inference_edrl::{TensorTable, WeightFormat};
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::{ArtifactProfile, ModelConfig};
use euhedral_inference_engine::residency::footprint::PAGE_TOKENS;
use euhedral_inference_engine::residency::offload::{
    EMBEDDING, HOST_ALIGNMENT, MAPPED_HEAD_MAX_ROWS, OUTPUT_HEAD, mapped, weight_objects,
};
use euhedral_inference_engine::residency::planner::{
    ARENA_ALIGNMENT, DEFAULT_STAGING_SLOTS, PlanInputs, ResidencyPlan, Tier, plan_closest, plan_with, sequences_for,
};
use euhedral_inference_settings::Settings;

mod common;

use common::{config, rco_table, settings};

const AUTO_CAP: usize = 102;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Device deficits: the resident need exactly, then short of it by sizes from a few steps of the search to more than
/// anything the plan may offload.
const DEFICITS: [u64; 7] = [
    0,
    261 * MIB,
    1029 * MIB,
    3077 * MIB,
    6149 * MIB,
    10245 * MIB,
    12293 * MIB,
];
/// Host budgets: unlimited, then amounts that hold the offload whole, in part and not even the embedding.
const HOST_BUDGETS: [u64; 4] = [u64::MAX, 4 * GIB, 3 * GIB / 2, 600 * MIB];
/// Sequence counts; `None` is `auto`.
const SEQUENCES: [Option<usize>; 4] = [Some(1), Some(2), Some(4), None];
/// `(context, tokens of cache pages kept on the device)`: the pages all on the device, then the pages past a device
/// share spilled to the host.
const CONTEXTS: [(u32, u32); 3] = [(4096, 65_536), (32_768, 65_536), (32_768, 8192)];

fn align(value: u64, to: u64) -> u64 {
    value.div_ceil(to) * to
}

struct Setup {
    config: ModelConfig,
    table: TensorTable,
    settings: Settings,
}

impl Setup {
    fn new(context: (u32, u32)) -> Setup {
        let mut settings = settings(context.0);
        settings.context.kv_cache_on_gpu_tokens = context.1;
        Setup {
            config: config(),
            table: rco_table(),
            settings,
        }
    }

    fn inputs(&self, device_free: u64, host: u64, sequences: usize) -> PlanInputs {
        PlanInputs {
            device_free,
            multiprocessors: Some(70),
            exact_numerics: false,
            host_pinned_bytes: Some(host),
            sequences,
        }
    }

    /// The device bytes of a plan of `sequences` with everything on the device.
    fn need(&self, sequences: usize) -> u64 {
        let plan = plan_with(
            &self.config,
            &self.table,
            &self.settings,
            self.inputs(u64::MAX, u64::MAX, sequences),
        )
        .unwrap();
        assert!(plan.offload.is_empty() && plan.kv_host_tokens == 0);
        plan.device_bytes
    }

    /// The sequences `sequences` asks for: its count, or what `auto` makes of `free` and `host`.
    fn resolve(&self, sequences: Option<usize>, free: u64, host: u64) -> usize {
        sequences.unwrap_or_else(|| {
            match sequences_for(
                &self.config,
                &self.table,
                &self.settings,
                self.inputs(free, host, 1),
                AUTO_CAP,
            ) {
                Ok(count) => count,
                Err(PlanError::HostTooSmall { .. } | PlanError::DeviceTooSmallWithOffload { .. }) => 1,
                Err(other) => panic!("{other}"),
            }
        })
    }
}

/// What the rules say a plan holds, counted again from the placements it reports.
struct Counts {
    /// The device bytes of the objects that stay, each aligned.
    resident: u64,
    /// The device bytes of every object, each aligned: what a plan with nothing offloaded keeps.
    total: u64,
    /// The pinned host bytes of the objects held on the host, each at a multiple of the host alignment.
    host_arena: u64,
    /// The largest object left in the file.
    largest_disk: u64,
    /// The device bytes of the weights the plan may offload, the staging ring of the largest of them, and whether the
    /// output head is among them.
    offloadable: u64,
    ring_if_all: u64,
}

fn count(plan: &ResidencyPlan, table: &TensorTable, config: &ModelConfig, head_rows: u64) -> Counts {
    let profile = ArtifactProfile::of(table, config).unwrap();
    let objects = weight_objects(table, &profile);
    let stageable: HashSet<&str> = objects
        .iter()
        .filter(|object| object.stageable)
        .map(|object| object.name)
        .collect();
    let mut counts = Counts {
        resident: 0,
        total: 0,
        host_arena: 0,
        largest_disk: 0,
        offloadable: 0,
        ring_if_all: 0,
    };
    let mut largest_staged = 0;
    for placement in &plan.device {
        let aligned = align(placement.bytes, ARENA_ALIGNMENT);
        counts.total += aligned;
        match placement.tier {
            Tier::Device => counts.resident += aligned,
            Tier::Host { .. } => counts.host_arena += align(placement.bytes, HOST_ALIGNMENT),
            Tier::Disk => counts.largest_disk = counts.largest_disk.max(placement.bytes),
        }
        let head_in_reach = placement.name == OUTPUT_HEAD
            && !(placement.format == WeightFormat::Nvfp4 && head_rows > MAPPED_HEAD_MAX_ROWS);
        if placement.name == EMBEDDING || head_in_reach {
            counts.offloadable += aligned;
        } else if stageable.contains(placement.name.as_str()) {
            counts.offloadable += aligned;
            largest_staged = largest_staged.max(placement.bytes);
        }
    }
    counts.ring_if_all = u64::from(DEFAULT_STAGING_SLOTS) * align(largest_staged, ARENA_ALIGNMENT);
    counts
}

/// The pinned host bytes a plan needs by the rules: the arena of the host objects, the window of the file objects
/// (two slots for each staging slot, each holding the blocks around the largest) and the cache pages put on the host.
fn host_needed(plan: &ResidencyPlan, counts: &Counts) -> u64 {
    let window = if counts.largest_disk == 0 {
        0
    } else {
        2 * u64::from(DEFAULT_STAGING_SLOTS) * (align(counts.largest_disk, HOST_ALIGNMENT) + HOST_ALIGNMENT)
    };
    counts
        .host_arena
        .saturating_add(window)
        .saturating_add(plan.kv_host_bytes)
}

#[test]
fn the_plans_over_the_grid_follow_the_rules_of_the_plan() {
    let mut kinds = [0usize; 3];
    for context in CONTEXTS {
        let setup = Setup::new(context);
        for sequences in SEQUENCES {
            let need = setup.need(sequences.unwrap_or(1));
            for deficit in DEFICITS {
                let free = need.saturating_sub(deficit);
                // The count is the one the largest host budget resolves, so the budgets below are compared like for
                // like.
                let resolved = setup.resolve(sequences, free, HOST_BUDGETS[0]);
                let mut refused_at_a_larger_budget = false;
                for host in HOST_BUDGETS {
                    let at = format!("context {context:?}, sequences {sequences:?}, deficit {deficit}, host {host}");
                    let (plan, refusal) = plan_closest(
                        &setup.config,
                        &setup.table,
                        &setup.settings,
                        setup.inputs(free, host, resolved),
                    )
                    .unwrap();
                    let head_rows = resolved as u64 * (1 + plan.draft_depth as u64);
                    let counts = count(&plan, &setup.table, &setup.config, head_rows);
                    let needed = host_needed(&plan, &counts);
                    let fixed = fixed(&plan);

                    // What the plan reports of itself is what its placements and the rules of its parts say.
                    assert_eq!(plan.arena_bytes, counts.resident, "{at}: the arena");
                    assert_eq!(plan.offload.host_arena_bytes, counts.host_arena, "{at}: the host arena");
                    assert_eq!(
                        plan.offload.host_bytes() + plan.kv_host_bytes,
                        needed,
                        "{at}: the host bytes"
                    );
                    assert_eq!(
                        plan.device_bytes,
                        plan.arena_bytes + plan.offload.staging_bytes() + fixed,
                        "{at}: the device bytes"
                    );
                    let page = PAGE_TOKENS as usize;
                    assert_eq!(plan.kv_device_tokens % page, 0, "{at}");
                    assert_eq!(plan.kv_device_tokens + plan.kv_host_tokens, context.0 as usize, "{at}");
                    // The embedding and the head are read in place, so they are never left in the file.
                    assert!(plan.offload.disk.iter().all(|name| !mapped(name)), "{at}");

                    match &refusal {
                        None => {
                            kinds[0] += 1;
                            assert!(plan.device_bytes <= free, "{at}: the plan does not fit the device");
                            assert!(needed <= host, "{at}: the plan does not fit the host");
                            assert!(
                                !refused_at_a_larger_budget,
                                "{at}: a larger budget refused what this one admits"
                            );
                        }
                        Some(PlanError::DeviceTooSmallWithOffload { .. }) => {
                            kinds[2] += 1;
                            // No selection fits: with everything the plan may offload gone, the device is still short.
                            let lightest = counts.total - counts.offloadable + counts.ring_if_all + fixed;
                            assert!(lightest > free, "{at}: {lightest} bytes fit {free} after all");
                            refused_at_a_larger_budget = true;
                        }
                        Some(PlanError::HostTooSmall {
                            needed: stated,
                            cache_pages,
                            budget,
                        }) => {
                            kinds[1] += 1;
                            // The device fits, and the host is short by its own count.
                            assert!(plan.device_bytes <= free, "{at}: the device is short too");
                            assert!(needed > host, "{at}: the host holds {needed}");
                            assert_eq!(
                                (*stated, *cache_pages, *budget),
                                (needed, plan.kv_host_bytes, host),
                                "{at}"
                            );
                            refused_at_a_larger_budget = true;
                        }
                        Some(other) => panic!("{at}: {other}"),
                    }
                }
            }
        }
    }
    assert!(
        kinds.iter().all(|&kind| kind > 20),
        "the grid reaches every outcome: {kinds:?}"
    );
}

#[test]
fn auto_sequences_is_the_most_that_offload_no_more_than_one_does() {
    for context in CONTEXTS {
        let setup = Setup::new(context);
        let need = setup.need(1);
        for deficit in DEFICITS {
            let free = need.saturating_sub(deficit);
            for host in HOST_BUDGETS {
                let at = format!("context {context:?}, deficit {deficit}, host {host}");
                let plan_of = |sequences: usize| {
                    plan_with(
                        &setup.config,
                        &setup.table,
                        &setup.settings,
                        setup.inputs(free, host, sequences),
                    )
                };
                let resolved = setup.resolve(None, free, host);
                assert!((1..=AUTO_CAP).contains(&resolved), "{at}: {resolved}");
                let Ok(one) = plan_of(1) else {
                    assert_eq!(resolved, 1, "{at}: one sequence is refused");
                    continue;
                };
                let kept = plan_of(resolved).unwrap_or_else(|refusal| panic!("{at}: {resolved} sequences: {refusal}"));
                assert!(
                    kept.offloaded_bytes() <= one.offloaded_bytes(),
                    "{at}: {resolved} offload more"
                );
                if resolved < AUTO_CAP
                    && let Ok(more) = plan_of(resolved + 1)
                {
                    assert!(
                        more.offloaded_bytes() > one.offloaded_bytes(),
                        "{at}: {} sequences offload no more than one",
                        resolved + 1
                    );
                }
            }
        }
    }
}

/// The device bytes of a plan besides its weights and its staging ring: the cache, the reserve, the scratch and the
/// state of the sequences beyond the first.
fn fixed(plan: &ResidencyPlan) -> u64 {
    plan.kv_device_bytes * plan.sequences as u64
        + plan.extra_sequence_bytes * (plan.sequences as u64 - 1)
        + plan.runtime_reserve_bytes
        + plan.scratch_bytes
}

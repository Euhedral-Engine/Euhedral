//! What leaves the device when the model does not fit, on the tensor table of the RCO artifact with fake device sizes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::ArtifactProfile;
use euhedral_inference_engine::residency::accounting::Consumer;
use euhedral_inference_engine::residency::offload::{
    EMBEDDING, FAMILIES, HOST_ALIGNMENT, OUTPUT_HEAD, mapped, select, stageable,
};
use euhedral_inference_engine::residency::planner::{
    ARENA_ALIGNMENT, PlanInputs, ResidencyPlan, Tier, plan_closest, plan_with, sequences_for,
};
use euhedral_inference_settings::{Auto, ByteSize};

mod common;

use common::{config, rco_table, settings};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn inputs(device_free: u64, host: Option<u64>) -> PlanInputs {
    PlanInputs {
        device_free,
        multiprocessors: Some(70),
        host_pinned_bytes: host,
        sequences: 1,
    }
}

fn resident(max_tokens: u32) -> ResidencyPlan {
    plan_with(
        &config(),
        &rco_table(),
        &settings(max_tokens),
        inputs(64 * GIB, Some(u64::MAX)),
    )
    .unwrap()
}

fn profile() -> ArtifactProfile {
    ArtifactProfile::of(&rco_table(), &config()).unwrap()
}

#[test]
fn a_resident_plan_is_unchanged() {
    let plan = resident(4096);
    assert!(plan.offload.is_empty());
    assert_eq!(plan.offload.host_bytes(), 0);
    assert!(plan.device.iter().all(|placement| placement.tier == Tier::Device));
    // The arena is the placements in artifact order, each aligned: nothing was moved.
    let mut next = 0;
    for placement in &plan.device {
        assert_eq!(placement.offset, next, "{}", placement.name);
        next += placement.bytes.div_ceil(ARENA_ALIGNMENT) * ARENA_ALIGNMENT;
    }
    assert_eq!(plan.arena_bytes, next);
    assert_eq!(
        plan.weights_bytes,
        plan.device.iter().map(|placement| placement.bytes).sum::<u64>()
    );
    assert_eq!(plan.location_bytes().host, plan.kv_host_bytes);
    let consumers = plan.consumers();
    let of = |consumer: Consumer| consumers.iter().find(|asked| asked.consumer == consumer).unwrap().bytes;
    assert_eq!(of(Consumer::StagingRing), 0);
    assert_eq!(of(Consumer::HostWeights), 0);
    assert_eq!(of(Consumer::DiskWindow), 0);
}

#[test]
fn a_device_one_byte_short_gives_up_the_embedding_alone() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    let plan = plan_with(&config, &table, &settings, inputs(need - 1, Some(u64::MAX))).unwrap();
    assert_eq!(plan.offload.host, [EMBEDDING]);
    assert!(plan.offload.disk.is_empty());
    // The embedding is read in place: no staging ring, no window.
    assert_eq!(plan.offload.staging, None);
    assert_eq!(plan.offload.window, None);
    assert!(plan.device_bytes < need);
    assert_eq!(plan.placement(EMBEDDING).unwrap().tier, Tier::Host { offset: 0 });
    assert_eq!(
        plan.offload.host_arena_bytes,
        plan.placement(EMBEDDING)
            .unwrap()
            .bytes
            .next_multiple_of(HOST_ALIGNMENT)
    );
    assert_eq!(plan.offloaded_bytes(), plan.placement(EMBEDDING).unwrap().bytes);
    // The device arena holds what stayed, in artifact order and aligned.
    let mut next = 0;
    for placement in plan.on_device() {
        assert_eq!(placement.offset, next);
        next += placement.bytes.div_ceil(ARENA_ALIGNMENT) * ARENA_ALIGNMENT;
    }
    assert_eq!(plan.arena_bytes, next);
}

#[test]
fn the_cache_pages_leave_before_any_weight() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX))).unwrap();
    let plan = plan_with(
        &config,
        &table,
        &settings,
        inputs(whole.device_bytes - 1, Some(u64::MAX)),
    )
    .unwrap();
    assert!(plan.kv_host_tokens > 0 && plan.offload.is_empty());
    // Short of the spilled plan too, the weights follow, and the pages stay on the host.
    let deeper = plan_with(
        &config,
        &table,
        &settings,
        inputs(plan.device_bytes - 1, Some(u64::MAX)),
    )
    .unwrap();
    assert_eq!(deeper.kv_host_bytes, plan.kv_host_bytes);
    assert_eq!(deeper.offload.host, [EMBEDDING]);
}

#[test]
fn projections_follow_the_embedding_smallest_families_first_and_the_head_last() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    let plan = plan_with(&config, &table, &settings, inputs(need - 700 * MIB, Some(u64::MAX))).unwrap();
    assert_eq!(plan.offload.host[0], EMBEDDING);
    assert!(plan.offload.host.len() > 1);
    assert!(
        plan.offload.host[1..]
            .iter()
            .all(|name| name.ends_with("/gdn/query_key"))
    );
    let ring = plan.offload.staging.unwrap();
    assert_eq!(ring.slots, 4);
    assert_eq!(ring.slot_bytes % ARENA_ALIGNMENT, 0);
    // Every byte that left the device is accounted: the arena shrank by the offloaded bytes and grew by the ring.
    assert!(plan.device_bytes <= need - 700 * MIB);

    let deepest = plan_closest(&config, &table, &settings, inputs(GIB, Some(u64::MAX))).unwrap();
    assert!(deepest.1.is_some());
    let taken: Vec<&String> = deepest.0.offload.host.iter().chain(&deepest.0.offload.disk).collect();
    assert_eq!(taken.last().map(|name| name.as_str()), Some(OUTPUT_HEAD));
    assert!(
        taken
            .iter()
            .all(|name| !name.starts_with("mtp/") && *name != "text/draft_head")
    );
    assert!(taken.contains(&&"text/layers/0/mlp/gate_up".to_string()));
}

#[test]
fn the_staging_slots_are_the_setting() {
    let (config, table) = (config(), rco_table());
    let need = resident(4096).device_bytes;
    let mut settings = settings(4096);
    let free = need - 700 * MIB;
    let auto = plan_with(&config, &table, &settings, inputs(free, Some(u64::MAX))).unwrap();
    settings.memory.staging_slots = Auto::Value(6);
    let six = plan_with(&config, &table, &settings, inputs(free, Some(u64::MAX))).unwrap();
    assert_eq!(auto.offload.staging.unwrap().slots, 4);
    assert_eq!(six.offload.staging.unwrap().slots, 6);
    assert_eq!(
        auto.offload.staging.unwrap().slot_bytes,
        six.offload.staging.unwrap().slot_bytes
    );
}

#[test]
fn a_host_budget_beyond_the_plan_leaves_objects_in_the_file() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    // 3 GiB short of the device: about 3 GiB leave; a host budget of 1 GiB holds the embedding and some projections.
    let plan = plan_with(&config, &table, &settings, inputs(need - 3 * GIB, Some(GIB))).unwrap();
    assert!(!plan.offload.disk.is_empty());
    assert!(plan.offload.host_bytes() <= GIB);
    let window = plan.offload.window.unwrap();
    assert_eq!(window.slots, 2 * plan.offload.staging.unwrap().slots);
    assert!(plan.offload.holds_on_host(EMBEDDING));
    for name in &plan.offload.disk {
        assert_eq!(plan.placement(name).unwrap().tier, Tier::Disk);
    }
    // Host objects start on aligned offsets, in artifact order, one after the other.
    let mut next = 0;
    for placement in &plan.device {
        if let Tier::Host { offset } = placement.tier {
            assert_eq!(offset, next, "{}", placement.name);
            assert_eq!(offset % HOST_ALIGNMENT, 0);
            next += placement.bytes.next_multiple_of(HOST_ALIGNMENT);
        }
    }
    assert_eq!(next, plan.offload.host_arena_bytes);
    // The account asks the host for the arena and the window.
    let consumers = plan.consumers();
    let host: u64 = consumers
        .iter()
        .filter(|asked| asked.host)
        .map(|asked| asked.bytes)
        .sum();
    assert_eq!(
        host,
        plan.kv_host_bytes + plan.offload.host_bytes() + plan.prefix_cache_bytes
    );
    let device: u64 = consumers
        .iter()
        .filter(|asked| !asked.host)
        .map(|asked| asked.bytes)
        .sum();
    assert_eq!(device, plan.device_bytes);
}

#[test]
fn a_host_budget_below_the_weights_is_refused_naming_the_memory_that_is_short() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    let Err(error) = plan_with(&config, &table, &settings, inputs(need - 3 * GIB, Some(64 * MIB))) else {
        panic!("expected a refusal");
    };
    let PlanError::HostTooSmall { needed, budget } = error else {
        panic!("expected HostTooSmall, got {error:?}");
    };
    assert_eq!(budget, 64 * MIB);
    assert!(needed > budget);
    let text = PlanError::HostTooSmall { needed, budget }.to_string();
    assert!(
        text.contains("64 MiB host budget")
            && text.contains("memory.host-weights-limit")
            && text.contains("context.max-tokens")
            && text.contains("batching.max-sequences"),
        "{text}"
    );

    // A device too small for what cannot leave it is the device's refusal, with the plan's figures.
    let Err(error) = plan_with(&config, &table, &settings, inputs(GIB, Some(u64::MAX))) else {
        panic!("expected a refusal");
    };
    let PlanError::DeviceTooSmallWithOffload {
        available,
        free,
        sequences,
        context_tokens,
        ..
    } = error
    else {
        panic!("expected DeviceTooSmallWithOffload, got {error:?}");
    };
    assert_eq!((available, free, sequences, context_tokens), (GIB, GIB, 1, 4096));
    let text = error.to_string();
    assert!(
        text.contains("a context of 4096 tokens does not fit in the 1024 MiB free on this GPU"),
        "{text}"
    );
    assert!(!text.contains("fewer sequences"), "{text}");

    let mut two = inputs(GIB, Some(u64::MAX));
    two.sequences = 2;
    let text = plan_with(&config, &table, &settings, two).unwrap_err().to_string();
    assert!(
        text.contains("2 sequences of 4096 tokens")
            && text.contains("or fewer sequences (batching.max-sequences)")
            && text.contains("context.max-tokens"),
        "{text}"
    );
}

#[test]
fn an_unknown_host_budget_refuses_a_plan_that_needs_host_memory_and_names_the_bytes() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    let known = plan_with(&config, &table, &settings, inputs(need - 1, Some(u64::MAX))).unwrap();
    match plan_with(&config, &table, &settings, inputs(need - 1, None)) {
        Err(PlanError::HostBudgetUnknown { needed }) => assert_eq!(needed, known.offload.host_bytes()),
        other => panic!("expected HostBudgetUnknown, got {other:?}"),
    }
}

#[test]
fn the_staging_slots_do_not_alter_a_resident_plan() {
    let (config, table) = (config(), rco_table());
    let mut one = settings(4096);
    one.memory.staging_slots = Auto::Value(1);
    let plan = plan_with(&config, &table, &one, inputs(64 * GIB, Some(0))).unwrap();
    assert!(plan.offload.is_empty());
}

#[test]
fn auto_sequences_never_offload_more_than_one_sequence_does() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = resident(4096).device_bytes;
    // Short by 80 MiB: the embedding leaves and the room it frees holds a second sequence, so the search has more
    // than one to find.
    let free = need - 80 * MIB;
    let single = plan_with(&config, &table, &settings, inputs(free, Some(u64::MAX))).unwrap();
    assert_eq!(single.offload.host, [EMBEDDING]);
    let found = sequences_for(&config, &table, &settings, inputs(free, Some(u64::MAX)), 102).unwrap();
    assert!(found > 1, "the search finds {found}");
    let chosen = PlanInputs {
        sequences: found,
        ..inputs(free, Some(u64::MAX))
    };
    // The count plans, and with no more offloaded than one sequence has.
    let at = plan_with(&config, &table, &settings, chosen).unwrap();
    assert!(at.offloaded_bytes() <= single.offloaded_bytes());
    assert_eq!(at.offload.host, single.offload.host);
    // One more is refused or offloads more.
    let more = plan_with(
        &config,
        &table,
        &settings,
        PlanInputs {
            sequences: found + 1,
            ..chosen
        },
    );
    assert!(
        more.map_or(true, |plan| plan.offloaded_bytes() > single.offloaded_bytes()),
        "{found} sequences is the largest count that offloads no more than one"
    );
    // A device that holds everything: the cap is the answer.
    assert_eq!(
        sequences_for(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX)), 3).unwrap(),
        3
    );
}

#[test]
fn a_device_too_small_for_one_sequence_is_the_answer_of_the_search() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let error = sequences_for(&config, &table, &settings, inputs(GIB, Some(u64::MAX)), 8).unwrap_err();
    assert!(matches!(error, PlanError::DeviceTooSmallWithOffload { .. }));
}

#[test]
fn only_layer_projections_in_a_family_are_stageable() {
    let table = rco_table();
    let profile = profile();
    let mut staged = 0;
    for entry in table.entries() {
        let name = entry.name();
        let in_family = FAMILIES
            .iter()
            .any(|family| name.starts_with("text/layers/") && name.ends_with(&format!("/{family}")));
        assert_eq!(stageable(entry), in_family, "{name}");
        staged += usize::from(stageable(entry));
        assert_eq!(mapped(name), name == EMBEDDING || name == OUTPUT_HEAD);
    }
    assert_eq!(staged, 48 * 3 + 16 * 3 + 64 * 2);
    // The draft's objects are not among the candidates, however much is asked.
    let everything = select(&table, &profile, u64::MAX / 2);
    assert!(
        everything
            .iter()
            .all(|name| name.starts_with("text/layers/") || name == EMBEDDING)
    );
    assert!(!everything.iter().any(|name| name == OUTPUT_HEAD));
    assert_eq!(everything.len(), 1 + staged);
}

#[test]
fn the_selection_takes_the_embedding_then_whole_families_then_spreads_the_last() {
    let table = rco_table();
    let profile = profile();
    assert!(select(&table, &profile, 0).is_empty());
    assert_eq!(select(&table, &profile, 1), [EMBEDDING]);
    let bytes_of = |name: &str| table.get(name).unwrap().bytes();
    let family = |family: &str| -> Vec<String> {
        (0..64)
            .map(|layer| format!("text/layers/{layer}/{family}"))
            .filter(|name| table.get(name).is_some())
            .collect()
    };
    let total = |names: &[String]| names.iter().map(|name| bytes_of(name)).sum::<u64>();
    let (first, second) = (family(FAMILIES[0]), family(FAMILIES[1]));
    let upto = bytes_of(EMBEDDING) + total(&first) + total(&second);
    let both = select(&table, &profile, upto);
    assert_eq!(both.len(), 1 + first.len() + second.len());
    assert_eq!(both[1..=first.len()], first[..]);
    // A request that ends inside the third family takes `count` layers of it at index * n / count.
    let third = family(FAMILIES[2]);
    let size = bytes_of(&third[0]);
    let spread = select(&table, &profile, upto + 3 * size + 1);
    let count = 4;
    let expected: Vec<String> = (0..count)
        .map(|index| third[index * third.len() / count].clone())
        .collect();
    assert_eq!(spread[both.len()..], expected[..]);
    assert_eq!(spread.len(), both.len() + count);
    let layers: Vec<usize> = expected
        .iter()
        .map(|name| name.split('/').nth(2).unwrap().parse().unwrap())
        .collect();
    assert_eq!(layers, [3, 19, 35, 51], "evenly spaced over the 16 attention layers");
}

#[test]
fn a_gpu_limit_below_the_need_offloads_like_a_short_device() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(4096);
    let need = resident(4096).device_bytes;
    settings.memory.gpu_limit = Auto::Value(ByteSize(need - 1));
    let plan = plan_with(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX))).unwrap();
    assert_eq!(plan.offload.host, [EMBEDDING]);
}

/// With the device short, more sequences keep fewer weights on it: each one's state takes room the weights had.
#[test]
fn more_sequences_keep_fewer_weights_on_the_device() {
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let free = resident(4096).device_bytes - 4 * GIB;
    let with = |sequences| {
        plan_with(
            &config,
            &table,
            &settings,
            PlanInputs {
                sequences,
                ..inputs(free, Some(u64::MAX))
            },
        )
        .unwrap()
    };
    let (one, four) = (with(1), with(4));
    assert!(!one.offload.is_empty());
    assert!(four.offload.host.len() + four.offload.disk.len() >= one.offload.host.len() + one.offload.disk.len());
    assert!(four.offloaded_bytes() >= one.offloaded_bytes());
    assert!(four.device_bytes <= free && one.device_bytes <= free);
}

/// A row of the reference planner's refusals where the device kind is named although the device's figures fit: the
/// cache pages put on the host leave the host short. The kind stays the reference's; the message carries the host
/// figures so the shortfall is visible.
#[test]
fn a_device_refusal_that_is_short_on_the_host_carries_the_host_figures() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(32_768);
    settings.context.kv_cache_on_gpu_tokens = 8192;
    let need = plan_with(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX)))
        .unwrap()
        .device_bytes;
    let host = 600 * MIB;
    let error = plan_with(&config, &table, &settings, inputs(need - 542_113_792, Some(host))).unwrap_err();
    let PlanError::DeviceTooSmallWithOffload {
        weights_needed,
        state_needed,
        available,
        host: figures,
        ..
    } = error
    else {
        panic!("expected the device kind of the reference, got {error:?}");
    };
    assert!(
        weights_needed + state_needed <= available,
        "the device's own figures fit"
    );
    assert_eq!(figures.budget, host);
    assert!(figures.cache_pages > 0 && figures.weights > 0);
    assert!(figures.needed() > host);
    let text = error.to_string();
    for figure in [figures.weights, figures.cache_pages, host] {
        assert!(text.contains(&format!("{figure} bytes")), "{text}");
    }
    assert!(text.contains("the host is short too"), "{text}");
}

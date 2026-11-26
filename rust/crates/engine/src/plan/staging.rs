//! The staged form of a view: each use of a weight that is held in host memory reads a slot of the staging ring, and
//! a transfer of its own fills the slot first.
//!
//! The uses of a view are numbered in the order of its stage list. Use `u` reads slot `u mod slots`. Its transfer is
//! listed once the slot is free: after the stage that read the slot for use `u - slots`, or at the start for the first
//! `slots` uses. A transfer has no dependency of its own. The slot is a storage ([`super::hazards`]), so the transfer
//! follows the last stage that read the slot, in this view and, through the edges a quantum adds when it starts, in
//! the quantum before; and the stage that reads the weight follows the transfer, which it also lists as a dependency.
//!
//! The first transfers are listed after the view's first stage, which stays the root a quantum enters through.
//!
//! A weight held on disk is staged the same way; its transfer copies from the disk window, not the host arena. The
//! window has a whole multiple of the ring's slots, and the `k`th disk use of staging slot `s` in a view reads window
//! slot `s + slots * (k mod (window / slots))`: the read of one use of a slot runs while the copy of the last one is
//! not done, and slots of the window are shared by no two staging slots.

use std::sync::Arc;

use euhedral_inference_cuda::{DevicePtr, HostWeightsArena};

use super::PlanBuildError;
use super::builder::{Make, Node};
use crate::model::{Residence, WeightRef};
use crate::residency::disk::DiskUse;
use crate::stages::transfer::DiskTier;

/// The device slots every staged weight is copied into before it is read: `slots` slots of `slot_bytes` bytes each,
/// one after the other from `base`. A slot starts on an address the weights' formats may read from (the ring's
/// geometry aligns `slot_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingRing {
    pub base: DevicePtr,
    pub slots: usize,
    pub slot_bytes: u64,
}

impl StagingRing {
    /// The address of slot `index`.
    pub fn slot(&self, index: usize) -> DevicePtr {
        debug_assert!(index < self.slots, "slot {index} of {}", self.slots);
        self.base.offset(index as u64 * self.slot_bytes)
    }
}

/// Whether the weight is in host memory or on disk, to be copied to a slot before it is read.
fn is_held(weight: &WeightRef) -> bool {
    matches!(weight.residence, Residence::Host { .. } | Residence::Disk { .. })
}

/// The first weight a view reads that is held off the device (in host memory to be staged, or on disk), in name order.
pub(super) fn first_held(nodes: &[Node]) -> Option<String> {
    nodes
        .iter()
        .flat_map(|node| node.make.weights())
        .filter(|weight| matches!(weight.residence, Residence::Host { .. } | Residence::Disk { .. }))
        .map(|weight| weight.name.clone())
        .min()
}

/// `nodes` with each use of a staged weight reading a slot of `ring`, and the transfers that fill the slots; `None`
/// when no node reads a staged weight, which leaves the view as it is.
pub(super) fn staged(
    nodes: &[Node],
    arena: Option<&Arc<HostWeightsArena>>,
    ring: &StagingRing,
    disk: Option<&Arc<DiskTier>>,
) -> Result<Option<Vec<Node>>, PlanBuildError> {
    // The node of each use, in order.
    let mut uses: Vec<usize> = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let weights = node.make.weights();
        for weight in &weights {
            if matches!(weight.residence, Residence::Disk { .. }) && disk.is_none() {
                return Err(PlanBuildError::NoDiskWindow {
                    name: weight.name.clone(),
                });
            }
        }
        let held = weights.iter().filter(|weight| is_held(weight)).count();
        if held == 0 {
            continue;
        }
        // Only a projection read through one pointer is staged; the first stage, the root of the view, reads none.
        match &node.make {
            Make::Linear { .. } if held == 1 && index > 0 => {
                if ring.slots == 0 {
                    return Err(PlanBuildError::NoSlots {
                        name: weights
                            .iter()
                            .find(|weight| is_held(weight))
                            .map(|weight| weight.name.clone())
                            .unwrap_or_default(),
                    });
                }
                uses.push(index);
            }
            _ => {
                let name = weights.iter().find(|weight| is_held(weight));
                return Err(PlanBuildError::NotStageable {
                    name: name.map(|weight| weight.name.clone()).unwrap_or_default(),
                });
            }
        }
    }
    if uses.is_empty() {
        return Ok(None);
    }
    if let Some(tier) = disk {
        let window = tier.geometry().slots;
        if window < ring.slots || !window.is_multiple_of(ring.slots) {
            return Err(PlanBuildError::WindowSlots {
                window,
                staging: ring.slots,
            });
        }
    }
    let total = uses.len();
    let mut result: Vec<Node> = Vec::with_capacity(nodes.len() + total);
    let mut remap: Vec<usize> = Vec::with_capacity(nodes.len());
    let mut transfer_of: Vec<usize> = vec![0; total];
    let mut next_use = 0;
    // How many disk uses each staging slot has had, and the view's disk uses so far.
    let mut disk_uses_of: Vec<usize> = vec![0; ring.slots];
    let mut disk_use = 0;
    let mut emit = |result: &mut Vec<Node>, use_index: usize| -> Result<usize, PlanBuildError> {
        let Make::Linear { weight, .. } = &nodes[uses[use_index]].make else {
            unreachable!("a staged use is a linear");
        };
        if weight.bytes > ring.slot_bytes {
            return Err(PlanBuildError::WeightTooLarge {
                name: weight.name.clone(),
                bytes: weight.bytes,
                slot_bytes: ring.slot_bytes,
            });
        }
        let slot = use_index % ring.slots;
        let make = match weight.residence {
            Residence::Host { offset } => Make::Transfer {
                name: weight.name.clone(),
                source: arena.cloned(),
                offset,
                bytes: weight.bytes,
                slot,
                destination: ring.slot(slot),
            },
            Residence::Disk { file_offset } => {
                let Some(tier) = disk else {
                    return Err(PlanBuildError::NoDiskWindow {
                        name: weight.name.clone(),
                    });
                };
                let window = tier.geometry();
                let window_slot = slot + ring.slots * (disk_uses_of[slot] % (window.slots / ring.slots));
                let use_ = DiskUse {
                    name: weight.name.clone(),
                    file_offset,
                    bytes: weight.bytes,
                    slot: window_slot,
                };
                if use_.bytes == 0 || use_.read_bytes() > window.slot_bytes {
                    return Err(PlanBuildError::ReadTooLarge {
                        name: weight.name.clone(),
                        bytes: use_.read_bytes(),
                        slot_bytes: window.slot_bytes,
                    });
                }
                disk_uses_of[slot] += 1;
                disk_use += 1;
                Make::DiskTransfer {
                    tier: Arc::clone(tier),
                    use_,
                    use_index: disk_use - 1,
                    slot,
                    destination: ring.slot(slot),
                }
            }
            _ => unreachable!("a staged use is held in host memory or on disk"),
        };
        let id = result.len();
        result.push(Node { make, deps: Vec::new() });
        Ok(id)
    };
    for (index, node) in nodes.iter().enumerate() {
        let mut deps: Vec<usize> = node.deps.iter().map(|&dependency| remap[dependency]).collect();
        let make = if uses.get(next_use) == Some(&index) {
            let use_index = next_use;
            next_use += 1;
            let Make::Linear {
                weight,
                input,
                output,
                rows,
                swiglu,
            } = &node.make
            else {
                unreachable!("a staged use is a linear");
            };
            deps.push(transfer_of[use_index]);
            Make::Staged {
                slot: use_index % ring.slots,
                inner: Box::new(Make::Linear {
                    weight: WeightRef {
                        ptr: ring.slot(use_index % ring.slots),
                        ..weight.clone()
                    },
                    input: *input,
                    output: *output,
                    rows: *rows,
                    swiglu: *swiglu,
                }),
            }
        } else {
            node.make.clone()
        };
        remap.push(result.len());
        result.push(Node { make, deps });
        if index == 0 {
            for (use_index, transfer) in transfer_of.iter_mut().enumerate().take(ring.slots) {
                *transfer = emit(&mut result, use_index)?;
            }
        }
        // The slot of this node's use is free again: the transfer of the use `slots` later may be listed.
        if let Some(use_index) = next_use.checked_sub(1)
            && uses[use_index] == index
            && use_index + ring.slots < total
        {
            transfer_of[use_index + ring.slots] = emit(&mut result, use_index + ring.slots)?;
        }
    }
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;

    use super::super::buffers::Access;
    use super::super::builder::{WeightLookup, base_nodes};
    use super::super::hazards::with_hazards;
    use super::*;
    use crate::model::{LayerIndex, LayerKind, ModelConfig, Residence, Role, object_key};
    use crate::residency::disk::{DiskReader, ReadDone, ReadPort, ReadSpec};
    use crate::residency::offload::WindowGeometry;
    use euhedral_graph::Completions;
    use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};

    const LAYERS: usize = 8;
    const BYTES: u64 = 1000;
    const SLOT_BYTES: u64 = 1024;

    struct Lookup(HashMap<(LayerIndex, Role), WeightRef>);

    impl WeightLookup for Lookup {
        fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
            self.0.get(&(layer, role))
        }
    }

    fn kinds() -> Vec<LayerKind> {
        (0..LAYERS)
            .map(|layer| {
                if layer % 4 == 3 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::GatedDeltaNet
                }
            })
            .collect()
    }

    fn names() -> Vec<String> {
        let mut names: Vec<String> = ["token_embedding", "output_head", "final_norm"]
            .iter()
            .map(|name| format!("text/{name}"))
            .collect();
        for (layer, kind) in kinds().iter().enumerate() {
            let objects: &[&str] = match kind {
                LayerKind::GatedDeltaNet => &[
                    "gdn/a_projection",
                    "gdn/b_projection",
                    "gdn/a_log",
                    "gdn/dt_bias",
                    "gdn/convolution",
                    "gdn/query_key",
                    "gdn/value_z",
                    "gdn/output",
                    "gdn/norm",
                ],
                LayerKind::FullAttention => &[
                    "attention/query_key",
                    "attention/gate_value",
                    "attention/query_norm",
                    "attention/key_norm",
                    "attention/output",
                ],
            };
            for object in objects
                .iter()
                .chain(&["input_norm", "post_attention_norm", "mlp/gate_up", "mlp/down"])
            {
                names.push(format!("text/layers/{layer}/{object}"));
            }
        }
        names
    }

    /// Where the object at `index` of [`names`] is in the host arena, when it is held there.
    fn offset_of(index: usize) -> u64 {
        4096 * index as u64
    }

    fn lookup(residence: impl Fn(&str, u64) -> Residence) -> Lookup {
        Lookup(
            names()
                .into_iter()
                .enumerate()
                .filter_map(|(index, name)| {
                    let key = object_key(&name, LAYERS)?;
                    Some((
                        key,
                        WeightRef {
                            residence: residence(&name, offset_of(index)),
                            name,
                            shape: vec![1, 1],
                            dtype: TensorDataType::Bf16,
                            format: WeightFormat::Nvfp4,
                            layout: WeightLayout::ContiguousLeV1,
                            ptr: DevicePtr::NULL,
                            bytes: BYTES,
                            trellis: None,
                        },
                    ))
                })
                .collect(),
        )
    }

    fn config() -> ModelConfig {
        ModelConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: LAYERS,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            attention_head_dim: 1,
            intermediate_size: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            rms_norm_epsilon: 1.0,
            rope_theta: 1.0,
            partial_rotary_factor: 1.0,
            max_position_embeddings: 1,
            layer_kinds: kinds(),
            num_experts: 0,
            tie_word_embeddings: false,
            attention_output_gate: true,
            mtp_layer_count: 1,
            draft_depth: None,
        }
    }

    fn ring(slots: usize) -> StagingRing {
        StagingRing {
            base: DevicePtr::NULL.offset(1 << 32),
            slots,
            slot_bytes: SLOT_BYTES,
        }
    }

    /// The nodes of a view over weights of which those whose name ends with one of `held` are in host memory and
    /// those that end with one of `disk` are on disk.
    fn base_of(held: &[&str], disk: &[&str]) -> Vec<Node> {
        let ends = |name: &str, suffixes: &[&str]| suffixes.iter().any(|suffix| name.ends_with(suffix));
        let lookup = lookup(|name, offset| {
            if ends(name, held) {
                Residence::Host { offset }
            } else if ends(name, disk) {
                Residence::Disk { file_offset: offset }
            } else if name.ends_with("token_embedding") || name.ends_with("output_head") {
                Residence::Mapped
            } else {
                Residence::Device
            }
        });
        base_nodes(&config(), &lookup).unwrap()
    }

    fn base() -> Vec<Node> {
        base_of(&[], &[])
    }

    fn staged_of(suffixes: &[&str], slots: usize) -> Vec<Node> {
        staged(&base_of(suffixes, &[]), None, &ring(slots), None)
            .unwrap()
            .unwrap()
    }

    /// The ids of the transfers and of the staged readers, in list order.
    fn accessors(nodes: &[Node]) -> (Vec<usize>, Vec<usize>) {
        let mut transfers = Vec::new();
        let mut readers = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            match node.make {
                Make::Transfer { .. } => transfers.push(index),
                Make::Staged { .. } => readers.push(index),
                _ => {}
            }
        }
        (transfers, readers)
    }

    #[test]
    fn an_empty_selection_stages_nothing() {
        // Objects read in place over the bus are not staged either.
        assert!(staged(&base(), None, &ring(3), None).unwrap().is_none());
        assert_eq!(first_held(&base()), None);
    }

    #[test]
    fn the_consumer_reads_the_slot_its_transfer_wrote() {
        let slots = 3;
        let nodes = staged_of(&["/gdn/output", "/mlp/down"], slots);
        let ring = ring(slots);
        let (transfers, readers) = accessors(&nodes);
        // Six GDN layers' outputs and eight FFN downs.
        assert_eq!(readers.len(), 6 + LAYERS);
        assert_eq!(transfers.len(), readers.len(), "one transfer for each use");
        for (use_index, (&transfer, &reader)) in transfers.iter().zip(&readers).enumerate() {
            let slot = use_index % slots;
            let Make::Transfer {
                name,
                offset,
                bytes,
                slot: written,
                destination,
                ..
            } = &nodes[transfer].make
            else {
                unreachable!()
            };
            let Make::Staged { slot: read, inner } = &nodes[reader].make else {
                unreachable!()
            };
            let Make::Linear { weight, .. } = &**inner else {
                panic!("a staged use is a linear")
            };
            assert_eq!((*written, *read), (slot, slot), "use {use_index}");
            assert_eq!(*destination, ring.slot(slot));
            assert_eq!(
                weight.ptr,
                ring.slot(slot),
                "the consumer reads the slot its transfer writes"
            );
            assert_eq!(&weight.name, name);
            assert_eq!((*bytes, weight.bytes), (BYTES, BYTES));
            assert_eq!(
                *offset,
                offset_of(names().iter().position(|candidate| candidate == name).unwrap())
            );
            assert!(nodes[transfer].deps.is_empty(), "a transfer's edges are the slot's");
            assert!(nodes[reader].deps.contains(&transfer));
            assert!(transfer < reader);
            if use_index >= slots {
                assert!(
                    readers[use_index - slots] < transfer,
                    "a transfer is listed once the slot's last reader is"
                );
            } else {
                assert_eq!(
                    transfer,
                    1 + use_index,
                    "the first transfers follow the view's first stage"
                );
            }
        }
        assert!(
            !matches!(nodes[0].make, Make::Transfer { .. }),
            "the first stage stays a root"
        );
    }

    #[test]
    fn each_slot_is_a_workspace_buffer_whose_first_accessor_is_a_transfer_and_last_a_reader() {
        for slots in [1, 2, 3, 4] {
            let nodes = staged_of(&["/gdn/output", "/mlp/down"], slots);
            let access: Vec<Access> = nodes.iter().map(|node| node.make.access()).collect();
            for slot in 0..slots {
                let touching: Vec<usize> = (0..nodes.len())
                    .filter(|&node| access[node].slot_read == Some(slot) || access[node].slot_write == Some(slot))
                    .collect();
                let first = *touching.first().unwrap();
                let last = *touching.last().unwrap();
                assert!(
                    access[first].slot_write.is_some(),
                    "slot {slot}: the first accessor fills it"
                );
                assert!(
                    access[last].slot_read.is_some(),
                    "slot {slot}: the last accessor reads it"
                );
            }
        }
    }

    #[test]
    fn a_transfer_follows_the_last_reader_of_its_slot_and_a_reader_its_transfer() {
        let slots = 2;
        let nodes = staged_of(&["/mlp/down"], slots);
        let access: Vec<Access> = nodes.iter().map(|node| node.make.access()).collect();
        let deps: Vec<Vec<usize>> = nodes.iter().map(|node| node.deps.clone()).collect();
        let regions = access
            .iter()
            .filter_map(|touched| touched.scratch)
            .max()
            .map_or(0, |region| region + 1);
        let edges = with_hazards(&access, &deps, regions, slots);
        let (transfers, readers) = accessors(&nodes);
        for (use_index, &transfer) in transfers.iter().enumerate() {
            if use_index < slots {
                assert!(
                    edges[transfer].is_empty(),
                    "the first transfer of a slot waits for nothing"
                );
            } else {
                assert!(
                    edges[transfer].contains(&readers[use_index - slots]),
                    "transfer {use_index} follows the reader of use {}",
                    use_index - slots
                );
            }
        }
    }

    #[test]
    fn a_ring_with_no_slot_is_refused_naming_the_weight() {
        let error = staged(&base_of(&["/mlp/down"], &[]), None, &ring(0), None).unwrap_err();
        assert!(matches!(error, PlanBuildError::NoSlots { .. }), "{error}");
        assert_eq!(
            error.to_string(),
            "text/layers/0/mlp/down is held in host memory and the staging ring has no slot"
        );
    }

    #[test]
    fn a_weight_larger_than_a_slot_is_refused() {
        let mut tight = ring(3);
        tight.slot_bytes = BYTES - 1;
        let error = staged(&base_of(&["/mlp/down"], &[]), None, &tight, None).unwrap_err();
        assert!(
            matches!(error, PlanBuildError::WeightTooLarge { bytes: BYTES, .. }),
            "{error}"
        );
    }

    #[test]
    fn a_host_weight_no_projection_reads_is_refused() {
        let error = staged(&base_of(&["/gdn/a_projection"], &[]), None, &ring(3), None).unwrap_err();
        assert!(matches!(error, PlanBuildError::NotStageable { .. }), "{error}");
        assert_eq!(
            error.to_string(),
            "text/layers/0/gdn/a_projection is held in host memory but no stage of the plan can read it from a staging slot"
        );
    }

    #[test]
    fn a_disk_weight_is_refused() {
        let nodes = base_of(&[], &["layers/0/mlp/down"]);
        assert_eq!(first_held(&nodes).as_deref(), Some("text/layers/0/mlp/down"));
        let error = staged(&nodes, None, &ring(3), None).unwrap_err();
        assert!(matches!(error, PlanBuildError::NoDiskWindow { .. }), "{error}");
        assert_eq!(
            error.to_string(),
            "text/layers/0/mlp/down is held on disk and the plan was given no disk window"
        );
    }

    struct NoPort;

    impl ReadPort for NoPort {
        fn start(&self, _spec: ReadSpec, _done: ReadDone) -> io::Result<()> {
            Err(io::Error::other("a plan test reads nothing"))
        }
        fn has_direct(&self) -> bool {
            false
        }
        fn file_len(&self) -> u64 {
            u64::MAX
        }
    }

    /// A tier of `slots` window slots of `slot_bytes` that registers and reads nothing.
    fn tier(slots: usize, slot_bytes: u64) -> Arc<DiskTier> {
        let geometry = WindowGeometry { slots, slot_bytes };
        let reader = DiskReader::new(Arc::new(NoPort), geometry, Box::new(|_| false));
        DiskTier::detached(reader, geometry, Completions::without_lanes())
    }

    fn disk_nodes(tier: &Arc<DiskTier>, ring_slots: usize) -> Result<Vec<Node>, PlanBuildError> {
        staged(&base_of(&[], &["/mlp/down"]), None, &ring(ring_slots), Some(tier)).map(Option::unwrap)
    }

    #[test]
    fn disk_uses_transfer_through_the_window_in_plan_order() {
        let (slots, window) = (3, 6);
        let tier = tier(window, 4096);
        let nodes = disk_nodes(&tier, slots).unwrap();
        let mut seen = vec![0usize; slots];
        let mut count = 0;
        for node in &nodes {
            let Make::DiskTransfer {
                use_,
                use_index,
                slot,
                destination,
                ..
            } = &node.make
            else {
                continue;
            };
            assert_eq!(*use_index, count, "the uses are numbered in plan order");
            assert_eq!(*slot, count % slots);
            assert_eq!(*destination, ring(slots).slot(*slot));
            // The k-th use of a staging slot reads the window slot s + S * (k mod W/S).
            let expected = *slot + slots * (seen[*slot] % (window / slots));
            assert_eq!(use_.slot, expected, "use {count}");
            assert_eq!(use_.bytes, BYTES);
            seen[*slot] += 1;
            count += 1;
        }
        assert_eq!(count, LAYERS, "a disk use for each FFN down");
        assert!(seen.iter().any(|&uses| uses > window / slots), "the rotation wraps");
    }

    #[test]
    fn a_disk_backed_plan_needs_a_window() {
        let error = staged(&base_of(&[], &["/mlp/down"]), None, &ring(3), None).unwrap_err();
        assert!(matches!(error, PlanBuildError::NoDiskWindow { .. }), "{error}");
    }

    #[test]
    fn a_window_that_is_not_a_whole_multiple_of_the_ring_is_refused() {
        for window in [2, 4, 7] {
            let error = disk_nodes(&tier(window, 4096), 3).unwrap_err();
            assert!(
                matches!(error, PlanBuildError::WindowSlots { window: w, staging: 3 } if w == window),
                "{window}: {error}"
            );
        }
    }

    #[test]
    fn a_read_larger_than_a_window_slot_is_refused() {
        let error = disk_nodes(&tier(6, 2048), 3).unwrap_err();
        assert!(matches!(error, PlanBuildError::ReadTooLarge { .. }), "{error}");
    }

    #[test]
    fn a_disk_object_larger_than_a_ring_slot_is_refused() {
        let mut tight = ring(3);
        tight.slot_bytes = BYTES - 1;
        let error = staged(&base_of(&[], &["/mlp/down"]), None, &tight, Some(&tier(6, 4096))).unwrap_err();
        assert!(matches!(error, PlanBuildError::WeightTooLarge { .. }), "{error}");
    }
}

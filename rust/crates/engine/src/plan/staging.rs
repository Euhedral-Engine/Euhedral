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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use euhedral_inference_cuda::{DevicePtr, HostWeightsArena};

use super::PlanBuildError;
use super::builder::{Make, Node};
use crate::model::WeightRef;
use crate::residency::offload::mapped;
use crate::residency::planner::{ResidencyPlan, Tier};

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

/// Where the weights that are not on the device are held.
pub(super) struct HostSet {
    host: HashSet<String>,
    disk: HashSet<String>,
    offsets: HashMap<String, u64>,
    arena: Option<Arc<HostWeightsArena>>,
}

impl HostSet {
    /// The host and disk objects of `residency`, staged from `arena`.
    pub(super) fn of(residency: &ResidencyPlan, arena: Option<Arc<HostWeightsArena>>) -> HostSet {
        let offsets = residency
            .device
            .iter()
            .filter_map(|placement| match placement.tier {
                Tier::Host { offset } => Some((placement.name.clone(), offset)),
                Tier::Device | Tier::Disk => None,
            })
            .collect();
        HostSet {
            host: residency
                .offload
                .host
                .iter()
                .filter(|name| !mapped(name))
                .cloned()
                .collect(),
            disk: residency.offload.disk.iter().cloned().collect(),
            offsets,
            arena,
        }
    }

    /// Whether the plan holds no weight that a transfer stages.
    pub(super) fn is_empty(&self) -> bool {
        self.host.is_empty() && self.disk.is_empty()
    }
}

/// `nodes` with each use of a staged weight reading a slot of `ring`, and the transfers that fill the slots; `None`
/// when no node reads a staged weight, which leaves the view as it is.
pub(super) fn staged(nodes: &[Node], host: &HostSet, ring: &StagingRing) -> Result<Option<Vec<Node>>, PlanBuildError> {
    // The node of each use, in order.
    let mut uses: Vec<usize> = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let weights = node.make.weights();
        for weight in &weights {
            if host.disk.contains(&weight.name) {
                return Err(PlanBuildError::OnDisk {
                    name: weight.name.clone(),
                });
            }
        }
        let held = weights.iter().filter(|weight| host.host.contains(&weight.name)).count();
        if held == 0 {
            continue;
        }
        if held > ring.slots {
            return Err(PlanBuildError::TooManyUses {
                node: index,
                uses: held,
                slots: ring.slots,
            });
        }
        // Only a projection read through one pointer is staged; the first stage, the root of the view, reads none.
        match &node.make {
            Make::Linear { .. } if held == 1 && index > 0 => uses.push(index),
            _ => {
                let name = weights.iter().find(|weight| host.host.contains(&weight.name));
                return Err(PlanBuildError::NotStageable {
                    name: name.map(|weight| weight.name.clone()).unwrap_or_default(),
                });
            }
        }
    }
    if uses.is_empty() {
        return Ok(None);
    }
    let total = uses.len();
    let mut result: Vec<Node> = Vec::with_capacity(nodes.len() + total);
    let mut remap: Vec<usize> = Vec::with_capacity(nodes.len());
    let mut transfer_of: Vec<usize> = vec![0; total];
    let mut next_use = 0;
    let emit = |result: &mut Vec<Node>, use_index: usize| -> Result<usize, PlanBuildError> {
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
        let offset = host
            .offsets
            .get(&weight.name)
            .copied()
            .ok_or_else(|| PlanBuildError::NotPlacedOnHost {
                name: weight.name.clone(),
            })?;
        let slot = use_index % ring.slots;
        let id = result.len();
        result.push(Node {
            make: Make::Transfer {
                name: weight.name.clone(),
                source: host.arena.clone(),
                offset,
                bytes: weight.bytes,
                slot,
                destination: ring.slot(slot),
            },
            deps: Vec::new(),
        });
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
    use super::super::buffers::Access;
    use super::super::builder::{WeightLookup, base_nodes};
    use super::super::hazards::with_hazards;
    use super::*;
    use crate::model::{LayerIndex, LayerKind, ModelConfig, Role, object_key};
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

    fn lookup() -> Lookup {
        Lookup(
            names()
                .into_iter()
                .filter_map(|name| {
                    let key = object_key(&name, LAYERS)?;
                    Some((
                        key,
                        WeightRef {
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

    /// The host set of the objects whose name ends with one of `suffixes`, each at its own offset.
    fn host(suffixes: &[&str]) -> HostSet {
        let chosen: Vec<String> = names()
            .into_iter()
            .filter(|name| suffixes.iter().any(|suffix| name.ends_with(suffix)))
            .collect();
        HostSet {
            offsets: chosen
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), 4096 * index as u64))
                .collect(),
            host: chosen.into_iter().collect(),
            disk: HashSet::new(),
            arena: None,
        }
    }

    fn base() -> Vec<Node> {
        base_nodes(&config(), &lookup()).unwrap()
    }

    fn staged_of(suffixes: &[&str], slots: usize) -> (Vec<Node>, HostSet) {
        let host = host(suffixes);
        let nodes = staged(&base(), &host, &ring(slots)).unwrap().unwrap();
        (nodes, host)
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
        let nodes = base();
        assert!(staged(&nodes, &host(&[]), &ring(3)).unwrap().is_none());
        // Only the objects that are read in place are held on the host: nothing to stage.
        let mapped_only = host(&["token_embedding", "output_head"]);
        let set = HostSet {
            host: mapped_only.host.iter().filter(|name| !mapped(name)).cloned().collect(),
            ..mapped_only
        };
        assert!(staged(&nodes, &set, &ring(3)).unwrap().is_none());
    }

    #[test]
    fn the_consumer_reads_the_slot_its_transfer_wrote() {
        let slots = 3;
        let (nodes, host) = staged_of(&["/gdn/output", "/mlp/down"], slots);
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
            assert_eq!(*offset, host.offsets[name]);
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
            let (nodes, _) = staged_of(&["/gdn/output", "/mlp/down"], slots);
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
        let (nodes, _) = staged_of(&["/mlp/down"], slots);
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
    fn a_node_that_reads_more_weights_than_there_are_slots_is_refused() {
        let error = staged(&base(), &host(&["/mlp/down"]), &ring(0)).unwrap_err();
        assert!(
            matches!(error, PlanBuildError::TooManyUses { uses: 1, slots: 0, .. }),
            "{error}"
        );
    }

    #[test]
    fn a_weight_larger_than_a_slot_is_refused() {
        let mut tight = ring(3);
        tight.slot_bytes = BYTES - 1;
        let error = staged(&base(), &host(&["/mlp/down"]), &tight).unwrap_err();
        assert!(
            matches!(error, PlanBuildError::WeightTooLarge { bytes: BYTES, .. }),
            "{error}"
        );
    }

    #[test]
    fn a_host_weight_no_projection_reads_is_refused() {
        let error = staged(&base(), &host(&["/gdn/a_projection"]), &ring(3)).unwrap_err();
        assert!(matches!(error, PlanBuildError::NotStageable { .. }), "{error}");
    }

    #[test]
    fn a_staged_weight_the_residency_plan_does_not_place_is_refused() {
        let mut set = host(&["/mlp/down"]);
        set.offsets.clear();
        let error = staged(&base(), &set, &ring(3)).unwrap_err();
        assert!(matches!(error, PlanBuildError::NotPlacedOnHost { .. }), "{error}");
    }

    #[test]
    fn a_disk_weight_is_refused() {
        let mut set = host(&[]);
        set.disk.insert("text/layers/0/mlp/down".into());
        let error = staged(&base(), &set, &ring(3)).unwrap_err();
        assert!(matches!(error, PlanBuildError::OnDisk { .. }), "{error}");
    }
}

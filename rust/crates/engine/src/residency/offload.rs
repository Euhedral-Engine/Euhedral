//! What leaves the device when the model does not fit: which objects, and where each goes instead.
//!
//! The plan moves weights last. The attention cache pages past `context.kv-cache-on-gpu-tokens` go to pinned host
//! memory first ([`planner`](super::planner)); only when the weights and the device share still do not fit does a
//! weight leave. Every offloaded byte costs decode time, so the smallest selection that makes room is taken:
//! - the token embedding first: a gather, so kernels read the few rows they need in place over the bus;
//! - then layer projections, whole families in a fixed order (smallest tensor first), the last family spread evenly
//!   over the layers so that transfers interleave with resident work. Every projection is read once per decode
//!   token, so each offloaded byte costs the same transfer time whichever tensor it belongs to; the choice therefore
//!   minimises the staging ring (the device slots a projection is copied into before it is read);
//! - the output head last, read in place like the embedding, when everything else offloadable is not enough.
//!
//! The selection is then split between pinned host memory and the artifact file ([`Tier`]): the embedding and the head
//! always go to host memory (they are read in place), each other object goes to host memory while the arena and the
//! window the rest would need fit the host budget, and from the first that does not, every object stays in the file
//! and is read through the window.
//!
//! This module is arithmetic over object sizes: it reads no machine and allocates nothing.
//!
//! [`Tier`]: super::planner::Tier

use euhedral_inference_edrl::{TensorEntry, TensorTable, WeightFormat, WeightLayout};

use crate::model::ArtifactProfile;
use crate::residency::planner::{ARENA_ALIGNMENT, Placement};

/// The token embedding, the one object that is always read in place.
pub const EMBEDDING: &str = "text/token_embedding";

/// The output head, the other object read in place, taken last.
pub const OUTPUT_HEAD: &str = "text/output_head";

/// The layer projections that may leave the device, in selection order: the smallest tensors first.
pub const FAMILIES: [&str; 8] = [
    "gdn/query_key",
    "gdn/output",
    "attention/output",
    "attention/query_key",
    "attention/gate_value",
    "gdn/value_z",
    "mlp/down",
    "mlp/gate_up",
];

/// The bytes the selection grows by when it looks for the smallest one that fits.
///
/// This is not a setting: it is the granularity of a search, and it decides at most this many bytes of extra offload,
/// whatever the resource. It is kept at the value the reference planner searches with so that both plan the same
/// selection for the same inputs.
pub const SELECTION_STEP: u64 = 64 << 20;

/// Host objects start on this many bytes of the host arena, and a disk read is whole blocks of this many bytes.
///
/// This is not a setting: it is the block a direct read of the artifact file must be aligned to, a property of the
/// file system interface and not a resource.
pub const HOST_ALIGNMENT: u64 = 4096;

/// The most rows of one verification that an NVFP4 matrix computes on the tensor-core decode kernels: 8 rows in 4
/// groups. Beyond it the row-exact verification runs the scalar reference, which sweeps the weights once per row, so
/// a head read over the bus would be swept once per row.
///
/// This is not a setting: it is the geometry of the NVFP4 decode kernel (its row and group limits), which no resource
/// changes.
pub const MAPPED_HEAD_MAX_ROWS: u64 = 32;

/// The objects of a selection, as the planner sees them: a name, the bytes of the payload and whether a projection may
/// be staged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightObject<'a> {
    pub name: &'a str,
    pub bytes: u64,
    /// A layer projection of one of the [`FAMILIES`] whose format a transfer can stage whole.
    pub stageable: bool,
    /// An NVFP4 matrix.
    pub nvfp4: bool,
}

/// The device slots every staged projection is copied into before it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingGeometry {
    pub slots: usize,
    /// Each slot holds the largest staged object, aligned like a device allocation.
    pub slot_bytes: u64,
}

impl StagingGeometry {
    /// The device bytes of the ring.
    pub fn bytes(&self) -> u64 {
        self.slots as u64 * self.slot_bytes
    }
}

/// The pinned host slots a disk read lands in before it is copied into a staging slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowGeometry {
    /// Two for each staging slot, so the next read into a staging slot's window runs while the last one is copied.
    pub slots: usize,
    /// Each slot holds the aligned blocks around the largest disk object at any file offset.
    pub slot_bytes: u64,
}

impl WindowGeometry {
    /// The pinned host bytes of the window.
    pub fn bytes(&self) -> u64 {
        self.slots as u64 * self.slot_bytes
    }
}

/// What leaves the device, and the memory that takes its place. Empty when the plan is resident.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Offload {
    /// The objects held in pinned host memory, in selection order (the output head, when it is taken, last).
    pub host: Vec<String>,
    /// The objects left in the artifact file, in selection order.
    pub disk: Vec<String>,
    /// The ring the staged objects are copied through; `None` when nothing is staged.
    pub staging: Option<StagingGeometry>,
    /// The window the disk objects are read through; `None` without a disk object.
    pub window: Option<WindowGeometry>,
    /// The pinned bytes of the host arena: each host object at a multiple of [`HOST_ALIGNMENT`].
    pub host_arena_bytes: u64,
}

impl Offload {
    /// Nothing leaves the device.
    pub fn is_empty(&self) -> bool {
        self.host.is_empty() && self.disk.is_empty()
    }

    /// Whether the named object is held in pinned host memory.
    pub fn holds_on_host(&self, name: &str) -> bool {
        self.host.iter().any(|held| held == name)
    }

    /// Whether the named object stays in the artifact file.
    pub fn holds_on_disk(&self, name: &str) -> bool {
        self.disk.iter().any(|held| held == name)
    }

    /// The device bytes of the staging ring.
    pub fn staging_bytes(&self) -> u64 {
        self.staging.map_or(0, |ring| ring.bytes())
    }

    /// The pinned host bytes of the disk window.
    pub fn window_bytes(&self) -> u64 {
        self.window.map_or(0, |window| window.bytes())
    }

    /// The pinned host bytes the offload needs: the arena and the window.
    pub fn host_bytes(&self) -> u64 {
        self.host_arena_bytes.saturating_add(self.window_bytes())
    }
}

/// Whether the named object, when held in host memory, is read in place over the bus instead of staged.
pub fn mapped(name: &str) -> bool {
    name == EMBEDDING || name == OUTPUT_HEAD
}

/// The layer and the object name of `text/layers/<layer>/<object>`.
fn layer_object(name: &str) -> Option<(u64, &str)> {
    let rest = name.strip_prefix("text/layers/")?;
    let (layer, object) = rest.split_once('/')?;
    if layer.is_empty() || object.is_empty() || !layer.bytes().all(|digit| digit.is_ascii_digit()) {
        return None;
    }
    Some((layer.parse().ok()?, object))
}

/// Whether `entry` is a layer projection of the [`FAMILIES`] whose format a transfer can stage whole. A P2E2 payload is
/// never staged: its row offsets index the payload where it was loaded. A trellis object's companions (its sign
/// vectors and segment ends) are not projections: a few KiB that every launch of the object reads, they never leave
/// the device.
pub fn stageable(entry: &TensorEntry) -> bool {
    in_a_family(entry.name()) && stages_whole(entry.format(), entry.layout())
}

fn in_a_family(name: &str) -> bool {
    layer_object(name).is_some_and(|(_, object)| FAMILIES.contains(&object))
}

/// Whether a payload of `format` and `layout` is one a copy to another address leaves readable as it was.
fn stages_whole(format: WeightFormat, layout: WeightLayout) -> bool {
    matches!(
        (format, layout),
        (
            WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16,
            WeightLayout::RowSplitK128V1
        ) | (
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128V1 | WeightLayout::RowSplitK128Sd4V1
        ) | (WeightFormat::Bf16, WeightLayout::ContiguousLeV1)
            | (WeightFormat::Trellis, WeightLayout::TrellisTile16V1)
    )
}

/// The objects of `table` the plan's load places, as the planner sees them: one per entry, in table order.
pub fn weight_objects<'a>(table: &'a TensorTable, profile: &ArtifactProfile) -> Vec<WeightObject<'a>> {
    table
        .entries()
        .iter()
        .filter(|entry| profile.uploads(entry.name()))
        .map(|entry| WeightObject {
            name: entry.name(),
            bytes: entry.bytes(),
            stageable: stageable(entry),
            nvfp4: entry.format() == WeightFormat::Nvfp4,
        })
        .collect()
}

/// The objects of a plan's placements, as the planner sees them.
pub(crate) fn placed_objects(placements: &[Placement]) -> Vec<WeightObject<'_>> {
    placements
        .iter()
        .map(|placement| WeightObject {
            name: &placement.name,
            bytes: placement.bytes,
            stageable: in_a_family(&placement.name) && stages_whole(placement.format, placement.layout),
            nvfp4: placement.format == WeightFormat::Nvfp4,
        })
        .collect()
}

/// The names of the objects that are held in host memory to make `bytes` of room: the embedding, then projections, as
/// many as hold at least `bytes` in all (or every candidate when they hold fewer), in selection order. Empty for a
/// request of none.
pub fn select(table: &TensorTable, profile: &ArtifactProfile, bytes: u64) -> Vec<String> {
    let objects = weight_objects(table, profile);
    Selector::new(&objects)
        .select(bytes)
        .into_iter()
        .map(|index| objects[index].name.to_owned())
        .collect()
}

/// The candidates of a selection, indexed once: the selection itself is repeated for every step of the search.
struct Selector<'a, 'b> {
    objects: &'b [WeightObject<'a>],
    embedding: Option<usize>,
    /// The projections of each family, by layer.
    families: Vec<Vec<usize>>,
}

impl<'a, 'b> Selector<'a, 'b> {
    fn new(objects: &'b [WeightObject<'a>]) -> Self {
        let embedding = objects.iter().position(|object| object.name == EMBEDDING);
        let mut families: Vec<Vec<(u64, usize)>> = vec![Vec::new(); FAMILIES.len()];
        for (index, object) in objects.iter().enumerate() {
            if !object.stageable {
                continue;
            }
            let Some((layer, family)) = layer_object(object.name) else {
                continue;
            };
            if let Some(slot) = FAMILIES.iter().position(|name| *name == family) {
                families[slot].push((layer, index));
            }
        }
        let families = families
            .into_iter()
            .map(|mut members| {
                members.sort_by_key(|(layer, _)| *layer);
                members.into_iter().map(|(_, index)| index).collect()
            })
            .collect();
        Selector {
            objects,
            embedding,
            families,
        }
    }

    /// The indices of the objects held in host memory to make `bytes` of room, in selection order.
    fn select(&self, bytes: u64) -> Vec<usize> {
        let mut selected = Vec::new();
        if bytes == 0 {
            return selected;
        }
        let mut remaining = i128::from(bytes);
        if let Some(index) = self.embedding {
            selected.push(index);
            remaining -= i128::from(self.objects[index].bytes);
        }
        if remaining <= 0 {
            return selected;
        }
        for members in &self.families {
            let Some(first) = members.first() else {
                continue;
            };
            let family_bytes: u64 = members.iter().map(|index| self.objects[*index].bytes).sum();
            if i128::from(family_bytes) <= remaining {
                selected.extend(members.iter().copied());
                remaining -= i128::from(family_bytes);
            } else {
                // The last family: as many layers as the room takes, spread evenly over all of them.
                let size = i128::from(self.objects[*first].bytes).max(1);
                let count = (members.len() as i128).min((remaining + size - 1) / size) as usize;
                for index in 0..count {
                    selected.push(members[index * members.len() / count]);
                }
                remaining = 0;
            }
            if remaining <= 0 {
                break;
            }
        }
        selected
    }
}

/// What the offload is planned against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffloadInputs {
    /// The device bytes the plan may use.
    pub device_free: u64,
    /// The device bytes the plan needs beside the weights and the staging ring: the cache, the reserve, the scratch and
    /// the state of the sequences beyond the first.
    pub fixed_bytes: u64,
    /// The pinned host bytes the weights may take (what the cache pages left of the host budget).
    pub host_budget: u64,
    /// The slots of the staging ring (`memory.staging-slots`).
    pub staging_slots: usize,
    /// The slots of the disk window (`memory.disk-window-slots`).
    pub window_slots: usize,
    /// The rows a verification of every sequence holds: sequences times a draft's rows.
    pub head_rows: u64,
}

/// Where the search for an offload ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reach {
    pub offload: Offload,
    /// The device bytes of the objects that stay, each aligned like an arena placement.
    pub resident_bytes: u64,
    /// Whether the device need (the resident objects, the ring and the fixed bytes) fits the device.
    pub device_fits: bool,
    /// Whether the arena and the window fit the host budget.
    pub host_fits: bool,
}

impl Reach {
    /// Whether the plan fits both memories.
    pub fn fits(&self) -> bool {
        self.device_fits && self.host_fits
    }

    /// The pinned host bytes the offload needs.
    pub fn host_bytes(&self) -> u64 {
        self.offload.host_bytes()
    }
}

fn align(value: u64, to: u64) -> u64 {
    value.div_ceil(to) * to
}

/// Plans the offload of `objects`, the objects the load places in artifact order: grows the selection a step at a time
/// until the device need fits `inputs.device_free`, then maps the output head if that is still not enough, then splits
/// the selection between host memory and the file. The result is the plan that comes closest when nothing fits.
pub fn offload(objects: &[WeightObject<'_>], inputs: &OffloadInputs) -> Reach {
    let total_aligned: u64 = objects.iter().map(|object| align(object.bytes, ARENA_ALIGNMENT)).sum();
    let need = |selection: &[usize]| -> (u64, u64) {
        let mut taken = 0u64;
        let mut largest = 0u64;
        for index in selection {
            let object = &objects[*index];
            taken += align(object.bytes, ARENA_ALIGNMENT);
            if !mapped(object.name) {
                largest = largest.max(object.bytes);
            }
        }
        let ring = if largest > 0 {
            inputs.staging_slots as u64 * align(largest, ARENA_ALIGNMENT)
        } else {
            0
        };
        (total_aligned - taken, ring)
    };
    let fits = |resident: u64, ring: u64| {
        resident.saturating_add(ring).saturating_add(inputs.fixed_bytes) <= inputs.device_free
    };
    let reach = |selection: &[usize], device_fits: bool| tiers(objects, selection, inputs, device_fits);

    let (resident, ring) = need(&[]);
    if fits(resident, ring) {
        return reach(&[], true);
    }
    let head = objects.iter().position(|object| object.name == OUTPUT_HEAD);
    let mappable = head.is_some_and(|index| !(objects[index].nvfp4 && inputs.head_rows > MAPPED_HEAD_MAX_ROWS));
    let all: u64 = objects.iter().map(|object| object.bytes).sum();
    let selector = Selector::new(objects);
    let mut bytes = SELECTION_STEP;
    loop {
        let selection = selector.select(bytes);
        let (resident, ring) = need(&selection);
        if fits(resident, ring) {
            return reach(&selection, true);
        }
        if bytes > all {
            // Everything else offloadable is offloaded: the output head goes last.
            let Some(head) = head.filter(|_| mappable) else {
                return reach(&selection, false);
            };
            if selection.contains(&head) {
                return reach(&selection, false);
            }
            let mut with_head = selection;
            with_head.push(head);
            let (resident, ring) = need(&with_head);
            return reach(&with_head, fits(resident, ring));
        }
        bytes += SELECTION_STEP;
    }
}

/// The pinned bytes of a window whose largest object has `largest` bytes, in `slots` slots; 0 for none.
fn window_bytes(slots: usize, largest: u64) -> u64 {
    if largest == 0 {
        0
    } else {
        slots as u64 * window_slot_bytes(largest)
    }
}

/// A window slot that holds the aligned blocks around a payload of `largest` bytes at any file offset.
fn window_slot_bytes(largest: u64) -> u64 {
    align(largest, HOST_ALIGNMENT) + HOST_ALIGNMENT
}

/// Splits `selection` (in selection order) between host memory and the file, and sizes the ring, the arena and the
/// window. The embedding and the output head stay in host memory (read in place); then each object goes to host memory
/// while the arena and the window that the remaining objects would need fit the host budget, and from the first that
/// does not, every object goes to the file.
pub fn tiers(objects: &[WeightObject<'_>], selection: &[usize], inputs: &OffloadInputs, device_fits: bool) -> Reach {
    let slots = inputs.staging_slots;
    let window_slots = inputs.window_slots;
    let budget = inputs.host_budget;
    let slot = |index: usize| align(objects[index].bytes, HOST_ALIGNMENT);
    let resident_bytes = {
        let mut taken = 0u64;
        for index in selection {
            taken += align(objects[*index].bytes, ARENA_ALIGNMENT);
        }
        objects
            .iter()
            .map(|object| align(object.bytes, ARENA_ALIGNMENT))
            .sum::<u64>()
            - taken
    };
    let largest_staged = selection
        .iter()
        .filter(|index| !mapped(objects[**index].name))
        .map(|index| objects[*index].bytes)
        .max()
        .unwrap_or(0);
    let staging = (largest_staged > 0).then(|| StagingGeometry {
        slots,
        slot_bytes: align(largest_staged, ARENA_ALIGNMENT),
    });
    let names = |picked: &[usize]| picked.iter().map(|index| objects[*index].name.to_owned()).collect();
    let total: u64 = selection.iter().map(|index| slot(*index)).sum();
    if total <= budget {
        return Reach {
            offload: Offload {
                host: names(selection),
                disk: Vec::new(),
                staging,
                window: None,
                host_arena_bytes: total,
            },
            resident_bytes,
            device_fits,
            host_fits: true,
        };
    }
    // The largest object at or after each position: the window the objects from there on would need.
    let mut largest_from = vec![0u64; selection.len() + 1];
    for position in (0..selection.len()).rev() {
        let object = &objects[selection[position]];
        let size = if mapped(object.name) { 0 } else { object.bytes };
        largest_from[position] = largest_from[position + 1].max(size);
    }
    // The output head's bytes are in the arena whatever comes before it.
    let head: u64 = selection
        .iter()
        .find(|index| objects[**index].name == OUTPUT_HEAD)
        .map_or(0, |index| slot(*index));
    let (mut host, mut disk) = (Vec::new(), Vec::new());
    let (mut arena, mut largest_disk) = (0u64, 0u64);
    for (position, index) in selection.iter().enumerate() {
        let object = &objects[*index];
        let size = slot(*index);
        let reserved = if mapped(object.name) { 0 } else { head };
        let window = window_bytes(window_slots, largest_from[position + 1]);
        if mapped(object.name)
            || (disk.is_empty()
                && arena
                    .saturating_add(size)
                    .saturating_add(window)
                    .saturating_add(reserved)
                    <= budget)
        {
            host.push(*index);
            arena += size;
        } else {
            disk.push(*index);
            largest_disk = largest_disk.max(object.bytes);
        }
    }
    let window = (largest_disk > 0).then(|| WindowGeometry {
        slots: window_slots,
        slot_bytes: window_slot_bytes(largest_disk),
    });
    let window_total = window.map_or(0, |window| window.bytes());
    Reach {
        offload: Offload {
            host: names(&host),
            disk: names(&disk),
            staging,
            window,
            host_arena_bytes: arena,
        },
        resident_bytes,
        device_fits,
        host_fits: arena.saturating_add(window_total) <= budget,
    }
}

/// The bytes of `objects` named in `offload`, host and disk both: what leaves the device, the output head and the
/// embedding included.
pub fn offloaded_bytes(objects: &[WeightObject<'_>], offload: &Offload) -> u64 {
    objects
        .iter()
        .filter(|object| offload.holds_on_host(object.name) || offload.holds_on_disk(object.name))
        .map(|object| object.bytes)
        .sum()
}

/// The device bytes of the objects of `objects` that the offload leaves on the device, each aligned.
pub fn resident_bytes(objects: &[WeightObject<'_>], offload: &Offload) -> u64 {
    objects
        .iter()
        .filter(|object| !offload.holds_on_host(object.name) && !offload.holds_on_disk(object.name))
        .map(|object| align(object.bytes, ARENA_ALIGNMENT))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn object(name: &str, bytes: u64, stageable: bool) -> WeightObject<'_> {
        WeightObject {
            name,
            bytes,
            stageable,
            nvfp4: true,
        }
    }

    fn names(selected: &[usize], objects: &[WeightObject<'_>]) -> Vec<String> {
        selected.iter().map(|index| objects[*index].name.to_owned()).collect()
    }

    #[test]
    fn layer_objects_are_found_by_their_name() {
        assert_eq!(layer_object("text/layers/12/mlp/down"), Some((12, "mlp/down")));
        assert_eq!(layer_object("text/layers/x/mlp/down"), None);
        assert_eq!(layer_object("text/layers//mlp/down"), None);
        assert_eq!(layer_object("text/layers/3"), None);
        assert_eq!(layer_object("mtp/layer/mlp/down"), None);
        assert!(!mapped("text/draft_head"));
        assert!(mapped(EMBEDDING) && mapped(OUTPUT_HEAD));
    }

    /// The reference planner's weight selection over a small table: 4 layers of a 10-byte and a 100-byte family, an
    /// embedding of 500 and a head of 700.
    #[test]
    fn the_embedding_goes_first_then_the_smallest_projections() {
        let names_of: Vec<String> = (0..4)
            .flat_map(|layer| {
                [
                    format!("text/layers/{layer}/gdn/query_key"),
                    format!("text/layers/{layer}/mlp/down"),
                ]
            })
            .collect();
        let mut objects = vec![object(EMBEDDING, 500, false), object(OUTPUT_HEAD, 700, false)];
        for (position, name) in names_of.iter().enumerate() {
            objects.push(object(name, if position % 2 == 0 { 10 } else { 100 }, true));
        }
        let selector = Selector::new(&objects);
        assert_eq!(selector.select(0), Vec::<usize>::new());
        assert_eq!(names(&selector.select(500), &objects), [EMBEDDING]);
        assert_eq!(
            names(&selector.select(500 + 40 + 150), &objects),
            [
                EMBEDDING,
                "text/layers/0/gdn/query_key",
                "text/layers/1/gdn/query_key",
                "text/layers/2/gdn/query_key",
                "text/layers/3/gdn/query_key",
                "text/layers/0/mlp/down",
                "text/layers/2/mlp/down",
            ]
        );
        let everything = selector.select(1 << 40);
        assert_eq!(everything.len(), 9);
        assert!(!names(&everything, &objects).contains(&OUTPUT_HEAD.to_owned()));
    }

    #[test]
    fn the_last_family_is_spread_evenly_over_the_layers() {
        // 8 layers of 10 bytes: 30 bytes take 3 of them, at index * 8 / 3 = 0, 2 and 5.
        let layer_names: Vec<String> = (0..8).map(|layer| format!("text/layers/{layer}/mlp/down")).collect();
        let objects: Vec<WeightObject<'_>> = layer_names.iter().map(|name| object(name, 10, true)).collect();
        let selector = Selector::new(&objects);
        assert_eq!(selector.select(30), [0, 2, 5]);
        assert_eq!(selector.select(25), [0, 2, 5]);
        assert_eq!(selector.select(31), [0, 2, 4, 6]);
        assert_eq!(selector.select(10), [0]);
        assert_eq!(selector.select(80), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn what_is_not_a_stageable_projection_is_never_selected() {
        let objects = vec![
            object(EMBEDDING, 100, false),
            object("text/layers/0/mlp/down", 10, false),
            object("text/layers/0/mlp/down.suh", 10, true),
            object("text/layers/0/input_norm", 10, true),
            object("mtp/layer/mlp/down", 10, true),
            object("text/draft_head", 10, true),
            object(OUTPUT_HEAD, 10, true),
        ];
        let selector = Selector::new(&objects);
        assert_eq!(selector.select(1 << 30), [0]);
    }

    /// An artifact of the reference's planner tests: an embedding, a head, and eight layers of a 10 MiB and a 100 MiB
    /// projection, a 50 MiB object of the drafter and a 70 MiB draft head that stay.
    struct Fixture {
        names: Vec<String>,
        bytes: Vec<u64>,
        stageable: Vec<bool>,
    }

    impl Fixture {
        fn small() -> Fixture {
            let mut fixture = Fixture {
                names: Vec::new(),
                bytes: Vec::new(),
                stageable: Vec::new(),
            };
            fixture.push(EMBEDDING, 400 * MIB, false);
            fixture.push(OUTPUT_HEAD, 600 * MIB, false);
            for layer in 0..8 {
                fixture.push(&format!("text/layers/{layer}/gdn/query_key"), 10 * MIB, true);
                fixture.push(&format!("text/layers/{layer}/mlp/down"), 100 * MIB, true);
            }
            fixture.push("mtp/input_projection", 50 * MIB, false);
            fixture.push("text/draft_head", 70 * MIB, false);
            fixture
        }

        fn push(&mut self, name: &str, bytes: u64, stageable: bool) {
            self.names.push(name.to_owned());
            self.bytes.push(bytes);
            self.stageable.push(stageable);
        }

        /// Eight layers of every family and an embedding and a head, each `bytes`.
        fn families(bytes: u64) -> Fixture {
            let mut fixture = Fixture {
                names: Vec::new(),
                bytes: Vec::new(),
                stageable: Vec::new(),
            };
            fixture.push(EMBEDDING, bytes, false);
            fixture.push(OUTPUT_HEAD, bytes, false);
            for layer in 0..8 {
                for family in FAMILIES {
                    fixture.push(&format!("text/layers/{layer}/{family}"), bytes, true);
                }
            }
            fixture
        }

        fn objects(&self) -> Vec<WeightObject<'_>> {
            (0..self.names.len())
                .map(|index| WeightObject {
                    name: &self.names[index],
                    bytes: self.bytes[index],
                    stageable: self.stageable[index],
                    nvfp4: true,
                })
                .collect()
        }

        fn resident(&self) -> u64 {
            self.objects().iter().map(|o| align(o.bytes, ARENA_ALIGNMENT)).sum()
        }
    }

    const SLOTS: usize = 4;

    fn inputs(free: u64) -> OffloadInputs {
        OffloadInputs {
            device_free: free,
            fixed_bytes: 0,
            host_budget: u64::MAX,
            staging_slots: SLOTS,
            window_slots: 2 * SLOTS,
            head_rows: 5,
        }
    }

    fn need(reach: &Reach) -> u64 {
        reach.resident_bytes + reach.offload.staging_bytes()
    }

    #[test]
    fn nothing_leaves_the_device_when_everything_fits() {
        let fixture = Fixture::small();
        let reach = offload(&fixture.objects(), &inputs(fixture.resident()));
        assert!(reach.fits() && reach.offload.is_empty());
        assert_eq!(reach.resident_bytes, fixture.resident());
        assert_eq!(reach.offload, Offload::default());
    }

    #[test]
    fn the_embedding_and_small_projections_leave_first_when_device_memory_is_short() {
        let fixture = Fixture::small();
        let free = fixture.resident() - 100 * MIB;
        let reach = offload(&fixture.objects(), &inputs(free));
        assert!(reach.fits());
        assert!(reach.offload.holds_on_host(EMBEDDING));
        assert!(!reach.offload.holds_on_host(OUTPUT_HEAD));
        assert!(need(&reach) <= free);
        // A partially offloaded plan is a prefix of the full selection, and never takes the head.
        let objects = fixture.objects();
        let everything = Selector::new(&objects).select(1 << 40);
        let everything = names(&everything, &objects);
        assert!(reach.offload.host.iter().all(|name| everything.contains(name)));
    }

    #[test]
    fn the_output_head_is_mapped_only_when_everything_else_offloadable_is_not_enough() {
        let fixture = Fixture::small();
        // The embedding and every projection leave 880 MiB net of the staging ring: 1000 MiB needs the head too.
        let free = fixture.resident() - 1000 * MIB;
        let reach = offload(&fixture.objects(), &inputs(free));
        assert!(reach.fits());
        assert!(reach.offload.holds_on_host(OUTPUT_HEAD));
        assert!(!reach.offload.holds_on_host("text/draft_head"));
        assert!(reach.offload.disk.is_empty());
        assert_eq!(reach.offload.host.last().map(String::as_str), Some(OUTPUT_HEAD));
        // The head is in the host arena and not in the staging ring.
        assert_eq!(reach.offload.host_arena_bytes, (400 + 600 + 8 * 110) * MIB);
        assert_eq!(reach.offload.staging.unwrap().slot_bytes, 100 * MIB);
        assert!(need(&reach) <= free);
        let too_little = offload(&fixture.objects(), &inputs(fixture.resident() - 2000 * MIB));
        assert!(!too_little.fits());
    }

    #[test]
    fn the_output_head_stays_in_host_memory_when_the_window_sends_earlier_objects_to_disk() {
        let fixture = Fixture::small();
        let mut wanted = inputs(fixture.resident() - 1000 * MIB);
        wanted.host_budget = 1100 * MIB;
        let reach = offload(&fixture.objects(), &wanted);
        assert!(reach.offload.holds_on_host(OUTPUT_HEAD));
        assert!(!reach.offload.holds_on_disk(OUTPUT_HEAD));
        assert!(reach.offload.holds_on_host(EMBEDDING));
    }

    #[test]
    fn an_nvfp4_head_is_not_mapped_when_verification_outgrows_the_decode_kernels() {
        let fixture = Fixture::small();
        let free = fixture.resident() - 1000 * MIB;
        let mut within = inputs(free);
        within.head_rows = MAPPED_HEAD_MAX_ROWS;
        assert!(offload(&fixture.objects(), &within).offload.holds_on_host(OUTPUT_HEAD));
        let mut beyond = inputs(free);
        beyond.head_rows = MAPPED_HEAD_MAX_ROWS + 1;
        let reach = offload(&fixture.objects(), &beyond);
        assert!(!reach.offload.holds_on_host(OUTPUT_HEAD));
        assert!(!reach.fits());
        // A head that is not NVFP4 is mapped at any row count.
        let mut objects = fixture.objects();
        objects[1].nvfp4 = false;
        let reach = offload(&objects, &beyond);
        assert!(reach.offload.holds_on_host(OUTPUT_HEAD) && reach.fits());
    }

    #[test]
    fn the_staging_slots_are_a_setting() {
        let fixture = Fixture::small();
        let mut one = inputs(fixture.resident() - 500 * MIB);
        one.staging_slots = 1;
        assert_eq!(offload(&fixture.objects(), &one).offload.staging.unwrap().slots, 1);
        let mut eight = inputs(fixture.resident() - 500 * MIB);
        eight.staging_slots = 8;
        let ring = offload(&fixture.objects(), &eight).offload.staging.unwrap();
        assert_eq!(ring.slots, 8);
        let four = offload(&fixture.objects(), &inputs(fixture.resident() - 500 * MIB))
            .offload
            .staging
            .unwrap();
        assert_eq!((four.slots, four.slot_bytes), (4, ring.slot_bytes));
    }

    #[test]
    fn weights_beyond_the_host_budget_go_to_disk_in_selection_order() {
        let fixture = Fixture::families(16 * MIB);
        let objects = fixture.objects();
        let free = fixture.resident() - 200 * MIB;
        let mut wanted = inputs(free);
        wanted.host_budget = 256 * MIB;
        let reach = offload(&objects, &wanted);
        assert!(reach.fits());
        assert!(!reach.offload.disk.is_empty());
        assert!(reach.offload.holds_on_host(EMBEDDING));
        assert!(reach.host_bytes() <= 256 * MIB);
        let window = reach.offload.window.unwrap();
        assert_eq!(window.slots, 2 * SLOTS);
        assert_eq!(window.bytes(), 8 * (16 * MIB + HOST_ALIGNMENT));
        let host_bytes: u64 = objects
            .iter()
            .filter(|object| reach.offload.holds_on_host(object.name))
            .map(|object| align(object.bytes, HOST_ALIGNMENT))
            .sum();
        assert_eq!(host_bytes, reach.offload.host_arena_bytes);
        let order = Selector::new(&objects).select(u64::MAX >> 1);
        let order = names(&order, &objects);
        let position = |name: &String| order.iter().position(|other| other == name).unwrap();
        let last_host = reach.offload.host.iter().map(position).max().unwrap();
        let first_disk = reach.offload.disk.iter().map(position).min().unwrap();
        assert!(last_host < first_disk);
        assert!(reach.offload.disk.iter().all(|name| !reach.offload.holds_on_host(name)));
        assert!(need(&reach) <= free);
    }

    #[test]
    fn the_device_ring_holds_the_largest_host_or_disk_tensor() {
        let fixture = Fixture::families(16 * MIB);
        let objects = fixture.objects();
        let free = fixture.resident() - 200 * MIB;
        let mut limited = inputs(free);
        limited.host_budget = 256 * MIB;
        let disk = offload(&objects, &limited);
        let host = offload(&objects, &inputs(free));
        // The same device need: every offloaded tensor is staged through the same ring, whichever tier holds it.
        assert_eq!(need(&disk), need(&host));
        let mut offloaded = disk.offload.host.clone();
        offloaded.extend(disk.offload.disk.clone());
        assert_eq!(host.offload.host, offloaded);
    }

    #[test]
    fn a_window_that_does_not_fit_the_host_budget_is_refused() {
        let fixture = Fixture::families(16 * MIB);
        let mut wanted = inputs(fixture.resident() - 200 * MIB);
        wanted.host_budget = 32 * MIB;
        let reach = offload(&fixture.objects(), &wanted);
        assert!(!reach.fits());
        assert!(reach.host_bytes() > 32 * MIB);
    }

    #[test]
    fn an_embedding_beyond_the_host_budget_is_refused() {
        let fixture = Fixture::families(16 * MIB);
        let mut wanted = inputs(fixture.resident() - 200 * MIB);
        wanted.host_budget = 8 * MIB;
        let reach = offload(&fixture.objects(), &wanted);
        assert!(!reach.fits());
        assert!(!reach.offload.holds_on_disk(EMBEDDING));
    }

    #[test]
    fn unlimited_host_memory_needs_no_disk() {
        let fixture = Fixture::families(16 * MIB);
        let reach = offload(&fixture.objects(), &inputs(fixture.resident() - 200 * MIB));
        assert!(reach.fits());
        assert!(!reach.offload.host.is_empty());
        assert!(reach.offload.disk.is_empty());
        assert_eq!(reach.offload.window, None);
    }

    #[test]
    fn a_host_budget_that_holds_the_selection_needs_no_window() {
        let fixture = Fixture::families(16 * MIB);
        let objects = fixture.objects();
        let free = fixture.resident() - 200 * MIB;
        let unlimited = offload(&objects, &inputs(free));
        let mut exact = inputs(free);
        exact.host_budget = unlimited.offload.host_arena_bytes;
        let reach = offload(&objects, &exact);
        assert!(reach.fits());
        assert_eq!(unlimited.offload.host, reach.offload.host);
        assert!(reach.offload.disk.is_empty());
        assert_eq!(reach.offload.window, None);
    }

    #[test]
    fn the_host_arena_is_planned_as_it_is_allocated() {
        // Objects of 5000 bytes take 8192 of the arena, each at a block boundary: planning their payload alone would
        // under-count the pinned bytes the load allocates.
        let mut fixture = Fixture {
            names: Vec::new(),
            bytes: Vec::new(),
            stageable: Vec::new(),
        };
        fixture.push(EMBEDDING, 5000, false);
        fixture.push("text/layers/0/mlp/down", 5000, true);
        let objects = fixture.objects();
        let free = fixture.resident() - 5000;
        let reach = offload(&objects, &inputs(free));
        assert!(reach.offload.holds_on_host(EMBEDDING));
        assert_eq!(reach.offload.host_arena_bytes, 8192 * reach.offload.host.len() as u64);
        let mut tight = inputs(free);
        tight.host_budget = 5000 * reach.offload.host.len() as u64;
        assert!(!offload(&objects, &tight).host_fits);
    }
}

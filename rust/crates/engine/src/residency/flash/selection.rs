//! Which fixed objects leave the device when the minimum expert cache needs the room (`Priority`, `select`).
//!
//! A byte of any fixed weight is read on every token, so each byte moved to the host costs the same bus time: the order
//! follows the device-residency priority, and within a rank the plan moves the fewest and smallest tensors. Routers,
//! norms and every object of at most [`SMALL_BYTES`] never leave: they are tiny and read first in each layer. The
//! output head is the one big matrix read in place from mapped host memory instead of staged, since a staging slot
//! would have to hold all of it.

use std::collections::BTreeMap;

use euhedral_inference_edrl::ComponentGroup;

/// Objects of at most this many bytes are always device resident.
pub const SMALL_BYTES: u64 = 4 << 20;

/// The groups in the order they leave the device: index 0 first.
const OFFLOAD_ORDER: [ComponentGroup; 6] = [
    ComponentGroup::FixedText,
    ComponentGroup::OutputHead,
    ComponentGroup::SharedExpert,
    ComponentGroup::Qsa,
    ComponentGroup::Gdn,
    ComponentGroup::HyperConnection,
];

/// The rank at which an object of `group` and `bytes` may leave the device (0 first), or `None` when it never does.
pub fn offload_rank(group: ComponentGroup, bytes: u64) -> Option<usize> {
    if bytes <= SMALL_BYTES {
        return None;
    }
    OFFLOAD_ORDER.iter().position(|candidate| *candidate == group)
}

/// Whether an object of `group` that leaves the device is read in place (mapped) rather than staged.
pub fn read_in_place(group: ComponentGroup) -> bool {
    group == ComponentGroup::OutputHead
}

/// An object that may leave the device.
#[derive(Clone, Copy, Debug)]
pub struct Movable<'a> {
    pub name: &'a str,
    pub bytes: u64,
    pub rank: usize,
    /// Read in place from pinned host memory once moved, instead of staged.
    pub mapped: bool,
}

/// The objects chosen to leave the device, with the sums the plan needs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// Indices into the movable objects given, in the order they were taken.
    pub taken: Vec<usize>,
    pub staged_bytes: u64,
    /// The largest staged object: what a staging slot must hold.
    pub staged_max: u64,
    pub mapped_bytes: u64,
}

/// `name` with the layer index of its first `/layers/<n>/` replaced by `*`: the family a per-layer object belongs to.
fn family(name: &str) -> String {
    const MARK: &str = "/layers/";
    let Some(at) = name.find(MARK) else {
        return name.to_owned();
    };
    let digits = name[at + MARK.len()..].bytes().take_while(u8::is_ascii_digit).count();
    let after = at + MARK.len() + digits;
    if digits == 0 || name.as_bytes().get(after) != Some(&b'/') {
        return name.to_owned();
    }
    format!("{}*{}", &name[..at + MARK.len()], &name[after..])
}

/// At least `bytes` of `movable` objects (or all of them), in rank order. Within a rank, whole families (the same
/// object in every layer) go smallest tensor first, and the last family takes layers spread evenly, so that transfers
/// interleave with resident work and the staging ring holds the smallest possible tensor.
pub fn select(movable: &[Movable<'_>], bytes: u64) -> Selection {
    let mut selection = Selection::default();
    if bytes == 0 {
        return selection;
    }
    let mut ranks: BTreeMap<usize, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
    for (index, object) in movable.iter().enumerate() {
        ranks
            .entry(object.rank)
            .or_default()
            .entry(family(object.name))
            .or_default()
            .push(index);
    }
    let mut remaining = bytes;
    for families in ranks.into_values() {
        let mut ordered: Vec<(String, Vec<usize>)> = families.into_iter().collect();
        // The map's order is the name's: a stable sort by size leaves equal sizes in name order.
        ordered.sort_by_key(|(_, members)| movable[members[0]].bytes);
        for (_, members) in ordered {
            let family_bytes: u64 = members.iter().map(|index| movable[*index].bytes).sum();
            let take: Vec<usize> = if family_bytes <= remaining {
                members
            } else {
                let size = movable[members[0]].bytes;
                let count = members.len().min(remaining.div_ceil(size) as usize);
                (0..count).map(|slot| members[slot * members.len() / count]).collect()
            };
            for index in take {
                let object = &movable[index];
                selection.taken.push(index);
                remaining = remaining.saturating_sub(object.bytes);
                if object.mapped {
                    selection.mapped_bytes += object.bytes;
                } else {
                    selection.staged_bytes += object.bytes;
                    selection.staged_max = selection.staged_max.max(object.bytes);
                }
            }
            if remaining == 0 {
                return selection;
            }
        }
    }
    selection
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn object(name: &str, mib: u64, rank: usize) -> Movable<'_> {
        Movable {
            name,
            bytes: mib * MIB,
            rank,
            mapped: false,
        }
    }

    fn names<'a>(movable: &[Movable<'a>], selection: &Selection) -> Vec<&'a str> {
        selection.taken.iter().map(|index| movable[*index].name).collect()
    }

    #[test]
    fn small_objects_and_unranked_groups_never_leave_and_the_rest_leave_in_the_documented_order() {
        assert_eq!(offload_rank(ComponentGroup::Gdn, SMALL_BYTES), None);
        assert_eq!(offload_rank(ComponentGroup::Router, 100 * MIB), None);
        assert_eq!(offload_rank(ComponentGroup::TokenEmbedding, 100 * MIB), None);
        assert_eq!(offload_rank(ComponentGroup::Mtp, 100 * MIB), None);
        let ranks: Vec<_> = [
            ComponentGroup::FixedText,
            ComponentGroup::OutputHead,
            ComponentGroup::SharedExpert,
            ComponentGroup::Qsa,
            ComponentGroup::Gdn,
            ComponentGroup::HyperConnection,
        ]
        .iter()
        .map(|group| offload_rank(*group, SMALL_BYTES + 1))
        .collect();
        assert_eq!(ranks, (0..6).map(Some).collect::<Vec<_>>());
        assert!(read_in_place(ComponentGroup::OutputHead));
        assert!(!read_in_place(ComponentGroup::Gdn));
    }

    #[test]
    fn nothing_is_selected_for_zero_bytes() {
        let movable = [object("a", 8, 0)];
        assert_eq!(select(&movable, 0), Selection::default());
    }

    #[test]
    fn a_lower_rank_leaves_first_and_a_smaller_family_before_a_larger_one_in_its_rank() {
        let movable = [
            object("text/layers/0/big", 10, 1),
            object("text/layers/1/big", 10, 1),
            object("text/layers/0/small", 6, 1),
            object("text/layers/1/small", 6, 1),
            object("text/head", 20, 0),
        ];
        // Rank 0 first (20 MiB), then the 6 MiB family whole (12 MiB): 32 MiB asked, 32 taken.
        let selection = select(&movable, 32 * MIB);
        assert_eq!(
            names(&movable, &selection),
            ["text/head", "text/layers/0/small", "text/layers/1/small"]
        );
        assert_eq!(selection.staged_bytes, 32 * MIB);
        assert_eq!(selection.staged_max, 20 * MIB);
    }

    #[test]
    fn the_last_family_gives_layers_spread_evenly_not_the_first_ones() {
        let owned: Vec<String> = (0..8).map(|layer| format!("text/layers/{layer}/proj")).collect();
        let movable: Vec<Movable<'_>> = owned.iter().map(|name| object(name, 5, 0)).collect();
        // 12 MiB asked of 5 MiB layers: three of eight, at layers 0, 2 and 5.
        let selection = select(&movable, 12 * MIB);
        assert_eq!(selection.taken, [0, 2, 5]);
    }

    #[test]
    fn objects_read_in_place_count_as_mapped_and_need_no_staging_slot() {
        let movable = [Movable {
            name: "text/output_head",
            bytes: 30 * MIB,
            rank: 1,
            mapped: true,
        }];
        let selection = select(&movable, MIB);
        assert_eq!(
            (selection.mapped_bytes, selection.staged_bytes, selection.staged_max),
            (30 * MIB, 0, 0)
        );
    }

    #[test]
    fn asking_for_more_than_there_is_takes_everything() {
        let movable = [object("a", 5, 0), object("b", 7, 3)];
        let selection = select(&movable, 1 << 40);
        assert_eq!(selection.taken.len(), 2);
        assert_eq!(selection.staged_bytes, 12 * MIB);
    }

    #[test]
    fn only_the_layer_segment_of_a_name_is_the_family() {
        assert_eq!(family("text/layers/12/gdn/qkv"), "text/layers/*/gdn/qkv");
        assert_eq!(family("text/layers/x/gdn"), "text/layers/x/gdn");
        assert_eq!(family("text/output_head"), "text/output_head");
    }
}

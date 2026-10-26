//! The ordering edges that a single stream used to provide implicitly (`ExecutionPlan.withStorageHazards`).
//!
//! Stages run on different lanes, so a lane's order no longer puts a write after the reads and writes of the same
//! storage before it. For every storage, each reader is ordered after the last writer, and each writer after the last
//! writer and every reader since it. The expansion scratch is one more storage per region that each of its stages
//! writes, so the stages of one region take it one after another. An edge that the dependencies already imply
//! transitively is not added.

use super::buffers::{Access, scratch_storage, slot_storage, storage_count};

/// A set of node indices.
#[derive(Clone, Debug)]
struct Bits {
    words: Vec<u64>,
}

impl Bits {
    fn new(len: usize) -> Bits {
        Bits {
            words: vec![0; len.div_ceil(64)],
        }
    }

    fn set(&mut self, index: usize) {
        self.words[index / 64] |= 1 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        self.words[index / 64] >> (index % 64) & 1 == 1
    }

    fn or(&mut self, other: &Bits) {
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word |= other;
        }
    }
}

/// The edges of each node: its dependencies, then the producers the storage hazards add (latest first).
///
/// `deps[node]` lists nodes before `node`; `access[node]` what it touches. `regions` is the number of scratch
/// regions.
pub(crate) fn with_hazards(access: &[Access], deps: &[Vec<usize>], regions: usize) -> Vec<Vec<usize>> {
    let count = access.len();
    let storages = storage_count(regions);
    let mut last_writer: Vec<Option<usize>> = vec![None; storages];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); storages];
    let mut ancestors: Vec<Bits> = Vec::with_capacity(count);
    let mut ordered: Vec<Vec<usize>> = Vec::with_capacity(count);
    for (node, touched) in access.iter().enumerate() {
        let mut edges: Vec<usize> = Vec::new();
        let mut reach = Bits::new(count);
        for &dependency in &deps[node] {
            if !edges.contains(&dependency) {
                edges.push(dependency);
            }
            reach.or(&ancestors[dependency]);
            reach.set(dependency);
        }
        let mut required: Vec<usize> = Vec::new();
        for storage in touched.reads.iter().copied().filter_map(slot_storage) {
            if let Some(writer) = last_writer[storage] {
                required.push(writer);
            }
        }
        for storage in touched.writes.iter().copied().filter_map(slot_storage) {
            if let Some(writer) = last_writer[storage] {
                required.push(writer);
            }
            required.extend_from_slice(&readers[storage]);
        }
        if let Some(region) = touched.scratch
            && let Some(writer) = last_writer[scratch_storage(region)]
        {
            required.push(writer);
        }
        // Latest first: an earlier requirement is often already an ancestor of a later one.
        required.sort_unstable();
        required.dedup();
        for &producer in required.iter().rev() {
            if producer == node || reach.get(producer) {
                continue;
            }
            edges.push(producer);
            reach.or(&ancestors[producer]);
            reach.set(producer);
        }
        ancestors.push(reach);
        ordered.push(edges);
        for storage in touched.reads.iter().copied().filter_map(slot_storage) {
            readers[storage].push(node);
        }
        for storage in touched.writes.iter().copied().filter_map(slot_storage) {
            last_writer[storage] = Some(node);
            readers[storage].clear();
        }
        if let Some(region) = touched.scratch {
            last_writer[scratch_storage(region)] = Some(node);
        }
    }
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::BufferSlot::{self, FfnDelta, HiddenState, InputNormalized, MixerDelta, TokenIds};

    fn access(reads: &[BufferSlot], writes: &[BufferSlot]) -> Access {
        Access {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
            scratch: None,
        }
    }

    #[test]
    fn a_reader_follows_the_last_writer_and_a_writer_follows_the_readers_since() {
        // 0 writes the hidden state, 1 and 2 read it (unordered), 3 writes it again.
        let nodes = [
            access(&[], &[HiddenState]),
            access(&[HiddenState], &[MixerDelta]),
            access(&[HiddenState], &[FfnDelta]),
            access(&[], &[HiddenState]),
        ];
        let deps = vec![vec![], vec![], vec![], vec![]];
        let edges = with_hazards(&nodes, &deps, 1);
        assert_eq!(edges[0], Vec::<usize>::new());
        assert_eq!(edges[1], vec![0], "a reader after the writer");
        assert_eq!(edges[2], vec![0]);
        assert_eq!(
            edges[3],
            vec![2, 1],
            "a writer after its readers, latest first; the first writer is reached"
        );
    }

    #[test]
    fn an_edge_the_dependencies_imply_is_not_added() {
        // 1 depends on 0 and reads what 0 wrote: the dependency is the edge, nothing is added twice.
        let nodes = [access(&[], &[HiddenState]), access(&[HiddenState], &[InputNormalized])];
        let edges = with_hazards(&nodes, &[vec![], vec![0]], 1);
        assert_eq!(edges[1], vec![0]);
        // 2 writes the hidden state: it must follow 1 (a reader) and 0 (the writer), but 0 is an ancestor of 1.
        let nodes = [
            access(&[], &[HiddenState]),
            access(&[HiddenState], &[InputNormalized]),
            access(&[], &[HiddenState]),
        ];
        let edges = with_hazards(&nodes, &[vec![], vec![0], vec![]], 1);
        assert_eq!(edges[2], vec![1], "the reader implies the writer before it");
    }

    #[test]
    fn the_stages_of_one_scratch_region_take_it_one_after_another() {
        let mut nodes = vec![access(&[], &[]); 4];
        nodes[0].scratch = Some(0);
        nodes[1].scratch = Some(1);
        nodes[2].scratch = Some(0);
        nodes[3].scratch = Some(1);
        let edges = with_hazards(&nodes, &vec![vec![]; 4], 2);
        assert_eq!(edges[2], vec![0], "region 0 in turn");
        assert_eq!(edges[3], vec![1], "region 1 in turn");
        assert_eq!(edges[1], Vec::<usize>::new(), "the regions are independent");
    }

    #[test]
    fn the_input_record_orders_nothing() {
        let nodes = [access(&[], &[TokenIds]), access(&[TokenIds], &[])];
        let edges = with_hazards(&nodes, &[vec![], vec![]], 1);
        assert_eq!(edges[1], Vec::<usize>::new());
    }
}

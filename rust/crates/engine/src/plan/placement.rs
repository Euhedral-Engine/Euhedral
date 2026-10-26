//! Lane placement: how many lanes the plan's graphs need, and where the path policy puts each stage.
//!
//! A graph of a chain needs one lane; a fan-out needs as many as the stages that can run at once. So the lane count is
//! the width of the widest graph: the most stages no edge orders, which is the size of a largest antichain (Dilworth:
//! the stage count less a largest matching of the ordering). RustEE's `StageGraph` places the stages by the path
//! policy (`StageGraph.markPaths`): it keeps a graph's critical chain on one lane and gives the other branches of a
//! fan-out lanes of their own; [`path_predecessors`] is the same rule, which the plan reads to know where a chain of a
//! lane ends.

use std::collections::VecDeque;

/// A set of node indices as words.
fn words(len: usize) -> usize {
    len.div_ceil(64)
}

/// The most stages of a graph that no chain of `edges` orders, where `edges[node]` lists the nodes before `node`
/// (every one a smaller index).
pub(crate) fn fan_out_width(edges: &[Vec<usize>]) -> usize {
    let count = edges.len();
    if count == 0 {
        return 0;
    }
    // below[node]: the nodes before `node` by any chain.
    let mut below: Vec<Vec<u64>> = Vec::with_capacity(count);
    for (node, predecessors) in edges.iter().enumerate() {
        let mut reach = vec![0u64; words(count)];
        for &predecessor in predecessors {
            debug_assert!(predecessor < node, "edges point to earlier nodes");
            for (word, other) in reach.iter_mut().zip(&below[predecessor]) {
                *word |= other;
            }
            reach[predecessor / 64] |= 1 << (predecessor % 64);
        }
        below.push(reach);
    }
    // The ordering as a bipartite graph: a node on the left matches a node on the right that comes after it.
    let mut after: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (node, reach) in below.iter().enumerate() {
        for (word_index, &word) in reach.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                after[word_index * 64 + bit].push(node as u32);
            }
        }
    }
    count - maximum_matching(&after, count)
}

/// Hopcroft-Karp over `adjacency` (left node to right nodes), both sides of `count` nodes.
fn maximum_matching(adjacency: &[Vec<u32>], count: usize) -> usize {
    const FREE: u32 = u32::MAX;
    let mut left_match = vec![FREE; count];
    let mut right_match = vec![FREE; count];
    let mut distance = vec![0u32; count];
    let mut matched = 0;
    loop {
        // Breadth-first layers from the free left nodes.
        let mut queue: Vec<u32> = Vec::new();
        for left in 0..count {
            if left_match[left] == FREE {
                distance[left] = 0;
                queue.push(left as u32);
            } else {
                distance[left] = FREE;
            }
        }
        let mut found = false;
        let mut head = 0;
        while head < queue.len() {
            let left = queue[head] as usize;
            head += 1;
            for &right in &adjacency[left] {
                let next = right_match[right as usize];
                if next == FREE {
                    found = true;
                } else if distance[next as usize] == FREE {
                    distance[next as usize] = distance[left] + 1;
                    queue.push(next);
                }
            }
        }
        if !found {
            return matched;
        }
        // Depth-first augmenting paths along the layers, with an explicit stack.
        let mut cursor = vec![0usize; count];
        for start in 0..count {
            if left_match[start] != FREE {
                continue;
            }
            let mut path: Vec<u32> = vec![start as u32];
            while let Some(&left) = path.last() {
                let left = left as usize;
                if cursor[left] == adjacency[left].len() {
                    distance[left] = FREE;
                    path.pop();
                    continue;
                }
                let right = adjacency[left][cursor[left]] as usize;
                cursor[left] += 1;
                let next = right_match[right];
                if next == FREE {
                    // Augment along the path: each left node takes the right node it was exploring.
                    let mut take = right;
                    while let Some(left) = path.pop() {
                        let left = left as usize;
                        let previous = left_match[left];
                        left_match[left] = take as u32;
                        right_match[take] = left as u32;
                        take = previous as usize;
                    }
                    matched += 1;
                    break;
                } else if distance[next as usize] == distance[left] + 1 {
                    path.push(next);
                }
            }
        }
    }
}

/// For each stage, the predecessor whose lane it continues under the path policy (`StageGraph.markPaths`): every
/// stage links to the one successor that continues its longest path to a sink (the first listed on ties), so a
/// graph's critical chain keeps one lane and the other successors of a fan-out are left to take lanes of their own.
/// `edges[node]` lists the nodes before `node`.
pub(crate) fn path_predecessors(edges: &[Vec<usize>]) -> Vec<Option<usize>> {
    let count = edges.len();
    // Successors and predecessors in the order the graph lists them: by ascending index.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (node, predecessors) in edges.iter().enumerate() {
        for &predecessor in predecessors {
            successors[predecessor].push(node);
        }
    }
    let mut predecessors_of: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (producer, listed) in successors.iter().enumerate() {
        for &successor in listed {
            predecessors_of[successor].push(producer);
        }
    }
    let mut height = vec![0usize; count];
    let mut pending: Vec<usize> = successors.iter().map(Vec::len).collect();
    let mut ready: VecDeque<usize> = (0..count).filter(|&node| pending[node] == 0).collect();
    let mut path = vec![None; count];
    while let Some(stage) = ready.pop_front() {
        let mut next: Option<usize> = None;
        for &successor in &successors[stage] {
            if next.is_none_or(|current| height[successor] > height[current]) {
                next = Some(successor);
            }
        }
        height[stage] = next.map_or(1, |next| height[next] + 1);
        if let Some(next) = next {
            path[next] = Some(stage);
        }
        for &predecessor in &predecessors_of[stage] {
            pending[predecessor] -= 1;
            if pending[predecessor] == 0 {
                ready.push_back(predecessor);
            }
        }
    }
    path
}

/// Which stages end a chain of lane continuations: those no stage continues the lane of (`path[s]` names no later
/// stage's predecessor). RustEE's graph puts a stage on the lane of its path predecessor and every other branch on a
/// lane of its own, so a lane's last marker belongs to the end of a chain; those stages signal. `edges[node]` lists
/// the nodes before `node`.
pub(crate) fn chain_ends(edges: &[Vec<usize>]) -> Vec<bool> {
    let mut continued = vec![false; edges.len()];
    for predecessor in path_predecessors(edges).into_iter().flatten() {
        continued[predecessor] = true;
    }
    continued.into_iter().map(|continued| !continued).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_is_one_wide_and_a_fan_out_is_as_wide_as_its_branches() {
        assert_eq!(fan_out_width(&[vec![], vec![0], vec![1], vec![2]]), 1);
        // 0 feeds 1, 2 and 3, which 4 joins.
        assert_eq!(fan_out_width(&[vec![], vec![0], vec![0], vec![0], vec![1, 2, 3]]), 3);
        // Two independent chains of two.
        assert_eq!(fan_out_width(&[vec![], vec![0], vec![], vec![2]]), 2);
        assert_eq!(fan_out_width(&[]), 0);
    }

    #[test]
    fn width_counts_stages_that_a_longer_path_leaves_unordered() {
        // 0 -> 1 -> 2 -> 4 and 0 -> 3 -> 4: 3 runs beside 1 and 2.
        assert_eq!(fan_out_width(&[vec![], vec![0], vec![1], vec![0], vec![2, 3]]), 2);
        // A diamond of diamonds stays as wide as one diamond.
        let edges = [vec![], vec![0], vec![0], vec![1, 2], vec![3], vec![3], vec![4, 5]];
        assert_eq!(fan_out_width(&edges), 2);
    }

    #[test]
    fn the_width_equals_a_brute_force_antichain_on_random_graphs() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..60 {
            let count = 2 + (next() % 11) as usize;
            let edges: Vec<Vec<usize>> = (0..count)
                .map(|node| (0..node).filter(|_| next() % 4 == 0).collect())
                .collect();
            // Reachability by repeated closure.
            let mut related = vec![vec![false; count]; count];
            for (node, predecessors) in edges.iter().enumerate() {
                for &predecessor in predecessors {
                    related[predecessor][node] = true;
                }
            }
            for through in 0..count {
                for from in 0..count {
                    for to in 0..count {
                        if related[from][through] && related[through][to] {
                            related[from][to] = true;
                        }
                    }
                }
            }
            let mut best = 0;
            for subset in 0u32..1 << count {
                let members: Vec<usize> = (0..count).filter(|index| subset >> index & 1 == 1).collect();
                let independent = members
                    .iter()
                    .all(|&a| members.iter().all(|&b| a == b || !related[a][b]));
                if independent {
                    best = best.max(members.len());
                }
            }
            assert_eq!(fan_out_width(&edges), best, "{edges:?}");
        }
    }

    #[test]
    fn the_critical_successor_continues_the_lane_and_the_other_branches_are_left_to_take_lanes() {
        // 0 feeds 1, 2 and 3; 5 joins 4, 2 and 3. 1 -> 4 -> 5 is the longest path from 0.
        let edges = [vec![], vec![0], vec![0], vec![0], vec![1], vec![4, 2, 3]];
        let path = path_predecessors(&edges);
        assert_eq!(
            path[1],
            Some(0),
            "the successor with the longest path continues the root's lane"
        );
        assert_eq!(path[2], None);
        assert_eq!(path[3], None);
        assert_eq!(path[4], Some(1));
        assert_eq!(path[5], Some(4));
    }

    #[test]
    fn of_equal_heights_the_first_listed_and_into_a_join_the_last_visited() {
        // 0 feeds 1 and 2 (equal heights): the first listed, 1, continues. 3 joins 1 and 2: both name it critical, and the
        // one visited last continues it.
        let edges = [vec![], vec![0], vec![0], vec![1, 2]];
        let path = path_predecessors(&edges);
        assert_eq!(path[1], Some(0));
        assert_eq!(path[2], None);
        assert_eq!(
            path[3],
            Some(2),
            "the join continues the last visited of the predecessors that name it"
        );
    }

    #[test]
    fn a_chain_ends_where_no_stage_continues_its_lane() {
        let edges = [vec![], vec![0], vec![0], vec![0], vec![1], vec![4, 2, 3]];
        // 2 and 3 are branches that end at the join; 5 ends the main chain.
        assert_eq!(chain_ends(&edges), vec![false, false, true, true, false, true]);
        // The tie: 2 continues the join, so 1 ends its chain.
        let tie = [vec![], vec![0], vec![0], vec![1, 2]];
        assert_eq!(chain_ends(&tie), vec![false, true, false, true]);
    }
}

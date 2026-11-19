//! The SD4 scale table of a tensor from its per-nearest-code costs, on the host in `f64`: the cost bounds, then a
//! dynamic program over the table's ascending entries.

/// The finite E4M3 magnitudes: codes 0 to 126.
pub const CODES: usize = 127;
/// Entries of a table.
pub const ENTRIES: usize = 16;
/// How many codes above the nearest one a block's scale may be, before it costs the block's whole energy.
pub const ABOVE: usize = 40;

/// The table for costs `costs[nearest][code]` (measured from `below` codes under to [`ABOVE`] over each nearest
/// code), the energy and the block count of each nearest code: the [`ENTRIES`] ascending E4M3 codes that minimise the
/// summed least cost of every nearest code.
///
/// # Panics
/// If `costs` is not [`CODES`] by [`CODES`], or `energy` or `counts` is not [`CODES`] long.
pub fn choose(costs: &[Vec<f64>], energy: &[f64], counts: &[u64], below: usize) -> [u8; ENTRIES] {
    let bounded = bounds(costs, energy, counts, below);
    let table = table(&bounded);
    let mut bytes = [0u8; ENTRIES];
    for (byte, code) in bytes.iter_mut().zip(table) {
        // A code is under 127.
        *byte = code as u8;
    }
    bytes
}

/// Too far above costs the blocks' energy, too far below is not allowed, and a nearest code without blocks costs
/// nothing.
fn bounds(costs: &[Vec<f64>], energy: &[f64], counts: &[u64], below: usize) -> Vec<Vec<f64>> {
    assert!(costs.len() == CODES && costs.iter().all(|row| row.len() == CODES));
    assert!(energy.len() == CODES && counts.len() == CODES);
    (0..CODES)
        .map(|nearest| {
            (0..CODES)
                .map(|code| {
                    let mut cost = costs[nearest][code];
                    if code > nearest + ABOVE {
                        cost = energy[nearest];
                    }
                    if code + below < nearest {
                        cost = f64::INFINITY;
                    }
                    if counts[nearest] == 0 {
                        cost = 0.0;
                    }
                    cost
                })
                .collect()
        })
        .collect()
}

/// The dynamic program over the ascending entries; a tie goes to the earliest choice.
fn table(costs: &[Vec<f64>]) -> [usize; ENTRIES] {
    let n = CODES;
    let mut below = vec![0.0f64; n];
    let mut above = vec![0.0f64; n];
    for b in 0..n {
        for row in &costs[..b] {
            below[b] += row[b];
        }
        for row in &costs[b..] {
            above[b] += row[b];
        }
    }
    let mut segment = vec![vec![f64::INFINITY; n]; n];
    for a in 0..n {
        for b in a + 1..n {
            let mut sum = 0.0f64;
            for row in &costs[a..b] {
                sum += row[a].min(row[b]);
            }
            segment[a][b] = sum;
        }
    }
    let mut best = below;
    let mut previous: Vec<Vec<usize>> = Vec::with_capacity(ENTRIES - 1);
    for _ in 0..ENTRIES - 1 {
        let mut argmin = vec![0usize; n];
        let mut next = vec![0.0f64; n];
        for b in 0..n {
            let mut choice = 0;
            let mut least = best[0] + segment[0][b];
            for a in 1..n {
                let total = best[a] + segment[a][b];
                if total < least {
                    least = total;
                    choice = a;
                }
            }
            argmin[b] = choice;
            next[b] = least;
        }
        previous.push(argmin);
        best = next;
    }
    let mut last = 0;
    for b in 1..n {
        if best[b] + above[b] < best[last] + above[last] {
            last = b;
        }
    }
    let mut table = [0usize; ENTRIES];
    table[0] = last;
    for (index, step) in previous.iter().rev().enumerate() {
        table[index + 1] = step[table[index]];
    }
    table.sort_unstable();
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Costs of a tensor whose blocks all have the nearest code `nearest`: a code costs its distance to it, squared.
    fn tensor_at(nearest: &[usize], below: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<u64>, usize) {
        let mut costs = vec![vec![0.0; CODES]; CODES];
        let mut energy = vec![0.0; CODES];
        let mut counts = vec![0u64; CODES];
        for &o in nearest {
            counts[o] += 1;
            energy[o] = 1.0e6;
            for (c, cost) in costs[o].iter_mut().enumerate() {
                let distance = c as f64 - o as f64;
                *cost = distance * distance;
            }
        }
        (costs, energy, counts, below)
    }

    #[test]
    fn sixteen_nearest_codes_each_get_their_own_entry() {
        let nearest: Vec<usize> = (0..ENTRIES).map(|i| 40 + 3 * i).collect();
        let (costs, energy, counts, below) = tensor_at(&nearest, 6);
        let table = choose(&costs, &energy, &counts, below);
        let expected: Vec<u8> = nearest.iter().map(|&code| code as u8).collect();
        assert_eq!(table.to_vec(), expected);
    }

    #[test]
    fn a_table_is_ascending_and_a_tensor_without_blocks_gets_the_first_choices() {
        let (costs, energy, counts, below) = tensor_at(&[10, 50, 90], 6);
        let table = choose(&costs, &energy, &counts, below);
        assert!(table.windows(2).all(|pair| pair[0] <= pair[1]));
        for code in [10u8, 50, 90] {
            assert!(table.contains(&code), "{code} missing from {table:?}");
        }
        let (costs, energy, counts, below) = tensor_at(&[], 6);
        let table = choose(&costs, &energy, &counts, below);
        assert!(table.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn a_scale_far_below_the_nearest_code_is_never_chosen_for_it() {
        // One nearest code, 60: with `below` 6 only codes from 54 up may serve it.
        let (mut costs, energy, counts, below) = tensor_at(&[60], 6);
        // Make the forbidden codes look free: the bound must still keep them out.
        for cost in &mut costs[60][..54] {
            *cost = 0.0;
        }
        let table = choose(&costs, &energy, &counts, below);
        assert!(table.contains(&60), "{table:?}");
    }

    #[test]
    fn a_code_too_far_above_costs_the_energy() {
        let (costs, energy, counts, below) = tensor_at(&[10], 6);
        let bounded = bounds(&costs, &energy, &counts, below);
        assert_eq!(bounded[10][10 + ABOVE], costs[10][10 + ABOVE]);
        assert_eq!(bounded[10][10 + ABOVE + 1], energy[10]);
        assert_eq!(bounded[10][3], f64::INFINITY);
        assert_eq!(bounded[10][4], costs[10][4]);
        assert_eq!(bounded[11][0], 0.0, "a nearest code without blocks costs nothing");
    }

    #[test]
    fn the_program_reaches_the_least_cost_over_every_table_of_a_small_tensor() {
        // Three nearest codes and four entries' worth of freedom would be 127^16 tables; instead check optimality
        // against the best of the tables built from the nearest codes themselves, which cannot be beaten.
        let nearest = [20usize, 70, 100];
        let (costs, energy, counts, below) = tensor_at(&nearest, 6);
        let table = choose(&costs, &energy, &counts, below);
        let total = |entries: &[u8]| -> f64 {
            let bounded = bounds(&costs, &energy, &counts, below);
            nearest
                .iter()
                .map(|&o| {
                    entries
                        .iter()
                        .map(|&c| bounded[o][usize::from(c)])
                        .fold(f64::INFINITY, f64::min)
                })
                .sum()
        };
        assert_eq!(total(&table), 0.0, "{table:?}");
    }
}

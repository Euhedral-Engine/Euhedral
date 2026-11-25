//! Bit allocation over the budgeted linears, as exllamav3 1.6.1's `create_q_strategy` does it without `--hq`, with
//! half-integer steps (the `mul1` codebook): every linear starts at the largest supported rate not above the target,
//! then groups of linears are promoted one step at a time while the parameter-weighted budget allows.
//!
//! Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).

use thiserror::Error;

/// One budgeted linear. Linears with the same `group` are promoted together; `priority` is the largest `q_priority`
/// among its enclosing modules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub name: String,
    pub layer: usize,
    /// The number of weights.
    pub numel: u64,
    pub priority: i32,
    pub group: String,
}

/// Why no allocation can be made.
#[derive(Debug, Error, PartialEq)]
pub enum AllocationError {
    /// The target is not a number, or is not positive.
    #[error("the target of {0} bits per weight is not a positive number")]
    BadTarget(f64),
    /// A linear is on a layer the stack does not have.
    #[error("{name} is on layer {layer} of a {layers}-layer stack")]
    BadLayer { name: String, layer: usize, layers: usize },
}

/// The largest supported rate not above `bpw`: halves below 4, integers up to 8, at least 1.
fn rate_floor(bpw: f64) -> f64 {
    let rate = if bpw < 4.0 {
        (bpw * 2.0).floor() / 2.0
    } else {
        bpw.floor()
    };
    rate.clamp(1.0, 8.0)
}

/// The next supported rate above `bpw` (8 stays 8).
fn rate_next(bpw: f64) -> f64 {
    (bpw + if bpw < 4.0 { 0.5 } else { 1.0 }).min(8.0)
}

/// A linear and the rate it is at now.
struct Target {
    numel: f64,
    bits: f64,
}

impl Target {
    fn delta(&self) -> f64 {
        (rate_next(self.bits) - self.bits) * self.numel
    }
}

/// The bits per weight of every linear, in the order of `linears`, for the average `bpw`. Groups are promoted in
/// order of priority (highest first), distance to the nearer end of the `layers`-layer stack, layer and first
/// appearance in `linears`; passes repeat until no group fits the budget `trunc(bpw * total weights)`.
pub fn allocate(linears: &[Linear], layers: usize, bpw: f64) -> Result<Vec<(String, f64)>, AllocationError> {
    if !bpw.is_finite() || bpw <= 0.0 {
        return Err(AllocationError::BadTarget(bpw));
    }
    if let Some(bad) = linears.iter().find(|linear| linear.layer >= layers) {
        return Err(AllocationError::BadLayer {
            name: bad.name.clone(),
            layer: bad.layer,
            layers,
        });
    }
    let base = rate_floor(bpw);
    let mut targets: Vec<Target> = Vec::with_capacity(linears.len());
    // The groups, each the indices of its linears, in the order the groups first appear.
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut total_numel = 0.0;
    let mut sum_bits = 0.0;
    for (index, linear) in linears.iter().enumerate() {
        let numel = linear.numel as f64;
        targets.push(Target { numel, bits: base });
        total_numel += numel;
        sum_bits += numel * base;
        match groups.iter_mut().find(|(group, _)| *group == linear.group) {
            Some((_, members)) => members.push(index),
            None => groups.push((&linear.group, vec![index])),
        }
    }
    let max_bits = (bpw * total_numel).trunc();

    // A stable sort, so equal keys keep the groups' first-appearance order.
    groups.sort_by_key(|(_, members)| {
        let priority = members
            .iter()
            .map(|&member| linears[member].priority)
            .max()
            .unwrap_or(0);
        let first = linears[members[0]].layer;
        (-i64::from(priority), first.min(layers - 1 - first), first)
    });

    while sum_bits < max_bits {
        let mut updates = false;
        for (_, members) in &groups {
            let cost: f64 = members.iter().map(|&member| targets[member].delta()).sum();
            if cost > 0.0 && sum_bits + cost <= max_bits {
                for &member in members {
                    targets[member].bits = rate_next(targets[member].bits);
                }
                updates = true;
                sum_bits += cost;
            }
        }
        if !updates {
            break;
        }
    }
    Ok(linears
        .iter()
        .zip(&targets)
        .map(|(linear, target)| (linear.name.clone(), target.bits))
        .collect())
}

#[cfg(test)]
mod tests;

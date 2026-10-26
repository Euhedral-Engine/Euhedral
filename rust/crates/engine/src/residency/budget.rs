//! The host memory a load may use (`HostBudget`).
//!
//! The budget is the memory the machine can spare now: `MemAvailable`, or the headroom of the process's cgroup when
//! that is less, less room for the operating system (10%, never closer than 4 GiB to exhaustion) and for the runtime.
//! `memory.host-limit` states the budget instead, in bytes, and is respected as stated: it may be smaller than the
//! automatic one (0 means no RAM tier) or larger. Reading the files these numbers come from is the startup's work
//! (`startup`): this module parses their text and does the arithmetic.

/// A budget never closer to exhaustion than this (`HostBudget.MARGIN_FLOOR`): the room the operating system needs.
const MARGIN_FLOOR: u64 = 4 << 30;

/// Where the budget came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// Derived from the memory the machine can spare.
    Automatic,
    /// Stated by the user (`memory.host-limit`).
    Explicit,
}

/// The host memory a load may use: pinnable (page-locked) and resident (all uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    pub pinnable_bytes: u64,
    pub resident_bytes: u64,
    pub source: BudgetSource,
}

impl HostBudget {
    /// A stated budget.
    pub fn stated(bytes: u64) -> Self {
        Self {
            pinnable_bytes: bytes,
            resident_bytes: bytes,
            source: BudgetSource::Explicit,
        }
    }

    /// A budget from the memory the machine can spare: 90% of it, but never closer than 4 GiB to exhaustion
    /// (`HostBudget.ofAvailable`).
    pub fn of_available(available_bytes: u64) -> Self {
        let margin = MARGIN_FLOOR.max(available_bytes / 10);
        let usable = available_bytes.saturating_sub(margin);
        Self {
            pinnable_bytes: usable,
            resident_bytes: usable,
            source: BudgetSource::Automatic,
        }
    }

    /// The budget of a machine that can spare `available` bytes now (`None`: unknown) and a runtime that keeps
    /// `runtime_reserve` bytes for itself, or `stated` bytes (`HostBudget.of`).
    pub fn of(stated: Option<u64>, available: Option<u64>, runtime_reserve: u64) -> Self {
        let machine = available.map_or(u64::MAX / 4, |available| {
            Self::of_available(available)
                .resident_bytes
                .saturating_sub(runtime_reserve)
        });
        match stated {
            None => Self {
                pinnable_bytes: machine,
                resident_bytes: machine,
                source: BudgetSource::Automatic,
            },
            Some(bytes) => Self {
                pinnable_bytes: machine,
                resident_bytes: bytes,
                source: BudgetSource::Explicit,
            },
        }
    }

    /// The pinned memory this budget allows: as stated when stated, else what the machine can spare
    /// (`HostBudget.pinnedBytes`).
    pub fn pinned_bytes(&self) -> u64 {
        match self.source {
            BudgetSource::Explicit => self.resident_bytes,
            BudgetSource::Automatic => self.pinnable_bytes.min(self.resident_bytes),
        }
    }
}

/// `MemAvailable` in bytes from the text of `/proc/meminfo`.
pub fn mem_available(meminfo: &str) -> Option<u64> {
    meminfo_kib(meminfo, "MemAvailable:")
}

/// `MemTotal` in bytes from the text of `/proc/meminfo`.
pub fn mem_total(meminfo: &str) -> Option<u64> {
    meminfo_kib(meminfo, "MemTotal:")
}

fn meminfo_kib(meminfo: &str, key: &str) -> Option<u64> {
    let line = meminfo.lines().find(|line| line.starts_with(key))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// The cgroup (v2) path in the text of `/proc/self/cgroup`: the `0::` line.
pub fn cgroup_path(proc_self_cgroup: &str) -> Option<&str> {
    proc_self_cgroup
        .lines()
        .filter_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
        .next_back()
}

/// The headroom of one cgroup: `memory.max` less `memory.current`, or `None` when it has no limit
/// (`HostBudget.headroom`).
pub fn cgroup_headroom(memory_max: &str, memory_current: &str) -> Option<u64> {
    let max = memory_max.trim();
    if max.is_empty() || max == "max" {
        return None;
    }
    let limit: u64 = max.parse().ok()?;
    let current: u64 = memory_current.trim().parse().ok()?;
    Some(limit.saturating_sub(current))
}

/// The memory the machine and the process's cgroups can spare now: the lesser of `MemAvailable` and the least cgroup
/// headroom (`HostBudget.availableBytes`). `None` when neither is known.
pub fn available_bytes(mem_available: Option<u64>, least_cgroup_headroom: Option<u64>) -> Option<u64> {
    match (mem_available, least_cgroup_headroom) {
        (Some(available), Some(headroom)) => Some(available.min(headroom)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn available_memory_keeps_ten_percent_and_never_less_than_four_gib() {
        // 100 GiB available: the 10% margin (10 GiB) is above the floor.
        assert_eq!(HostBudget::of_available(100 * GIB).resident_bytes, 90 * GIB);
        // 20 GiB: 10% is 2 GiB, the 4 GiB floor applies.
        assert_eq!(HostBudget::of_available(20 * GIB).resident_bytes, 16 * GIB);
        // Under the floor nothing is left.
        assert_eq!(HostBudget::of_available(3 * GIB).resident_bytes, 0);
    }

    #[test]
    fn a_stated_budget_is_used_as_stated_and_an_automatic_one_leaves_the_runtime_its_reserve() {
        let automatic = HostBudget::of(None, Some(100 * GIB), GIB);
        assert_eq!(automatic.source, BudgetSource::Automatic);
        assert_eq!(automatic.pinned_bytes(), 89 * GIB);

        let stated = HostBudget::of(Some(200 * GIB), Some(100 * GIB), GIB);
        assert_eq!(stated.source, BudgetSource::Explicit);
        assert_eq!(
            stated.pinned_bytes(),
            200 * GIB,
            "larger than the machine can spare, used as stated"
        );

        assert_eq!(HostBudget::of(Some(0), Some(100 * GIB), 0).pinned_bytes(), 0);
        assert!(
            HostBudget::of(None, None, 0).pinned_bytes() > 1 << 50,
            "unknown memory is not a limit"
        );
    }

    #[test]
    fn proc_files_parse() {
        let meminfo = "MemTotal:       65000000 kB\nMemFree:  1 kB\nMemAvailable:   52428800 kB\n";
        assert_eq!(mem_available(meminfo), Some(52_428_800 * 1024));
        assert_eq!(mem_total(meminfo), Some(65_000_000 * 1024));
        assert_eq!(mem_available("nothing"), None);

        assert_eq!(
            cgroup_path("0::/user.slice/run-1.scope\n"),
            Some("/user.slice/run-1.scope")
        );
        assert_eq!(cgroup_path("12:cpu:/x\n"), None);

        assert_eq!(cgroup_headroom("17179869184\n", "4294967296\n"), Some(12 * GIB));
        assert_eq!(cgroup_headroom("max\n", "1\n"), None);
        assert_eq!(cgroup_headroom("100", "400"), Some(0));

        assert_eq!(available_bytes(Some(10), Some(4)), Some(4));
        assert_eq!(available_bytes(None, Some(4)), Some(4));
        assert_eq!(available_bytes(None, None), None);
    }
}

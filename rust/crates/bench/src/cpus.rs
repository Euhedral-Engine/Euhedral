//! The benchmark's worker CPU selection: the configuration's `cpus` (`all`, `one-per-core`, `performance`,
//! `performance-one-per-core`, or processor ids and ranges such as `2-5,8`), then `excludeCpus` and `excludeCores`,
//! resolved to the explicit list the engine's `host.cpus` takes. It is the set the engine may run workers on, not a
//! pin.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::RangeInclusive;

use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::topology::Topology;
use euhedral_inference_settings::{Auto, CpuList, Settings};

use crate::config::ConfigError;

/// A named selection of the configuration's `cpus`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuSelection {
    All,
    OnePerCore,
    Performance,
    PerformanceOnePerCore,
    /// Explicit ids and inclusive ranges, as written (the same CPU named twice is selected once).
    Ids(Vec<RangeInclusive<u32>>),
}

impl CpuSelection {
    /// Parses the `cpus` text: a keyword, or ids and ranges. Each item is trimmed and empty items are skipped.
    pub fn parse(text: &str) -> Result<CpuSelection, ConfigError> {
        match text.trim() {
            "all" => Ok(CpuSelection::All),
            "one-per-core" => Ok(CpuSelection::OnePerCore),
            "performance" => Ok(CpuSelection::Performance),
            "performance-one-per-core" => Ok(CpuSelection::PerformanceOnePerCore),
            ids => {
                let items: Vec<String> = ids
                    .split(',')
                    .map(|item| item.chars().filter(|c| !c.is_whitespace()).collect::<String>())
                    .filter(|item| !item.is_empty())
                    .collect();
                if items.is_empty() {
                    return invalid("cpus selects no processors");
                }
                let list = CpuList::parse(&items.join(",")).map_err(|_| {
                    ConfigError::Invalid(format!(
                        "cpus must be all, one-per-core, performance, performance-one-per-core or processor ids and \
                         ranges such as 2-5,8 (got \"{ids}\")"
                    ))
                })?;
                Ok(CpuSelection::Ids(list.ranges().to_vec()))
            }
        }
    }
}

/// One logical CPU: the core it belongs to and whether that core is a performance core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuEntry {
    pub cpu: u32,
    pub core: u32,
    pub performance: bool,
}

/// The host's CPUs with their cores, and which of them the process may use.
///
/// `known` is every CPU of the topology: the exclusions are checked against it. `available` (the online CPUs inside
/// the process's affinity mask) is what a selection draws from. A platform that does not tell performance cores from
/// efficiency cores marks every CPU as performance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    entries: BTreeMap<u32, CpuEntry>,
    available: BTreeSet<u32>,
}

impl CpuTopology {
    /// The topology of `entries`, of which the CPUs in `available` may be used (CPUs not among the entries are
    /// ignored).
    pub fn new(entries: impl IntoIterator<Item = CpuEntry>, available: impl IntoIterator<Item = u32>) -> CpuTopology {
        let entries: BTreeMap<u32, CpuEntry> = entries.into_iter().map(|entry| (entry.cpu, entry)).collect();
        let available = available.into_iter().filter(|cpu| entries.contains_key(cpu)).collect();
        CpuTopology { entries, available }
    }

    /// A topology whose every CPU is available.
    pub fn unrestricted(entries: impl IntoIterator<Item = CpuEntry>) -> CpuTopology {
        let entries: Vec<CpuEntry> = entries.into_iter().collect();
        let available = entries.iter().map(|entry| entry.cpu).collect::<Vec<_>>();
        CpuTopology::new(entries, available)
    }

    /// The host's topology; the available CPUs are those of the process's base affinity mask.
    pub fn system() -> CpuTopology {
        let topology = Topology::system();
        let mask = affinity::base_mask();
        CpuTopology::new(
            topology.cpu_infos().map(|info| CpuEntry {
                cpu: info.cpu as u32,
                core: info.core as u32,
                performance: topology.p_cpu_set().contains(info.cpu) || !topology.e_cpu_set().contains(info.cpu),
            }),
            topology
                .cpu_infos()
                .filter(|info| mask.contains(info.cpu))
                .map(|info| info.cpu as u32),
        )
    }

    fn one_per_core(&self, cpus: &BTreeSet<u32>) -> BTreeSet<u32> {
        let mut cores = BTreeSet::new();
        // Ascending ids: the first CPU met of a core is its lowest.
        cpus.iter()
            .copied()
            .filter(|cpu| cores.insert(self.entries[cpu].core))
            .collect()
    }

    fn performance(&self) -> BTreeSet<u32> {
        self.entries
            .values()
            .filter(|entry| entry.performance && self.available.contains(&entry.cpu))
            .map(|entry| entry.cpu)
            .collect()
    }

    fn outside<T>(&self, cpu: u32) -> Result<T, ConfigError> {
        invalid(format!(
            "cpus names processor {cpu}, which is not available to the process (available: {})",
            ResolvedCpus(self.available.iter().copied().collect())
        ))
    }

    /// The CPUs `selection` names, less `exclude_cpus` and every CPU of `exclude_cores`, ascending.
    pub fn resolve(
        &self,
        selection: &CpuSelection,
        exclude_cpus: &[u32],
        exclude_cores: &[u32],
    ) -> Result<ResolvedCpus, ConfigError> {
        let all = &self.available;
        let mut chosen = match selection {
            CpuSelection::All => all.clone(),
            CpuSelection::OnePerCore => self.one_per_core(all),
            CpuSelection::Performance => self.performance(),
            CpuSelection::PerformanceOnePerCore => self.one_per_core(&self.performance()),
            CpuSelection::Ids(ranges) => {
                let mut ids = BTreeSet::new();
                for range in ranges {
                    // Bounds first: a huge range is refused without being expanded.
                    if all.last().is_none_or(|last| range.end() > last) {
                        return self.outside(*range.end());
                    }
                    for cpu in range.clone() {
                        if !all.contains(&cpu) {
                            return self.outside(cpu);
                        }
                        ids.insert(cpu);
                    }
                }
                ids
            }
        };
        if chosen.is_empty() {
            return invalid("cpus selects no processors");
        }
        if let Some(unknown) = exclude_cpus.iter().find(|cpu| !self.entries.contains_key(cpu)) {
            return invalid(format!("excludeCpus names unknown processor {unknown}"));
        }
        let cores: BTreeSet<u32> = self.entries.values().map(|entry| entry.core).collect();
        if let Some(unknown) = exclude_cores.iter().find(|core| !cores.contains(core)) {
            return invalid(format!("excludeCores names unknown core {unknown}"));
        }
        for cpu in exclude_cpus {
            chosen.remove(cpu);
        }
        chosen.retain(|cpu| !exclude_cores.contains(&self.entries[cpu].core));
        if chosen.is_empty() {
            return invalid("excludeCpus and excludeCores leave no processors");
        }
        Ok(ResolvedCpus(chosen.into_iter().collect()))
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(message.into()))
}

/// The resolved worker CPUs, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCpus(Vec<u32>);

impl ResolvedCpus {
    pub fn ids(&self) -> &[u32] {
        &self.0
    }

    /// The list in the form `host.cpus` takes.
    pub fn to_list(&self) -> CpuList {
        // The text is built from ascending ids and ranges, which `CpuList::parse` accepts.
        CpuList::parse(&self.to_string()).unwrap_or_else(|_| unreachable!("ids and ranges are a valid CPU list"))
    }
}

/// Hands the resolved CPUs to the engine as `host.cpus`, the set its workers may run on.
pub fn apply(settings: &mut Settings, cpus: &ResolvedCpus) {
    settings.host.cpus = Auto::Value(cpus.to_list());
}

impl fmt::Display for ResolvedCpus {
    /// Ids and ranges such as `2-5,8`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let mut ids = self.0.iter().copied().peekable();
        while let Some(start) = ids.next() {
            let mut end = start;
            while ids.next_if_eq(&(end + 1)).is_some() {
                end += 1;
            }
            if !first {
                f.write_str(",")?;
            }
            first = false;
            if end == start {
                write!(f, "{start}")?;
            } else {
                write!(f, "{start}-{end}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use euhedral_inference_settings::ModelSettings;

    use super::*;

    /// Cores 0-3 are performance cores with two CPUs each (CPUs c and c+8, as Intel numbers SMT siblings), cores
    /// 4-7 are efficiency cores with one CPU each.
    fn hybrid() -> CpuTopology {
        let performance = (0..4).flat_map(|core| [(core, core), (core + 8, core)]);
        let efficiency = (4..8).map(|core| (core, core));
        CpuTopology::unrestricted(performance.chain(efficiency).map(|(cpu, core)| CpuEntry {
            cpu,
            core,
            performance: core < 4,
        }))
    }

    fn flat(cpus: u32, smt: bool) -> CpuTopology {
        CpuTopology::unrestricted((0..cpus).map(|cpu| CpuEntry {
            cpu,
            core: if smt { cpu / 2 } else { cpu },
            performance: true,
        }))
    }

    fn resolve(topology: &CpuTopology, cpus: &str, cpu: &[u32], core: &[u32]) -> Result<String, String> {
        let selection = CpuSelection::parse(cpus).map_err(|ConfigError::Invalid(m)| m)?;
        topology
            .resolve(&selection, cpu, core)
            .map(|resolved| resolved.to_string())
            .map_err(|ConfigError::Invalid(m)| m)
    }

    #[test]
    fn keywords_on_a_hybrid_host() {
        let host = hybrid();
        assert_eq!(resolve(&host, "all", &[], &[]).unwrap(), "0-11");
        assert_eq!(resolve(&host, "one-per-core", &[], &[]).unwrap(), "0-7");
        assert_eq!(resolve(&host, "performance", &[], &[]).unwrap(), "0-3,8-11");
        assert_eq!(resolve(&host, "performance-one-per-core", &[], &[]).unwrap(), "0-3");
    }

    #[test]
    fn smt_on_and_off() {
        assert_eq!(resolve(&flat(8, true), "one-per-core", &[], &[]).unwrap(), "0,2,4,6");
        assert_eq!(resolve(&flat(8, false), "one-per-core", &[], &[]).unwrap(), "0-7");
        // Without a distinction every CPU is a performance CPU.
        assert_eq!(resolve(&flat(8, true), "performance", &[], &[]).unwrap(), "0-7");
        assert_eq!(
            resolve(&flat(8, true), "performance-one-per-core", &[], &[]).unwrap(),
            "0,2,4,6"
        );
    }

    #[test]
    fn explicit_ids_and_ranges() {
        let host = flat(12, false);
        assert_eq!(resolve(&host, "2-5,8", &[], &[]).unwrap(), "2-5,8");
        assert_eq!(resolve(&host, " 8,2-3,3 ", &[], &[]).unwrap(), "2-3,8");
        for lenient in ["2-5, 8", "2 - 5,8", "2-5,,8,"] {
            assert_eq!(resolve(&host, lenient, &[], &[]).unwrap(), "2-5,8", "{lenient:?}");
        }
        assert!(resolve(&host, ",", &[], &[]).unwrap_err().contains("selects no"));
        assert!(resolve(&host, "2-12", &[], &[]).unwrap_err().contains("processor 12"));
        for bad in ["", " , ", "a", "5-2", "-1", "0-4000000000"] {
            assert!(resolve(&host, bad, &[], &[]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn exclusions() {
        let host = hybrid();
        assert_eq!(resolve(&host, "performance-one-per-core", &[], &[0]).unwrap(), "1-3");
        // Excluding a core removes both of its SMT CPUs.
        assert_eq!(resolve(&host, "all", &[], &[0, 4]).unwrap(), "1-3,5-7,9-11");
        assert_eq!(
            resolve(&host, "all", &[0, 8, 15], &[]).unwrap_err(),
            "excludeCpus names unknown processor 15"
        );
        assert_eq!(resolve(&host, "all", &[1, 9], &[7]).unwrap(), "0,2-6,8,10-11");
        // A CPU outside the selection is still a known one.
        assert_eq!(resolve(&host, "performance-one-per-core", &[5], &[]).unwrap(), "0-3");
        assert!(resolve(&host, "all", &[], &[8]).unwrap_err().contains("unknown core 8"));
    }

    #[test]
    fn an_empty_result_is_refused() {
        let host = hybrid();
        assert!(
            resolve(&host, "performance-one-per-core", &[], &[0, 1, 2, 3])
                .unwrap_err()
                .contains("leave no")
        );
        assert!(resolve(&host, "2", &[2], &[]).unwrap_err().contains("leave no"));
        let efficiency_only = CpuTopology::unrestricted((0..2).map(|cpu| CpuEntry {
            cpu,
            core: cpu,
            performance: false,
        }));
        assert!(
            resolve(&efficiency_only, "performance", &[], &[])
                .unwrap_err()
                .contains("selects no")
        );
    }

    #[test]
    fn the_list_round_trips_into_host_cpus() {
        let resolved = hybrid().resolve(&CpuSelection::Performance, &[], &[]).unwrap();
        assert_eq!(resolved.to_list().ids().collect::<Vec<_>>(), resolved.ids());
        assert_eq!(resolved.to_string(), "0-3,8-11");
    }

    /// Eight CPUs, two per core; the process may use only 2-5.
    fn masked() -> CpuTopology {
        let entries = (0..8).map(|cpu| CpuEntry {
            cpu,
            core: cpu / 2,
            performance: cpu < 6,
        });
        CpuTopology::new(entries, 2..6)
    }

    #[test]
    fn the_process_mask_bounds_the_selection_but_not_the_exclusions() {
        let host = masked();
        assert_eq!(resolve(&host, "all", &[], &[]).unwrap(), "2-5");
        assert_eq!(resolve(&host, "one-per-core", &[], &[]).unwrap(), "2,4");
        assert_eq!(resolve(&host, "performance", &[], &[]).unwrap(), "2-5");
        assert_eq!(resolve(&host, "performance-one-per-core", &[], &[]).unwrap(), "2,4");
        assert!(resolve(&host, "1-3", &[], &[]).unwrap_err().contains("processor 1"));
        assert!(resolve(&host, "5-6", &[], &[]).unwrap_err().contains("processor 6"));
        assert_eq!(resolve(&host, "2-5", &[], &[]).unwrap(), "2-5");
        // Known CPUs and cores outside the mask may be excluded; ones the host does not have may not.
        assert_eq!(resolve(&host, "all", &[0, 7], &[2]).unwrap(), "2-3");
        assert_eq!(resolve(&host, "all", &[], &[0]).unwrap(), "2-5");
        assert!(resolve(&host, "all", &[8], &[]).is_err());
        assert!(resolve(&host, "all", &[], &[4]).is_err());
    }

    #[test]
    fn the_host_topology_is_consistent() {
        let host = CpuTopology::system();
        let all = host.resolve(&CpuSelection::All, &[], &[]).unwrap();
        for selection in [
            CpuSelection::OnePerCore,
            CpuSelection::Performance,
            CpuSelection::PerformanceOnePerCore,
        ] {
            let chosen = host.resolve(&selection, &[], &[]).unwrap();
            assert!(chosen.ids().iter().all(|cpu| all.ids().contains(cpu)), "{selection:?}");
        }
    }

    #[test]
    fn apply_sets_host_cpus() {
        let mut settings = Settings::with_model(ModelSettings {
            artifact: "a.edrl".into(),
            name: "a".into(),
        });
        let resolved = hybrid().resolve(&CpuSelection::Performance, &[], &[0]).unwrap();
        apply(&mut settings, &resolved);
        assert_eq!(settings.host.cpus, Auto::Value(CpuList::parse("1-3,9-11").unwrap()));
    }
}

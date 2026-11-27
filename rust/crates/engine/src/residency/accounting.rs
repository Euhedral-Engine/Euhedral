//! The account of the engine's memory: one admitted reservation of the process's memory planner, from which every
//! allocation of the startup takes a claim and holds it for as long as it lives.
//!
//! The plan names its consumers ([`ResidencyPlan::consumers`]):
//! the same list is what the memory planner admits, so the reservation and the plan cannot differ. An allocation
//! takes its claim from [`PlanReservation::claim`] before it allocates, so that one larger than the plan fails the
//! startup and names its consumer. At shutdown [`MemoryAccount::close`] ends the owner of the claims: whatever one
//! still holds is a leak, and is reported.
//!
//! What the reservation holds beyond the claims (the part of the runtime reserve nothing allocates: the kernel
//! modules and RustEE's own allocations) belongs to the account's own owner, not to the holders, so it is no leak.

use std::fmt;
use std::sync::Arc;

use euhedral_memory::{
    Claim, LeakReport, LocationId, LocationKind, LocationSpec, OwnerId, OwnerSpec, Placed, PlanItem, PlanRequest,
    Planner, Priority, Reservation, Scope,
};

use crate::error::{EngineError, PlanError};
use crate::residency::planner::{CAPACITY_CAP, LocationBytes, ResidencyPlan};

/// What the plan sets memory aside for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Consumer {
    /// The weight arena.
    Weights,
    /// What the runtime needs beyond the weights, the cache and the shared scratch: the workspace's slots and its
    /// second scratch region, the first sequence's state, and the kernel modules.
    RuntimeReserve,
    /// The workspace's first scratch region, the linear routes' expansion.
    Scratch,
    /// The device cache of every sequence.
    Cache,
    /// The state of every sequence beyond the first, besides its cache.
    ExtraSequences,
    /// The device ring that weights held outside the device are copied through before they are read.
    StagingRing,
    /// The pinned host pages of the cache of every sequence.
    HostPages,
    /// The pinned host arena of the weights the device does not hold, the embedding and the output head among them.
    HostWeights,
    /// The pinned host slots that reads of the artifact file land in.
    DiskWindow,
    /// The prefix cache's pinned arena, the one consumer the plan may drop.
    PrefixCache,
    /// The device slots the routed experts of a Flash-Next model are cached in.
    ExpertCache,
    /// The pinned host slots every Flash-Next expert record passes through on its way to the device.
    ExpertStaging,
}

impl Consumer {
    /// The name the memory planner and the errors know it by.
    pub fn name(self) -> &'static str {
        match self {
            Consumer::Weights => "weights",
            Consumer::RuntimeReserve => "runtime reserve",
            Consumer::Scratch => "scratch",
            Consumer::Cache => "cache",
            Consumer::ExtraSequences => "state of the sequences beyond the first",
            Consumer::StagingRing => "staging ring",
            Consumer::HostPages => "cache pages on the host",
            Consumer::HostWeights => "weights on the host",
            Consumer::DiskWindow => "disk window",
            Consumer::PrefixCache => "prefix cache",
            Consumer::ExpertCache => "expert cache",
            Consumer::ExpertStaging => "expert staging",
        }
    }
}

impl fmt::Display for Consumer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What the plan asks of one consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerBytes {
    pub consumer: Consumer,
    /// Pinned host memory when true, else the device.
    pub host: bool,
    pub bytes: u64,
    /// A required consumer is never dropped: the plan is refused instead.
    pub required: bool,
}

impl ConsumerBytes {
    pub(crate) fn device(consumer: Consumer, bytes: u64) -> ConsumerBytes {
        ConsumerBytes {
            consumer,
            host: false,
            bytes,
            required: true,
        }
    }

    pub(crate) fn host(consumer: Consumer, bytes: u64) -> ConsumerBytes {
        ConsumerBytes {
            consumer,
            host: true,
            bytes,
            required: true,
        }
    }

    pub(crate) fn optional_host(consumer: Consumer, bytes: u64) -> ConsumerBytes {
        ConsumerBytes {
            consumer,
            host: true,
            bytes,
            required: false,
        }
    }
}

/// The items of the memory planner's request for `consumers`: one each, in their order, the one that is not required
/// at the lowest priority.
pub(crate) fn plan_items(consumers: &[ConsumerBytes], host: LocationId, device: LocationId) -> Vec<PlanItem> {
    consumers
        .iter()
        .map(|asked| {
            let location = if asked.host { host } else { device };
            if asked.required {
                PlanItem::required(asked.consumer.name(), location, asked.bytes)
            } else {
                PlanItem::optional(asked.consumer.name(), Priority(0), location, asked.bytes)
            }
        })
        .collect()
}

/// The process's memory planner and the owners the engine's allocations are accounted to.
pub struct MemoryAccount {
    planner: Arc<Planner>,
    /// The owner the plan's reservation is accounted to.
    owner: OwnerId,
    /// A child of `owner`: the claims of the allocations. What it holds when it ends is a leak.
    holders: OwnerId,
    host: LocationId,
    device: LocationId,
}

impl fmt::Debug for MemoryAccount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("MemoryAccount").finish_non_exhaustive()
    }
}

impl MemoryAccount {
    /// An account of a device with `device_capacity` bytes. The capacities a plan is admitted against are the ones it
    /// was made against: [`MemoryAccount::admit`] sets them, the host's from the plan's own host budget.
    pub fn open(device_capacity: u64) -> MemoryAccount {
        let planner = Planner::with_owners(
            [
                LocationSpec {
                    kind: LocationKind::HostPinned,
                    capacity: 0,
                    reserve: 0,
                },
                LocationSpec {
                    kind: LocationKind::Device { ordinal: 0 },
                    capacity: device_capacity,
                    reserve: 0,
                },
            ],
            2,
        );
        let (host, device) = (LocationId(0), LocationId(1));
        let owner = planner.open_owner(OwnerSpec::new("engine memory", Priority(0), Scope::Plan(0)));
        let holders =
            planner.open_owner(OwnerSpec::new("engine allocations", Priority(0), Scope::Plan(0)).child_of(owner));
        MemoryAccount {
            planner,
            owner,
            holders,
            host,
            device,
        }
    }

    /// Admits `plan`: reserves its consumers at the capacities it was made against (the device bytes it was made
    /// against, and the host account's). The decision was made when the plan was; this reserves it, and refuses when
    /// the two disagree.
    pub fn admit(&self, plan: &ResidencyPlan) -> Result<PlanReservation, PlanError> {
        let consumers = plan.consumers();
        let (mut host_bytes, mut device_bytes) = (0u64, 0u64);
        for asked in &consumers {
            let total = if asked.host { &mut host_bytes } else { &mut device_bytes };
            *total = total.saturating_add(asked.bytes);
        }
        // An unknown host budget allowed no spill: the host holds what the plan admits there (the prefix arena and
        // any host pages), no more.
        let host_budget = plan.host_budget_bytes.unwrap_or(host_bytes);
        self.planner
            .set_capacity(self.host, host_budget.min(CAPACITY_CAP.max(host_bytes)));
        self.planner
            .set_capacity(self.device, plan.budget_bytes.min(CAPACITY_CAP.max(device_bytes)));
        let request = PlanRequest {
            owner: self.owner,
            items: plan_items(&consumers, self.host, self.device),
        };
        let admission = self
            .planner
            .admit(&request)
            .map_err(|shortfall| PlanError::NotAdmitted(shortfall.to_string()))?;
        if let Some(dropped) = admission.dropped().next() {
            return Err(PlanError::NotAdmitted(format!(
                "the {} was dropped",
                consumers[dropped].consumer
            )));
        }
        let admitted = LocationBytes {
            host: admission.bytes.get(usize::from(self.host.0)).copied().unwrap_or(0),
            device: admission.bytes.get(usize::from(self.device.0)).copied().unwrap_or(0),
        };
        debug_assert!(admission.placed.iter().all(|placed| *placed == Placed::Preferred));
        let Some(reservation) = admission.reservation else {
            return Err(PlanError::NotAdmitted(
                "the memory planner ordered give-backs and made no reservation".to_string(),
            ));
        };
        Ok(PlanReservation {
            reservation,
            holders: self.holders,
            host: self.host,
            device: self.device,
            left: consumers,
            admitted,
        })
    }

    /// Ends the account: the claims still held by the allocations are leaks. Whatever the reservation still holds is
    /// given back and is none. Run it once the allocations are freed. Takes `&self` because the account is shared
    /// with the frames of the engine; a second call finds nothing to report.
    pub fn close(&self) -> LeakReport {
        let report = self.planner.end_owner(self.holders);
        // The reservation's own bytes, which no allocation took: not a leak.
        self.planner.end_owner(self.owner);
        report
    }

    /// The bytes the account's owners hold now, per location.
    pub fn held(&self) -> LocationBytes {
        let snapshot = self.planner.snapshot();
        LocationBytes {
            host: snapshot.used(self.host),
            device: snapshot.used(self.device),
        }
    }
}

/// The plan's admitted reservation: what each consumer may still take.
#[derive(Debug)]
pub struct PlanReservation {
    reservation: Reservation,
    holders: OwnerId,
    host: LocationId,
    device: LocationId,
    left: Vec<ConsumerBytes>,
    admitted: LocationBytes,
}

impl PlanReservation {
    /// The bytes admitted, per location.
    pub fn admitted(&self) -> LocationBytes {
        self.admitted
    }

    /// The bytes `consumer` may still claim.
    pub fn left(&self, consumer: Consumer) -> u64 {
        self.left
            .iter()
            .filter(|asked| asked.consumer == consumer)
            .map(|asked| asked.bytes)
            .sum()
    }

    /// A claim of `bytes` of `consumer`'s share, to be held for as long as the allocation it accounts for. Refused,
    /// naming the consumer, when the plan has not that many bytes left for it: the allocation is larger than planned.
    pub fn claim(&mut self, consumer: Consumer, bytes: u64) -> Result<Claim, EngineError> {
        let refused = |left: u64| {
            let setting = match consumer {
                Consumer::RuntimeReserve => " (raise memory.runtime-reserve)",
                _ => "",
            };
            EngineError::Runtime(format!(
                "the {consumer} needs {bytes} bytes and the plan admitted {left} bytes more for it{setting}"
            ))
        };
        let Some(asked) = self.left.iter_mut().find(|asked| asked.consumer == consumer) else {
            return Err(refused(0));
        };
        if bytes > asked.bytes {
            return Err(refused(asked.bytes));
        }
        let location = if asked.host { self.host } else { self.device };
        let Some(claim) = self.reservation.take(self.holders, location, bytes, consumer.name()) else {
            return Err(refused(asked.bytes));
        };
        asked.bytes -= bytes;
        Ok(claim)
    }
}

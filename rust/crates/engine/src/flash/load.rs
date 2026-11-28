//! Where the fixed objects of a Flash-Next plan live, and what the upload reads for them.
//!
//! The plan names a storage class for each object of the artifact ([`FlashPlan::objects`], in table order). This
//! module turns it into places: an offset in the device weight arena for each resident object, an offset in the pinned
//! host arena for each object kernels read in place, and the file range of each n-gram table, which is read where it
//! lies. [`FixedLayout::pieces`] cuts what the arenas receive into the upload's pieces
//! ([`crate::residency::loader`]), and [`FixedLayout::resolve`] names every object by the address it has once the load
//! has ended ([`FixedWeights`]). The shared expert's down projections are widened to the width the block kernels read
//! ([`padding`]), as Java's `Weights.padSharedDown` does.
//!
//! A plan that stages fixed objects through a ring, or pins the n-gram tables, is refused with the object's name: the
//! upload serves the plans of a device that holds the fixed objects.

use std::collections::HashMap;

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{ComponentGroup, FlashTensor, WeightLayout};

use super::linear::Nvfp4Weight;
use super::moe::ops::{nvfp4_bytes, nvfp4_pad_k};
use super::{FlashError, Weight};
use crate::formats::shapes::Launch;
use crate::residency::flash::{FlashPlan, StorageClass};
use crate::residency::loader::{self, Piece};
use crate::residency::offload::HOST_ALIGNMENT;
use crate::residency::planner::{ARENA_ALIGNMENT, Placement, PlacementSource, Tier};

const OP: &str = "flash weights";

/// Where one object is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    /// `offset` bytes into the device weight arena.
    Device { offset: u64 },
    /// `offset` bytes into the pinned host arena, which kernels read in place.
    Host { offset: u64 },
    /// In the artifact file, `bytes` at `offset`; nothing is uploaded.
    File { offset: u64, bytes: u64 },
}

/// The homes of a plan's fixed objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedLayout {
    placements: Vec<Placement>,
    homes: Vec<Home>,
    device_bytes: u64,
    host_bytes: u64,
}

impl FixedLayout {
    /// The homes of `plan`'s objects over the artifact's `tensors`. Offsets follow table order, each rounded up to the
    /// device allocation alignment (device) or to the file system block (host). Refuses a plan that is not over these
    /// tensors, and an object the upload does not serve: staged through a ring, cached, or an n-gram table that is
    /// not read from the file. An object the plan defers has no home.
    pub fn of(plan: &FlashPlan, tensors: &[FlashTensor]) -> Result<FixedLayout, FlashError> {
        if plan.objects.len() != tensors.len() {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the plan places {} objects and the artifact has {} tensors",
                    plan.objects.len(),
                    tensors.len()
                ),
            ));
        }
        let (mut placements, mut homes) = (Vec::new(), Vec::new());
        let (mut device_bytes, mut host_bytes) = (0u64, 0u64);
        for (index, (object, tensor)) in plan.objects.iter().zip(tensors).enumerate() {
            if object.name != tensor.name || object.bytes != tensor.bytes {
                return Err(FlashError::refused(
                    OP,
                    format!(
                        "object {index} of the plan is {} of {} bytes, the artifact's is {} of {} bytes",
                        object.name, object.bytes, tensor.name, tensor.bytes
                    ),
                ));
            }
            let (home, tier, offset) = match object.storage {
                StorageClass::Deferred => continue,
                StorageClass::DeviceResident => {
                    let offset = device_bytes;
                    device_bytes = take(device_bytes, tensor, ARENA_ALIGNMENT)?;
                    (Home::Device { offset }, Tier::Device, offset)
                }
                StorageClass::HostMapped => {
                    let offset = host_bytes;
                    host_bytes = take(host_bytes, tensor, HOST_ALIGNMENT)?;
                    (Home::Host { offset }, Tier::Host { offset }, 0)
                }
                StorageClass::FileMapped if object.group == ComponentGroup::Ngram => (
                    Home::File {
                        offset: tensor.offset,
                        bytes: tensor.bytes,
                    },
                    Tier::Disk,
                    0,
                ),
                other => {
                    return Err(FlashError::refused(
                        OP,
                        format!(
                            "{} is placed as {other:?}, which the upload does not serve (it holds fixed objects on \
                             the device, the embedding in pinned host memory and the n-gram tables in the file)",
                            tensor.name
                        ),
                    ));
                }
            };
            homes.push(home);
            placements.push(Placement {
                name: tensor.name.clone(),
                source: PlacementSource::Object,
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
                format: tensor.format,
                layout: tensor.layout,
                file_offset: tensor.offset,
                bytes: tensor.bytes,
                offset,
                tier,
            });
        }
        Ok(FixedLayout {
            placements,
            homes,
            device_bytes,
            host_bytes,
        })
    }

    /// The objects that have a home, in table order. A piece's `placement` indexes this list.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    /// Where the object `name` is; None for a deferred or unknown name.
    pub fn home(&self, name: &str) -> Option<Home> {
        self.placements
            .iter()
            .position(|placement| placement.name == name)
            .map(|index| self.homes[index])
    }

    /// The size of the device weight arena.
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// The size of the pinned host arena; 0 when nothing is read in place.
    pub fn host_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// The reads that fill the arenas: each object of the device and host homes whole, or in the regions its format
    /// validates, cut to at most `piece_bytes` (at least 1), in file order. The n-gram tables have none.
    pub fn pieces(&self, piece_bytes: u64) -> Vec<Piece> {
        loader::pieces_of(&self.placements, piece_bytes.max(1))
    }

    /// Names every object by where it is once the arenas are loaded: `device` is the start of the device arena and
    /// `host` the address kernels read the host arena at. Refuses a host object with no host arena.
    pub fn resolve(&self, device: DevicePtr, host: Option<DevicePtr>) -> Result<FixedWeights, FlashError> {
        let mut by_name = HashMap::with_capacity(self.placements.len());
        let mut files = HashMap::new();
        for (placement, home) in self.placements.iter().zip(&self.homes) {
            let address = match *home {
                Home::Device { offset } => device.offset(offset),
                Home::Host { offset } => host
                    .ok_or_else(|| {
                        FlashError::refused(
                            OP,
                            format!(
                                "{} is read in place from host memory and there is no host arena",
                                placement.name
                            ),
                        )
                    })?
                    .offset(offset),
                Home::File { offset, bytes } => {
                    files.insert(placement.name.clone(), (offset, bytes));
                    continue;
                }
            };
            by_name.insert(
                placement.name.clone(),
                Weight {
                    address,
                    bytes: placement.bytes,
                },
            );
        }
        let sd4 = self
            .placements
            .iter()
            .map(|placement| {
                (
                    placement.name.clone(),
                    placement.layout == WeightLayout::RowSplitK128Sd4V1,
                )
            })
            .collect();
        Ok(FixedWeights { by_name, files, sd4 })
    }
}

/// `used` plus the tensor's bytes, rounded up to `alignment`.
fn take(used: u64, tensor: &FlashTensor, alignment: u64) -> Result<u64, FlashError> {
    used.checked_add(tensor.bytes)
        .map(|end| end.next_multiple_of(alignment))
        .ok_or_else(|| FlashError::refused(OP, format!("{} overflows the arena's 64-bit size", tensor.name)))
}

/// The fixed objects by name, at the addresses the load gave them.
#[derive(Clone, Debug)]
pub struct FixedWeights {
    by_name: HashMap<String, Weight>,
    files: HashMap<String, (u64, u64)>,
    sd4: HashMap<String, bool>,
}

impl FixedWeights {
    /// Weights with no objects: every lookup refuses.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            by_name: HashMap::new(),
            files: HashMap::new(),
            sd4: HashMap::new(),
        }
    }

    /// The object `name`. Refuses a name the plan did not place on the device or in host memory, naming it.
    pub fn weight(&self, name: &str) -> Result<Weight, FlashError> {
        self.by_name
            .get(name)
            .copied()
            .ok_or_else(|| FlashError::refused(OP, format!("{name} is not a loaded weight of this plan")))
    }

    /// The NVFP4 object `name` with its scale table's layout.
    pub fn nvfp4(&self, name: &str) -> Result<Nvfp4Weight, FlashError> {
        let weight = self.weight(name)?;
        let sd4 = self.sd4.get(name).copied().unwrap_or(false);
        Ok(Nvfp4Weight { weight, sd4 })
    }

    /// The file range `(offset, bytes)` of the n-gram table `name`.
    pub fn file_range(&self, name: &str) -> Result<(u64, u64), FlashError> {
        self.files
            .get(name)
            .copied()
            .ok_or_else(|| FlashError::refused(OP, format!("{name} is not an object read from the file")))
    }
}

/// The text layers' tensor name prefix.
pub fn layer_prefix(layer: u32) -> String {
    format!("text/layers/{layer}/")
}

/// One shared down projection to widen: the layer, the tensor, and where the widened copy goes in the padding region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadJob {
    pub layer: u32,
    pub source: Weight,
    pub target_offset: u64,
    pub target_bytes: u64,
}

/// The shared down projections to widen from `width` to `padded` values (`hidden` rows each) and the bytes of the
/// region the copies go to: one job for each of the `layers` whose tensor is a plain NVFP4 tensor of `hidden` by
/// `width` (what Java tests by its byte size). None when the width needs no padding.
pub fn padding(weights: &FixedWeights, layers: u32, hidden: u32, width: u32, padded: u32) -> (Vec<PadJob>, u64) {
    if padded == width {
        return (Vec::new(), 0);
    }
    let (plain, target_bytes) = (nvfp4_bytes(hidden, width), nvfp4_bytes(hidden, padded));
    let (mut jobs, mut total) = (Vec::new(), 0u64);
    for layer in 0..layers {
        let name = format!("{}moe/shared_expert/down_proj", layer_prefix(layer));
        let Ok(source) = weights.weight(&name) else { continue };
        if source.bytes != plain {
            continue;
        }
        jobs.push(PadJob {
            layer,
            source,
            target_offset: total,
            target_bytes,
        });
        total += target_bytes.next_multiple_of(ARENA_ALIGNMENT);
    }
    (jobs, total)
}

/// The launches that widen `jobs` into the region at `target` (`MoeOps.nvfp4PadK`), in order. Refuses a width the
/// kernel cannot take.
pub fn pad_launches(
    jobs: &[PadJob],
    target: DevicePtr,
    hidden: u32,
    width: u32,
    padded: u32,
) -> Result<Vec<Launch>, FlashError> {
    jobs.iter()
        .map(|job| {
            nvfp4_pad_k(
                job.source.address,
                target.offset(job.target_offset),
                hidden,
                width,
                padded,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;

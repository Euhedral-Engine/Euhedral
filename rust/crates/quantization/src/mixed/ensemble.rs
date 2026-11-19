//! Fidelity models fitted to bootstrap resamples of the measured candidates, each shrinking towards its own random point
//! of the prior (see [`FidelityModel::fit_perturbed`]): their mean is the prediction, their spread how unsure it is.
//! Where the data decides a weight the members agree; where it does not they differ.

use super::allocation::Allocation;
use super::error::MixedError;
use super::fidelity::{DEFAULT_BANDS, FidelityModel};
use super::measured::Measured;
use super::priors::Priors;
use crate::java_random::Random;

/// The prior perturbation of the members.
pub const PERTURB: f64 = 0.3;

/// The fit of one member: the bootstrap resample it is fitted to, as indices into the measured candidates, and its
/// seed. Every resample and seed is drawn in member order before any member is fitted, so the members are the same
/// whichever order, or on whichever threads, the fits run in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberFit {
    resample: Vec<usize>,
    seed: i64,
}

impl MemberFit {
    /// The member fitted to its resample of `data`, which must be the data the fits were drawn for.
    pub fn fit(
        &self,
        data: &[Measured],
        ridge: f64,
        priors: &Priors,
        bands: usize,
    ) -> Result<FidelityModel, MixedError> {
        let mut resample = Vec::with_capacity(self.resample.len());
        for &at in &self.resample {
            resample.push(
                data.get(at)
                    .ok_or_else(|| MixedError::invalid("the member was drawn for more measured candidates"))?
                    .clone(),
            );
        }
        FidelityModel::fit_perturbed(&resample, ridge, priors, bands, self.seed, PERTURB)
    }
}

/// The fitted members.
#[derive(Debug, Clone)]
pub struct Ensemble {
    members: Vec<FidelityModel>,
}

impl Ensemble {
    /// `members` models fitted to resamples of `data` drawn from `seed`, with the default bands.
    pub fn fit(
        data: &[Measured],
        members: usize,
        seed: i64,
        ridge: f64,
        priors: &Priors,
    ) -> Result<Ensemble, MixedError> {
        Ensemble::fit_banded(data, members, seed, ridge, priors, DEFAULT_BANDS)
    }

    /// As [`fit`](Self::fit) over `bands` bands per family.
    pub fn fit_banded(
        data: &[Measured],
        members: usize,
        seed: i64,
        ridge: f64,
        priors: &Priors,
        bands: usize,
    ) -> Result<Ensemble, MixedError> {
        let mut fitted = Vec::with_capacity(members);
        for fit in Ensemble::member_fits(data.len(), members, seed)? {
            fitted.push(fit.fit(data, ridge, priors, bands)?);
        }
        Ensemble::from_members(fitted)
    }

    /// The fit of each member to its bootstrap resample of `count` measured candidates, the resamples and seeds drawn
    /// from `seed` in member order. A caller that runs the fits as frames of its own draws them here, runs each
    /// [`MemberFit::fit`] and joins the models, in this order, with [`from_members`](Self::from_members).
    pub fn member_fits(count: usize, members: usize, seed: i64) -> Result<Vec<MemberFit>, MixedError> {
        if members < 1 {
            return Err(MixedError::invalid("an ensemble needs a member"));
        }
        let bound = i32::try_from(count).map_err(|_| MixedError::invalid("too many measured candidates"))?;
        let mut random = Random::new(seed);
        let mut fits = Vec::with_capacity(members);
        for _ in 0..members {
            let mut resample = Vec::with_capacity(count);
            for _ in 0..count {
                let at = random.next_int_bounded(bound).unwrap_or(0);
                resample.push(at as usize);
            }
            fits.push(MemberFit {
                resample,
                seed: random.next_long(),
            });
        }
        Ok(fits)
    }

    /// The ensemble of `members`, which must be fitted to the same objects and options.
    pub fn from_members(members: Vec<FidelityModel>) -> Result<Ensemble, MixedError> {
        let Some(first) = members.first() else {
            return Err(MixedError::invalid("an ensemble needs a member"));
        };
        if members
            .iter()
            .any(|member| member.objects() != first.objects() || member.options() != first.options())
        {
            return Err(MixedError::invalid("the members differ in their objects or options"));
        }
        Ok(Ensemble { members })
    }

    /// The members.
    pub fn members(&self) -> &[FidelityModel] {
        &self.members
    }

    /// A member, for the objects and options all members share.
    pub fn layout(&self) -> &FidelityModel {
        &self.members[0]
    }

    /// The mean of the members' predictions.
    pub fn mean(&self, allocation: &Allocation) -> Result<f64, MixedError> {
        let mut sum = 0.0;
        for member in &self.members {
            sum += member.predict(allocation)?;
        }
        Ok(sum / self.members.len() as f64)
    }

    /// The standard deviation of the members' predictions.
    pub fn spread(&self, allocation: &Allocation) -> Result<f64, MixedError> {
        let mean = self.mean(allocation)?;
        let mut squares = 0.0;
        for member in &self.members {
            let difference = member.predict(allocation)? - mean;
            squares += difference * difference;
        }
        Ok((squares / self.members.len() as f64).sqrt())
    }

    /// The mean of the members' predictions of the soft assignment `p`.
    pub fn mean_soft(&self, p: &[Vec<f64>]) -> Result<f64, MixedError> {
        let mut sum = 0.0;
        for member in &self.members {
            sum += member.predict_soft(p)?;
        }
        Ok(sum / self.members.len() as f64)
    }

    /// The derivative of [`mean_soft`](Self::mean_soft).
    pub fn gradient(&self, p: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MixedError> {
        let mut sum: Option<Vec<Vec<f64>>> = None;
        for member in &self.members {
            let gradient = member.gradient(p)?;
            match &mut sum {
                None => sum = Some(gradient),
                Some(total) => {
                    for (row, add) in total.iter_mut().zip(&gradient) {
                        for (slot, value) in row.iter_mut().zip(add) {
                            *slot += value;
                        }
                    }
                }
            }
        }
        let mut total = sum.unwrap_or_default();
        for row in &mut total {
            for slot in row {
                *slot /= self.members.len() as f64;
            }
        }
        Ok(total)
    }

    /// The mean over `unseen` of the absolute difference of the log of the mean prediction and of the measured KL; 0
    /// for none.
    pub fn held_out_error(&self, unseen: &[Measured]) -> Result<f64, MixedError> {
        if unseen.is_empty() {
            return Ok(0.0);
        }
        let mut sum = 0.0;
        for measured in unseen {
            sum += (self.mean(&measured.allocation)?.max(1e-9).ln() - measured.kl.max(1e-9).ln()).abs();
        }
        Ok(sum / unseen.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::mixed::option::QuantOption;
    use crate::mixed::synthetic::{OPTIONS, Synthetic};

    /// Allocations that vary only in the first half of the layers (`varied`) or only in the rest; the other half stays
    /// on NVFP4.
    fn front(world: &Synthetic, random: &mut Random, varied: bool) -> Allocation {
        let mut choices = BTreeMap::new();
        for object in &world.objects {
            let layer: usize = object.split('/').nth(2).unwrap().parse().unwrap();
            let first = layer < world.layers / 2;
            let option = if first == varied {
                OPTIONS[random.next_int_bounded(OPTIONS.len() as i32).unwrap() as usize]
            } else {
                QuantOption::Nvfp4
            };
            choices.insert(object.clone(), option);
        }
        Allocation::new(choices)
    }

    fn measured(world: &Synthetic, count: usize, seed: i64, front_only: bool) -> Vec<Measured> {
        let mut random = Random::new(seed);
        (0..count)
            .map(|_| {
                let allocation = if front_only {
                    front(world, &mut random, true)
                } else {
                    world.random(&mut random, &OPTIONS).unwrap()
                };
                world.measured(allocation).unwrap()
            })
            .collect()
    }

    #[test]
    fn the_spread_is_larger_off_the_training_distribution() {
        let world = Synthetic::new(16, 4, 1.2, 3.0, 3);
        let mut random = Random::new(1);
        let data: Vec<Measured> = (0..80)
            .map(|_| world.measured(front(&world, &mut random, true)).unwrap())
            .collect();
        let ensemble = Ensemble::fit_banded(&data, 8, 42, 1e-3, &world.priors, 4).unwrap();
        let (mut inside, mut outside) = (0.0, 0.0);
        for _ in 0..30 {
            inside += ensemble.spread(&front(&world, &mut random, true)).unwrap();
            outside += ensemble.spread(&front(&world, &mut random, false)).unwrap();
        }
        assert!(outside > 1.5 * inside, "inside {inside} outside {outside}");
    }

    #[test]
    fn the_mean_tracks_the_truth_and_the_held_out_error_is_small() {
        let world = Synthetic::new(16, 4, 1.2, 3.0, 3);
        let data = measured(&world, 300, 2, false);
        let unseen = measured(&world, 30, 12, false);
        let ensemble = Ensemble::fit_banded(&data, 6, 7, 1e-6, &world.priors, 4).unwrap();
        let held_out = ensemble.held_out_error(&unseen).unwrap();
        assert!(held_out < 0.05, "held-out {held_out}");
        assert_eq!(ensemble.held_out_error(&[]).unwrap(), 0.0);
    }

    #[test]
    fn the_same_seed_gives_the_same_ensemble_and_another_a_different_one() {
        let world = Synthetic::new(8, 4, 1.2, 3.0, 3);
        let data = measured(&world, 60, 2, false);
        let probe = world.random(&mut Random::new(9), &OPTIONS).unwrap();
        let mean = |seed| {
            Ensemble::fit_banded(&data, 4, seed, 1e-3, &world.priors, 4)
                .unwrap()
                .mean(&probe)
                .unwrap()
        };
        assert_eq!(mean(5), mean(5));
        assert_ne!(mean(5), mean(6));
    }

    #[test]
    fn members_fitted_in_any_order_give_the_sequential_ensemble() {
        let world = Synthetic::new(8, 4, 1.2, 3.0, 3);
        let data = measured(&world, 60, 3, false);
        let sequential = Ensemble::fit_banded(&data, 5, 11, 1e-3, &world.priors, 4).unwrap();
        let fits = Ensemble::member_fits(data.len(), 5, 11).unwrap();
        // The members finish in the reverse of the order they were drawn in.
        let mut done: Vec<(usize, FidelityModel)> = fits
            .iter()
            .enumerate()
            .rev()
            .map(|(at, fit)| (at, fit.fit(&data, 1e-3, &world.priors, 4).unwrap()))
            .collect();
        done.sort_by_key(|(at, _)| *at);
        let by_members = Ensemble::from_members(done.into_iter().map(|(_, model)| model).collect()).unwrap();
        assert_eq!(sequential.members().len(), by_members.members().len());
        for probe in 0..10 {
            let allocation = world.random(&mut Random::new(probe), &OPTIONS).unwrap();
            assert_eq!(
                sequential.mean(&allocation).unwrap(),
                by_members.mean(&allocation).unwrap()
            );
            assert_eq!(
                sequential.spread(&allocation).unwrap(),
                by_members.spread(&allocation).unwrap()
            );
        }
    }

    #[test]
    fn the_mean_soft_gradient_is_the_mean_of_the_members_and_a_member_is_required() {
        let world = Synthetic::new(4, 2, 1.2, 3.0, 3);
        let data = measured(&world, 30, 3, false);
        let ensemble = Ensemble::fit_banded(&data, 3, 1, 1e-3, &world.priors, 2).unwrap();
        let (objects, options) = (ensemble.layout().objects().len(), ensemble.layout().options().len());
        let p = vec![vec![1.0 / options as f64; options]; objects];
        let gradient = ensemble.gradient(&p).unwrap();
        let first = ensemble.members()[0].gradient(&p).unwrap();
        let mut mean = 0.0;
        for member in ensemble.members() {
            mean += member.gradient(&p).unwrap()[2][1] / 3.0;
        }
        let single = first[2][1];
        assert!((gradient[2][1] - mean).abs() <= 1e-15 * mean.abs().max(1.0));
        assert_ne!(gradient[2][1], single);
        assert!(Ensemble::member_fits(10, 0, 1).is_err());
        assert!(Ensemble::from_members(Vec::new()).is_err());
    }
}

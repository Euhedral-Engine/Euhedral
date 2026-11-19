//! Synthetic objects, priors and a known fidelity law, for the tests of the search and its dry runs: a space of
//! `layers` layers of four projection families whose true KL is a bent weighted sum of the options' prior errors.

use std::collections::BTreeMap;

use super::allocation::Allocation;
use super::error::MixedError;
use super::measured::Measured;
use super::option::QuantOption;
use super::priors::Priors;
use crate::java_random::Random;

/// The projection families of each layer, in the order a layer lists them.
pub const FAMILIES: [&str; 4] = ["gdn/query_key", "attention/output", "mlp/gate_up", "mlp/down"];

/// The options a synthetic space offers.
pub const OPTIONS: [QuantOption; 4] = [
    QuantOption::Nvfp4,
    QuantOption::Trellis2_5,
    QuantOption::Trellis3_0,
    QuantOption::Trellis4_0,
];

/// A synthetic space with its true law.
#[derive(Debug, Clone)]
pub struct Synthetic {
    pub layers: usize,
    pub bands: usize,
    /// The objects, layer by layer, a layer's families in [`FAMILIES`] order.
    pub objects: Vec<String>,
    pub priors: Priors,
    /// The true weight of each (family, band).
    pub weight: Vec<Vec<f64>>,
    /// The bend the true law applies.
    pub gamma: f64,
    pub scale: f64,
}

impl Synthetic {
    /// The space of `seed`: every draw comes from one seeded generator, the prior errors first, then the weights.
    pub fn new(layers: usize, bands: usize, gamma: f64, scale: f64, seed: i64) -> Synthetic {
        let mut random = Random::new(seed);
        let mut objects = Vec::new();
        let mut errors = BTreeMap::new();
        for layer in 0..layers {
            for family in FAMILIES {
                let name = format!("text/layers/{layer}/{family}");
                let base = 0.001 * (1.0 + random.next_double());
                let by_option = BTreeMap::from([
                    (QuantOption::Nvfp4, base * 3.0),
                    (QuantOption::Trellis2_5, base * 4.0),
                    (QuantOption::Trellis3_0, base * 2.0),
                    (QuantOption::Trellis4_0, base * 0.6),
                ]);
                objects.push(name.clone());
                errors.insert(name, by_option);
            }
        }
        let weight = (0..FAMILIES.len())
            .map(|_| (0..bands).map(|_| 0.5 + 1.5 * random.next_double()).collect())
            .collect();
        Synthetic {
            layers,
            bands,
            objects,
            priors: Priors::new(errors),
            weight,
            gamma,
            scale,
        }
    }

    /// The index of `object`'s family in [`FAMILIES`].
    pub fn family(&self, object: &str) -> Result<usize, MixedError> {
        object
            .strip_prefix("text/layers/")
            .and_then(|rest| rest.split_once('/'))
            .and_then(|(_, family)| FAMILIES.iter().position(|known| *known == family))
            .ok_or_else(|| MixedError::invalid(format!("{object} is not an object of a synthetic space")))
    }

    /// The band of `object`'s layer.
    pub fn band(&self, object: &str) -> Result<usize, MixedError> {
        let layer: usize = object
            .split('/')
            .nth(2)
            .and_then(|part| part.parse().ok())
            .ok_or_else(|| MixedError::invalid(format!("{object} is not an object of a synthetic space")))?;
        Ok(layer * self.bands / self.layers)
    }

    /// The true KL of `allocation`: `scale * (sum of weight * prior error) ^ gamma`.
    pub fn truth(&self, allocation: &Allocation) -> Result<f64, MixedError> {
        let mut x = 0.0;
        for object in &self.objects {
            let option = allocation
                .choices()
                .get(object)
                .ok_or_else(|| MixedError::invalid(format!("the allocation has no option for {object}")))?;
            x += self.weight[self.family(object)?][self.band(object)?] * self.priors.error(object, *option)?;
        }
        Ok(self.scale * x.powf(self.gamma))
    }

    /// An allocation that draws each object's option from `options`, in object order.
    pub fn random(&self, random: &mut Random, options: &[QuantOption]) -> Result<Allocation, MixedError> {
        let count = i32::try_from(options.len()).unwrap_or(0);
        let mut choices = BTreeMap::new();
        for object in &self.objects {
            let at = random
                .next_int_bounded(count)
                .ok_or_else(|| MixedError::invalid("no options to draw from"))?;
            choices.insert(object.clone(), options[at as usize]);
        }
        Ok(Allocation::new(choices))
    }

    /// `allocation` with its true KL as a measurement.
    pub fn measured(&self, allocation: Allocation) -> Result<Measured, MixedError> {
        let kl = self.truth(&allocation)?;
        Ok(Measured::new(allocation, kl))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_space_and_another_seed_a_different_one() {
        let a = Synthetic::new(4, 2, 1.2, 3.0, 7);
        let b = Synthetic::new(4, 2, 1.2, 3.0, 7);
        let c = Synthetic::new(4, 2, 1.2, 3.0, 8);
        assert_eq!(a.priors, b.priors);
        assert_eq!(a.weight, b.weight);
        assert_ne!(a.priors, c.priors);
        assert_eq!(a.objects.len(), 16);
    }

    #[test]
    fn families_and_bands_follow_the_object_names() {
        let space = Synthetic::new(16, 4, 1.2, 3.0, 7);
        assert_eq!(space.family("text/layers/3/mlp/gate_up").unwrap(), 2);
        assert_eq!(space.band("text/layers/0/mlp/down").unwrap(), 0);
        assert_eq!(space.band("text/layers/15/mlp/down").unwrap(), 3);
        assert_eq!(space.band("text/layers/8/mlp/down").unwrap(), 2);
        assert!(space.family("text/layers/3/other").is_err());
        assert!(space.band("mtp").is_err());
    }

    #[test]
    fn a_better_option_everywhere_lowers_the_true_kl() {
        let space = Synthetic::new(8, 2, 1.2, 3.0, 3);
        let uniform = |option| Allocation::new(space.objects.iter().map(|name| (name.clone(), option)).collect());
        let worst = space.truth(&uniform(QuantOption::Trellis2_5)).unwrap();
        let best = space.truth(&uniform(QuantOption::Trellis4_0)).unwrap();
        assert!(best < worst);
        assert!(space.truth(&Allocation::default()).is_err());
    }

    #[test]
    fn a_random_allocation_names_every_object_with_an_offered_option() {
        let space = Synthetic::new(8, 2, 1.2, 3.0, 3);
        let allocation = space.random(&mut Random::new(1), &OPTIONS[..2]).unwrap();
        assert_eq!(allocation.choices().len(), space.objects.len());
        assert!(
            allocation
                .choices()
                .values()
                .all(|option| OPTIONS[..2].contains(option))
        );
        assert!(space.random(&mut Random::new(1), &[]).is_err());
    }
}

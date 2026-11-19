//! The fidelity model: it predicts the KL of an allocation from the per-object prior errors and the candidates measured
//! so far.
//!
//! Each object `i` that takes option `o` contributes its prior error `e[i][o]`, divided by the mean prior error so that
//! the features are of order one. With `E_i` that contribution, or its expectation under the probabilities `p[i]` of a
//! soft assignment, the model is
//!
//! ```text
//! x  = sum_{f,b} w[f,b] * phi[f,b]  +  sum_{f<g} v[f,g] * psi[f,g]  +  sum_f u[f] * chi[f]
//! kl = s * x^gamma
//! ```
//!
//! with
//! - `phi[f,b]`, the sum of `E_i` over the objects of family `f` in band `b` (a family is the name after the layer
//!   index, `mlp/down`; the band of layer `l` of `L` is `l * bands / L`);
//! - `psi[f,g]`, the sum over layers of (the layer's `E` summed over family `f`) times (the same over family `g`): the
//!   pairs of families within a layer;
//! - `chi[f]`, the sum over bands `b` of `phi[f,b] * phi[f,b+1]`: adjacent bands of one family.
//!
//! `gamma` is the monotone bend and `s` its scale. The weights are fitted by ridge regression of `kl^(1/gamma)` on the
//! features, each column scaled to unit RMS, every weight non-negative (an error anywhere can only add KL), shrinking
//! the linear weights towards one common weight and the pair weights towards zero. `gamma` is the value on a grid whose
//! fit has the least RMS error of `log kl`, and `s` the geometric-mean correction of that fit. The prediction is
//! differentiable in `p`: [`FidelityModel::gradient`] is exact.

use std::collections::HashMap;

use super::allocation::Allocation;
use super::error::MixedError;
use super::option::QuantOption;
use super::priors::Priors;
use super::search_state::Measured;
use super::space::sorted_by_layer;
use crate::java_random::Random;

/// The default number of bands per family.
pub const DEFAULT_BANDS: usize = 8;

const FLOOR: f64 = 1e-12;
const MINIMUM_KL: f64 = 1e-9;
/// The coordinate-descent sweeps of the ridge solve at most, and the relative change that ends it sooner.
const SOLVE_SWEEPS: usize = 2000;
const SOLVE_TOLERANCE: f64 = 1e-10;
const GAMMAS: [f64; 14] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0];

/// The objects of the model and how they group.
#[derive(Debug, Clone)]
struct Layout {
    objects: Vec<String>,
    family: Vec<usize>,
    band: Vec<usize>,
    layer: Vec<usize>,
    families: usize,
    bands: usize,
    layers: usize,
    family_names: Vec<String>,
    /// The index of the first pair weight and of the first adjacency weight, and the number of weights.
    pair_base: usize,
    adjacent_base: usize,
    columns: usize,
}

impl Layout {
    fn new(objects: Vec<String>, bands: usize) -> Result<Layout, MixedError> {
        let n = objects.len();
        let mut family = Vec::with_capacity(n);
        let mut layer = Vec::with_capacity(n);
        let mut family_index: HashMap<String, usize> = HashMap::new();
        let mut family_names: Vec<String> = Vec::new();
        for name in &objects {
            let mut parts = name.splitn(4, '/');
            let (head, kind, number, rest) = (parts.next(), parts.next(), parts.next(), parts.next());
            let (Some("text"), Some("layers"), Some(number), Some(rest)) = (head, kind, number, rest) else {
                return Err(MixedError::invalid(format!(
                    "{name} is not an object of a decoder layer"
                )));
            };
            let number: usize = number
                .parse()
                .map_err(|_| MixedError::invalid(format!("{name} is not an object of a decoder layer")))?;
            layer.push(number);
            let next = family_names.len();
            let index = *family_index.entry(rest.to_owned()).or_insert(next);
            if index == next {
                family_names.push(rest.to_owned());
            }
            family.push(index);
        }
        let layers = layer.iter().max().map_or(0, |last| last + 1);
        let families = family_names.len();
        let band = layer.iter().map(|&at| at * bands / layers).collect();
        let pair_base = families * bands;
        let adjacent_base = pair_base + families * families;
        Ok(Layout {
            objects,
            family,
            band,
            layer,
            families,
            bands,
            layers,
            family_names,
            pair_base,
            adjacent_base,
            columns: adjacent_base + families,
        })
    }

    fn pair(&self, f: usize, g: usize) -> usize {
        self.pair_base + f.min(g) * self.families + f.max(g)
    }
}

/// The sums the features and the gradient are made of, for the expectations `expected` (per object).
struct Features {
    /// By `family * bands + band`.
    phi: Vec<f64>,
    /// By `[layer][family]`.
    layer_family: Vec<Vec<f64>>,
}

impl Features {
    fn new(layout: &Layout, expected: &[f64]) -> Features {
        let mut phi = vec![0.0; layout.families * layout.bands];
        let mut layer_family = vec![vec![0.0; layout.families]; layout.layers];
        for (i, &value) in expected.iter().enumerate() {
            phi[layout.family[i] * layout.bands + layout.band[i]] += value;
            layer_family[layout.layer[i]][layout.family[i]] += value;
        }
        Features { phi, layer_family }
    }
}

/// The feature vector, columns as the layout numbers them.
fn features(layout: &Layout, expected: &[f64]) -> Vec<f64> {
    let parts = Features::new(layout, expected);
    let mut x = vec![0.0; layout.columns];
    x[..parts.phi.len()].copy_from_slice(&parts.phi);
    for row in &parts.layer_family {
        for f in 0..layout.families {
            for g in f + 1..layout.families {
                x[layout.pair(f, g)] += row[f] * row[g];
            }
        }
    }
    for f in 0..layout.families {
        for b in 0..layout.bands.saturating_sub(1) {
            x[layout.adjacent_base + f] += parts.phi[f * layout.bands + b] * parts.phi[f * layout.bands + b + 1];
        }
    }
    x
}

/// A fitted fidelity model.
#[derive(Debug, Clone)]
pub struct FidelityModel {
    layout: Layout,
    options: Vec<QuantOption>,
    /// The normalized prior errors, `error[object][option]`.
    error: Vec<Vec<f64>>,
    mean_error: f64,
    weights: Vec<f64>,
    gamma: f64,
    scale: f64,
}

impl FidelityModel {
    /// The model fitted to `data` with ridge `ridge`, over `bands` bands per family, with no perturbation.
    pub fn fit(data: &[Measured], ridge: f64, priors: &Priors, bands: usize) -> Result<FidelityModel, MixedError> {
        FidelityModel::fit_perturbed(data, ridge, priors, bands, 0, 0.0)
    }

    /// The model fitted to `data` with ridge `ridge`, over `bands` bands per family. A `perturb` above zero draws, from
    /// `seed`, the point the ridge shrinks towards: each linear weight is scaled by `exp(perturb * N(0,1))` and each
    /// pair weight is given a random contribution of order `perturb`, so that models fitted from different seeds
    /// differ where the data does not decide.
    ///
    /// Refuses: no band, a ridge that is not finite, a priors table without an object of a decoder layer or without an option every object has, a
    /// mean prior error that is not positive, an allocation in `data` that lacks an object or names an option the
    /// priors lack, a KL that is not finite, and data to which no bend gives a finite fit.
    pub fn fit_perturbed(
        data: &[Measured],
        ridge: f64,
        priors: &Priors,
        bands: usize,
        seed: i64,
        perturb: f64,
    ) -> Result<FidelityModel, MixedError> {
        if bands < 1 {
            return Err(MixedError::invalid("the model needs at least one band"));
        }
        if !ridge.is_finite() {
            return Err(MixedError::invalid("the ridge is not a finite number"));
        }
        let names: Vec<String> = priors.objects().keys().cloned().collect();
        let layout = Layout::new(sorted_by_layer(&names)?, bands)?;
        let options = priors.options();
        let mut error = vec![vec![0.0; options.len()]; names.len()];
        let mut sum = 0.0;
        let mut count = 0usize;
        for (i, row) in error.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = priors.error(&layout.objects[i], options[k])?;
                sum += *slot;
                count += 1;
            }
        }
        let mean = sum / count as f64;
        if !(mean.is_finite() && mean > 0.0) {
            return Err(MixedError::invalid(format!(
                "the priors' mean error is {mean}, not positive"
            )));
        }
        for row in &mut error {
            for value in row {
                *value /= mean;
            }
        }
        let shell = FidelityModel {
            layout: layout.clone(),
            options: options.clone(),
            error: error.clone(),
            mean_error: mean,
            weights: vec![0.0; layout.columns],
            gamma: 1.0,
            scale: 1.0,
        };
        if data.is_empty() {
            let mut weights = vec![0.0; layout.columns];
            weights[..layout.pair_base].fill(1.0);
            return Ok(FidelityModel { weights, ..shell });
        }

        let rows = data.len();
        let mut x = Vec::with_capacity(rows);
        let mut kl = Vec::with_capacity(rows);
        for measured in data {
            if !measured.kl.is_finite() {
                return Err(MixedError::invalid("a measured KL is not a finite number"));
            }
            x.push(features(&layout, &shell.expected_of(&measured.allocation)?));
            kl.push(measured.kl.max(MINIMUM_KL));
        }
        let columns = layout.columns;
        let mut rms = vec![0.0; columns];
        for row in &x {
            for (j, slot) in rms.iter_mut().enumerate() {
                *slot += row[j] * row[j];
            }
        }
        for slot in &mut rms {
            *slot = (*slot / rows as f64).sqrt();
            if slot.is_nan() || *slot <= 1e-12 {
                *slot = 1.0;
            }
        }
        let scaled: Vec<Vec<f64>> = x
            .iter()
            .map(|row| row.iter().zip(&rms).map(|(value, size)| value / size).collect())
            .collect();

        let mut random = Random::new(seed);
        let noise: Vec<f64> = (0..columns)
            .map(|_| if perturb > 0.0 { random.next_gaussian() } else { 0.0 })
            .collect();

        let mut best: Option<FidelityModel> = None;
        let mut best_score = f64::INFINITY;
        for gamma in GAMMAS {
            let target: Vec<f64> = kl.iter().map(|value| value.powf(1.0 / gamma)).collect();
            let mut target_squares = 0.0;
            for value in &target {
                target_squares += value * value;
            }
            let target_rms = (target_squares / rows as f64).sqrt();
            // The common linear weight: least squares of the target on the sum of the linear columns.
            let (mut dot, mut squares) = (0.0, 0.0);
            for r in 0..rows {
                let total = scaled[r][..layout.pair_base].iter().fold(0.0, |sum, value| sum + value);
                dot += total * target[r];
                squares += total * total;
            }
            let common = if squares > 0.0 { dot / squares } else { 0.0 };
            let prior: Vec<f64> = (0..columns)
                .map(|j| {
                    if j < layout.pair_base {
                        common * (perturb * noise[j]).exp()
                    } else {
                        perturb * target_rms / (columns as f64).sqrt() * noise[j].abs()
                    }
                })
                .collect();
            let theta = ridge_solve(&scaled, &target, ridge, &prior);
            let weights: Vec<f64> = theta.iter().zip(&rms).map(|(value, size)| value / size).collect();
            let mut logs = 0.0;
            let mut residual = vec![0.0; rows];
            for r in 0..rows {
                let mut value = 0.0;
                for j in 0..weights.len() {
                    value += weights[j] * x[r][j];
                }
                residual[r] = kl[r].ln() - gamma * value.max(FLOOR).ln();
                logs += residual[r];
            }
            let log_scale = logs / rows as f64;
            let mut squares_of_error = 0.0;
            for r in &residual {
                squares_of_error += (r - log_scale) * (r - log_scale);
            }
            let score = (squares_of_error / rows as f64).sqrt();
            if score < best_score {
                best_score = score;
                best = Some(FidelityModel {
                    layout: layout.clone(),
                    options: options.clone(),
                    error: error.clone(),
                    mean_error: mean,
                    weights,
                    gamma,
                    scale: log_scale.exp(),
                });
            }
        }
        best.ok_or_else(|| MixedError::invalid("no bend gave a finite fit of the measured KL"))
    }

    /// The objects the model predicts over, in the order of the rows of `p` in [`predict_soft`](Self::predict_soft).
    pub fn objects(&self) -> &[String] {
        &self.layout.objects
    }

    /// The families (the names after the layer index), in the order of the weights' first index.
    pub fn families(&self) -> &[String] {
        &self.layout.family_names
    }

    /// The options, in the order of the columns of `p`.
    pub fn options(&self) -> &[QuantOption] {
        &self.options
    }

    /// The bend of the fitted law.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The scale of the fitted law.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The weight of the family (by the order its first object appears) in the band, in the model's own units.
    pub fn weight(&self, family: usize, band: usize) -> f64 {
        self.weights[family * self.layout.bands + band]
    }

    /// What the family-band weight multiplies a prior error by in the linear law `kl = c * (sum w * e)^gamma` (the law
    /// the model reduces to without pair terms), with `c` taken as one.
    pub fn effective_weight(&self, family: usize, band: usize) -> f64 {
        self.scale.powf(1.0 / self.gamma) * self.weight(family, band) / self.mean_error
    }

    fn expected_of(&self, allocation: &Allocation) -> Result<Vec<f64>, MixedError> {
        let mut expected = Vec::with_capacity(self.layout.objects.len());
        for (i, object) in self.layout.objects.iter().enumerate() {
            let option = allocation
                .choices()
                .get(object)
                .ok_or_else(|| MixedError::invalid(format!("the allocation has no option for {object}")))?;
            let k = self
                .options
                .iter()
                .position(|known| known == option)
                .ok_or_else(|| MixedError::invalid(format!("{} is not an option of the model", option.key())))?;
            expected.push(self.error[i][k]);
        }
        Ok(expected)
    }

    fn check_soft(&self, p: &[Vec<f64>]) -> Result<(), MixedError> {
        if p.len() != self.layout.objects.len() || p.iter().any(|row| row.len() != self.options.len()) {
            return Err(MixedError::invalid(format!(
                "a soft assignment has a row of {} probabilities for each of {} objects",
                self.options.len(),
                self.layout.objects.len()
            )));
        }
        Ok(())
    }

    fn expected_of_soft(&self, p: &[Vec<f64>]) -> Vec<f64> {
        let mut expected = vec![0.0; self.layout.objects.len()];
        for (i, slot) in expected.iter_mut().enumerate() {
            for (probability, error) in p[i].iter().zip(&self.error[i]) {
                *slot += probability * error;
            }
        }
        expected
    }

    fn linear(&self, expected: &[f64]) -> f64 {
        let x = features(&self.layout, expected);
        let mut value = 0.0;
        for (weight, feature) in self.weights.iter().zip(&x) {
            value += weight * feature;
        }
        value
    }

    /// The predicted KL of `allocation`, which names an option for every object of the model.
    pub fn predict(&self, allocation: &Allocation) -> Result<f64, MixedError> {
        Ok(self.curve(self.linear(&self.expected_of(allocation)?)))
    }

    /// The predicted KL of the soft assignment `p[object][option]`, each row a probability vector.
    pub fn predict_soft(&self, p: &[Vec<f64>]) -> Result<f64, MixedError> {
        self.check_soft(p)?;
        Ok(self.curve(self.linear(&self.expected_of_soft(p))))
    }

    fn curve(&self, x: f64) -> f64 {
        self.scale * x.max(FLOOR).powf(self.gamma)
    }

    /// The derivative of [`predict_soft`](Self::predict_soft) with respect to every `p[object][option]`.
    pub fn gradient(&self, p: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MixedError> {
        self.check_soft(p)?;
        let layout = &self.layout;
        let expected = self.expected_of_soft(p);
        let parts = Features::new(layout, &expected);
        let x = self.linear(&expected);
        let outer = if x > FLOOR {
            self.scale * self.gamma * x.powf(self.gamma - 1.0)
        } else {
            0.0
        };
        let mut gradient = vec![vec![0.0; self.options.len()]; expected.len()];
        for (i, row) in gradient.iter_mut().enumerate() {
            let (f, b) = (layout.family[i], layout.band[i]);
            let mut slope = self.weights[f * layout.bands + b];
            for g in 0..layout.families {
                if g != f {
                    slope += self.weights[layout.pair(f, g)] * parts.layer_family[layout.layer[i]][g];
                }
            }
            let mut adjacent = 0.0;
            if b > 0 {
                adjacent += parts.phi[f * layout.bands + b - 1];
            }
            if b + 1 < layout.bands {
                adjacent += parts.phi[f * layout.bands + b + 1];
            }
            slope += self.weights[layout.adjacent_base + f] * adjacent;
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = outer * slope * self.error[i][k];
            }
        }
        Ok(gradient)
    }
}

/// Minimizes `|X theta - t|^2 + ridge' |theta - prior|^2` over `theta >= 0` (`ridge'` is `ridge` times the rows), by
/// coordinate descent on the normal equations, each coordinate clipped at zero. The features are sums of non-negative
/// prior errors, so non-negative weights keep every prediction's linear part non-negative: an unconstrained fit could
/// give an allocation a part at or below zero, a predicted KL of nothing with no gradient, which a search would then
/// seek out.
fn ridge_solve(x: &[Vec<f64>], t: &[f64], ridge: f64, prior: &[f64]) -> Vec<f64> {
    let n = prior.len();
    let rows = x.len();
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for r in 0..rows {
        let row = &x[r];
        for i in 0..n {
            if row[i] == 0.0 {
                continue;
            }
            b[i] += row[i] * t[r];
            for j in 0..n {
                a[i][j] += row[i] * row[j];
            }
        }
    }
    let lambda = ridge.max(1e-12) * rows.max(1) as f64;
    for i in 0..n {
        a[i][i] += lambda;
        b[i] += lambda * prior[i];
    }
    let mut theta: Vec<f64> = prior.iter().map(|value| value.max(0.0)).collect();
    for _ in 0..SOLVE_SWEEPS {
        let (mut change, mut size) = (0.0f64, 0.0f64);
        for i in 0..n {
            let mut rest = b[i];
            for j in 0..n {
                if j != i {
                    rest -= a[i][j] * theta[j];
                }
            }
            let next = (rest / a[i][i]).max(0.0);
            change = change.max((next - theta[i]).abs());
            size = size.max(next.abs());
            theta[i] = next;
        }
        if change <= SOLVE_TOLERANCE * size.max(1e-300) {
            break;
        }
    }
    theta
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::mixed::synthetic::{FAMILIES, OPTIONS, Synthetic, measurement};

    fn data(world: &Synthetic, count: usize, seed: i64) -> Vec<Measured> {
        let mut random = Random::new(seed);
        (0..count)
            .map(|_| {
                let allocation = world.random(&mut random, &OPTIONS).unwrap();
                world.measured(allocation, 1).unwrap()
            })
            .collect()
    }

    fn uniform(world: &Synthetic, option: QuantOption) -> Allocation {
        Allocation::new(world.objects.iter().map(|name| (name.clone(), option)).collect())
    }

    #[test]
    fn the_law_of_synthetic_data_is_recovered() {
        let world = Synthetic::new(16, 4, 1.2, 3.0, 7);
        let model = FidelityModel::fit(&data(&world, 400, 1), 1e-9, &world.priors, 4).unwrap();
        // Noise-free data lands on the grid point of the true bend.
        assert_eq!(model.gamma(), 1.2);
        let mut random = Random::new(99);
        for i in 0..40 {
            let allocation = world.random(&mut random, &OPTIONS).unwrap();
            let truth = world.truth(&allocation).unwrap();
            let predicted = model.predict(&allocation).unwrap();
            assert!(
                (predicted - truth).abs() <= 1e-4 * truth,
                "prediction {i}: {predicted} for {truth}"
            );
        }
    }

    #[test]
    fn the_weights_of_a_purely_linear_law_are_recovered() {
        let world = Synthetic::new(16, 4, 1.0, 1.0, 11);
        let model = FidelityModel::fit(&data(&world, 400, 2), 1e-9, &world.priors, 4).unwrap();
        assert_eq!(model.gamma(), 1.0);
        for (family, name) in FAMILIES.iter().enumerate() {
            let at = model.families().iter().position(|known| known == name).unwrap();
            for band in 0..4 {
                let estimated = model.effective_weight(at, band);
                let truth = world.weight[family][band];
                assert!(
                    (estimated - truth).abs() <= 1e-3 * truth,
                    "{family}/{band}: {estimated} for {truth}"
                );
            }
        }
    }

    fn soft(objects: usize, options: usize, seed: i64) -> Vec<Vec<f64>> {
        let mut random = Random::new(seed);
        (0..objects)
            .map(|_| {
                let row: Vec<f64> = (0..options).map(|_| 0.1 + random.next_double()).collect();
                let sum: f64 = row.iter().sum();
                row.iter().map(|value| value / sum).collect()
            })
            .collect()
    }

    #[test]
    fn the_soft_gradient_matches_finite_differences() {
        let world = Synthetic::new(8, 4, 1.3, 2.0, 5);
        // A perturbed prior gives the pair and adjacent-band weights values, so every term is differentiated.
        let model = FidelityModel::fit_perturbed(&data(&world, 120, 3), 1e-3, &world.priors, 4, 11, 0.5).unwrap();
        let mut p = soft(world.objects.len(), model.options().len(), 4);
        let gradient = model.gradient(&p).unwrap();
        let step = 1e-6;
        for i in (0..p.len()).step_by(3) {
            for k in 0..model.options().len() {
                let kept = p[i][k];
                p[i][k] = kept + step;
                let up = model.predict_soft(&p).unwrap();
                p[i][k] = kept - step;
                let down = model.predict_soft(&p).unwrap();
                p[i][k] = kept;
                let numeric = (up - down) / (2.0 * step);
                let allowed = 1e-6 * numeric.abs().max(1.0) + 1e-9;
                assert!(
                    (numeric - gradient[i][k]).abs() <= allowed,
                    "{i}/{k}: {numeric} against {}",
                    gradient[i][k]
                );
            }
        }
    }

    #[test]
    fn a_hard_allocation_is_the_soft_one_of_its_indicators() {
        let world = Synthetic::new(8, 4, 1.1, 2.0, 5);
        let model = FidelityModel::fit(&data(&world, 100, 3), 1e-3, &world.priors, 4).unwrap();
        let allocation = world.random(&mut Random::new(8), &OPTIONS).unwrap();
        let mut p = vec![vec![0.0; model.options().len()]; model.objects().len()];
        for (row, object) in p.iter_mut().zip(model.objects()) {
            let option = allocation.choices()[object];
            row[model.options().iter().position(|known| *known == option).unwrap()] = 1.0;
        }
        let hard = model.predict(&allocation).unwrap();
        assert!((hard - model.predict_soft(&p).unwrap()).abs() <= 1e-12);
    }

    #[test]
    fn predictions_grow_with_the_prior_errors() {
        let world = Synthetic::new(8, 4, 1.1, 2.0, 5);
        let model = FidelityModel::fit(&data(&world, 150, 3), 1e-3, &world.priors, 4).unwrap();
        let best = model.predict(&uniform(&world, QuantOption::Trellis4_0)).unwrap();
        let worst = model.predict(&uniform(&world, QuantOption::Trellis2_5)).unwrap();
        assert!(best < worst);
    }

    #[test]
    fn few_noisy_measurements_never_give_an_allocation_a_near_zero_prediction() {
        // An unconstrained fit gives some allocations a non-positive linear part (a predicted KL of nothing, with no
        // gradient), which an optimizer would then seek out.
        let world = Synthetic::new(16, 4, 1.0, 1.0, 13);
        let mut random = Random::new(5);
        let measured: Vec<Measured> = (0..12)
            .map(|_| {
                let allocation = world.random(&mut random, &OPTIONS).unwrap();
                let noisy = world.truth(&allocation).unwrap() * (0.5 * random.next_gaussian()).exp();
                measurement(allocation, noisy, 1, "noisy")
            })
            .collect();
        let model = FidelityModel::fit(&measured, 1e-6, &world.priors, 4).unwrap();
        for family in 0..FAMILIES.len() {
            for band in 0..4 {
                assert!(model.weight(family, band) >= 0.0, "weight {family}/{band}");
            }
        }
        let least = measured.iter().map(|m| m.kl).fold(f64::INFINITY, f64::min);
        let mut other = Random::new(17);
        for i in 0..200 {
            let predicted = model.predict(&world.random(&mut other, &OPTIONS).unwrap()).unwrap();
            assert!(predicted > 1e-3 * least, "prediction {i} = {predicted}");
        }
    }

    #[test]
    fn without_measurements_every_object_weighs_the_same() {
        let world = Synthetic::new(4, 2, 1.1, 2.0, 5);
        let model = FidelityModel::fit(&[], 1e-3, &world.priors, 2).unwrap();
        assert_eq!((model.gamma(), model.scale()), (1.0, 1.0));
        for family in 0..FAMILIES.len() {
            assert_eq!(model.weight(family, 1), 1.0);
        }
    }

    #[test]
    fn what_the_model_cannot_take_is_refused_with_the_reason() {
        let world = Synthetic::new(4, 2, 1.1, 2.0, 5);
        let measured = data(&world, 20, 1);
        let fit = |data: &[Measured], bands| FidelityModel::fit(data, 1e-3, &world.priors, bands);
        assert!(fit(&measured, 0).unwrap_err().to_string().contains("band"));
        for ridge in [f64::NAN, f64::INFINITY] {
            let refused = FidelityModel::fit(&measured, ridge, &world.priors, 2);
            assert!(refused.unwrap_err().to_string().contains("ridge"));
        }
        let mut infinite = measured.clone();
        infinite[3].kl = f64::NAN;
        assert!(fit(&infinite, 2).unwrap_err().to_string().contains("finite"));
        let mut partial = BTreeMap::new();
        partial.insert(world.objects[0].clone(), QuantOption::Nvfp4);
        let short = vec![measurement(Allocation::new(partial), 0.1, 1, "partial")];
        assert!(fit(&short, 2).unwrap_err().to_string().contains(&world.objects[1]));
        let elsewhere = Priors::parse(r#"{"objects": {"mtp/fc": {"nvfp4": 0.1}}}"#).unwrap();
        let error = FidelityModel::fit(&[], 1e-3, &elsewhere, 2).unwrap_err();
        assert!(error.to_string().contains("mtp/fc"));
        let model = fit(&measured, 2).unwrap();
        assert!(model.predict_soft(&[vec![1.0]]).is_err());
        assert!(model.gradient(&[]).is_err());
        let two_options: Vec<QuantOption> = world.priors.options().into_iter().take(2).collect();
        let narrow = world.priors.restrict(&world.objects, &two_options).unwrap();
        let narrowed = FidelityModel::fit(&[], 1e-3, &narrow, 2).unwrap();
        let outside = uniform(&world, QuantOption::Trellis4_0);
        assert!(
            narrowed
                .predict(&outside)
                .unwrap_err()
                .to_string()
                .contains("trellis-4.0")
        );
    }
}

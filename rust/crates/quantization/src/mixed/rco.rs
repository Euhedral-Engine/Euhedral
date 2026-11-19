//! Riemannian Constrained Optimization (Helcig and Alistarh, arXiv:2605.00649), applied to choosing one option per object
//! under a byte budget.
//!
//! Every object `i` has logits `alpha[i][k]` over its options and probabilities `p = softmax(alpha)`. The expected bytes
//! `C(alpha) = sum_i sum_k p[i][k] c[i][k]` are held on the budget by working on the manifold `C = budget`:
//!
//! 1. the normal of the manifold is `(grad C)[i][k] = p[i][k] * (c[i][k] - E_{p_i}[c_i])`;
//! 2. a gradient of the objective is estimated by Gumbel straight-through sampling: for each of `samples` draws of
//!    Gumbel noise `G`, the forward pass is the knapsack optimum (exact within the byte budget) of the scores
//!    `(alpha + G) / tau`, the objective's gradient is taken at that hard assignment, and the backward pass goes
//!    through `softmax((alpha + G) / tau)`; `tau` falls exponentially from 1.0 to 0.01 over the steps;
//! 3. the gradient is projected onto the tangent space (the plane orthogonal to the normal);
//! 4. Adam turns the projected gradient into a direction, which is projected onto the tangent space again;
//! 5. `alpha` takes a step of `lr` against the direction;
//! 6. the retraction puts `alpha` back on the manifold by adding `t` times the cost vector, `t` found by bisection on
//!    `[-50, 50]` (the expected bytes grow with `t`);
//! 7. Adam's first moment is transported to the new tangent space by projecting it there.
//!
//! The final assignment is the knapsack optimum of `log p`, as in the paper, unless a hard assignment one of the samples
//! drew scored better on the objective; that one is returned instead (on small separable problems with known optima it
//! reached the optimum where the decoded `log p` did not).
//!
//! A [`Run`] is taken a little at a time. The samples of one step only read the logits and each makes its own
//! [`Sample`], so a caller may draw them as frames of its own, in any order, and end the step with them in index order.

use std::mem::take;
use std::sync::Arc;

use super::error::MixedError;
use super::knapsack;
use crate::java_random::Random;

/// The default number of steps.
pub const DEFAULT_STEPS: usize = 200;

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPSILON: f64 = 1e-8;
const TAU_START: f64 = 1.0;
const TAU_END: f64 = 0.01;
const RETRACTION_RANGE: f64 = 50.0;
const RETRACTION_TOLERANCE: f64 = 1e-8;
const INITIAL_SPREAD: f64 = 0.5;
/// The multiplier that spreads the noise seeds of the draws of a run.
const NOISE_STRIDE: i64 = 0x9E37_79B9_7F4A_7C15_u64 as i64;

/// What is minimized: a function of the soft assignment `p[object][option]` and its gradient.
pub trait Objective {
    /// The objective at `p`.
    fn value(&self, p: &[Vec<f64>]) -> f64;

    /// The derivative of [`value`](Self::value) with respect to every `p[object][option]`.
    fn gradient(&self, p: &[Vec<f64>]) -> Vec<Vec<f64>>;
}

impl<T: Objective + ?Sized> Objective for &T {
    fn value(&self, p: &[Vec<f64>]) -> f64 {
        (**self).value(p)
    }

    fn gradient(&self, p: &[Vec<f64>]) -> Vec<Vec<f64>> {
        (**self).gradient(p)
    }
}

impl<T: Objective + ?Sized> Objective for Arc<T> {
    fn value(&self, p: &[Vec<f64>]) -> f64 {
        (**self).value(p)
    }

    fn gradient(&self, p: &[Vec<f64>]) -> Vec<Vec<f64>> {
        (**self).gradient(p)
    }
}

/// `steps` of Adam with learning rate `lr`, `samples` Gumbel draws per step, and `restarts` runs from different
/// starting logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub steps: usize,
    pub lr: f64,
    pub samples: usize,
    pub restarts: usize,
}

impl Settings {
    /// Settings; refuses a count of zero and a learning rate that is not positive.
    pub fn new(steps: usize, lr: f64, samples: usize, restarts: usize) -> Result<Settings, MixedError> {
        let settings = Settings {
            steps,
            lr,
            samples,
            restarts,
        };
        settings.check()?;
        Ok(settings)
    }

    fn check(&self) -> Result<(), MixedError> {
        if self.steps < 1 || self.samples < 1 || self.restarts < 1 || self.lr.is_nan() || self.lr <= 0.0 {
            return Err(MixedError::invalid(
                "steps, samples and restarts must be positive, and lr too",
            ));
        }
        Ok(())
    }
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            steps: DEFAULT_STEPS,
            lr: 0.1,
            samples: 8,
            restarts: 4,
        }
    }
}

/// The assignment of a run, the probabilities it ended on, the objective at the assignment, and the largest relative
/// distance of the expected bytes from the manifold after any step.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub choice: Vec<usize>,
    pub p: Vec<Vec<f64>>,
    pub value: f64,
    pub max_manifold_error: f64,
}

/// Every restart of `settings`, best first.
pub fn solve<O: Objective>(
    objective: O,
    cost: &[Vec<i64>],
    budget: i64,
    unit: i64,
    settings: Settings,
    seed: i64,
) -> Result<Vec<Outcome>, MixedError> {
    settings.check()?;
    let mut results = Vec::with_capacity(settings.restarts);
    for restart in 0..settings.restarts {
        results.push(run(&objective, cost, budget, unit, settings, seed, restart)?);
    }
    results.sort_by(|a, b| a.value.total_cmp(&b.value));
    Ok(results)
}

/// One run, restart number `restart`. `unit` is the knapsack's step in bytes. Refuses a budget below the cheapest
/// options ([`MixedError::BudgetTooSmall`]).
pub fn run<O: Objective>(
    objective: O,
    cost: &[Vec<i64>],
    budget: i64,
    unit: i64,
    settings: Settings,
    seed: i64,
    restart: usize,
) -> Result<Outcome, MixedError> {
    let mut run = Run::new(objective, cost, budget, unit, settings, seed, restart)?;
    while !run.advance()? {}
    run.result()
}

/// One draw of a step: the knapsack optimum of the noisy scores, the objective there, and its gradient carried back
/// through the softmax.
#[derive(Debug, Clone)]
pub struct Sample {
    value: f64,
    choice: Vec<usize>,
    gradient: Vec<Vec<f64>>,
}

/// A run taken a little at a time: each [`advance`](Run::advance) draws one Gumbel sample, or ends a step once its
/// samples are drawn.
#[derive(Debug)]
pub struct Run<O> {
    objective: O,
    cost: Vec<Vec<i64>>,
    budget: i64,
    unit: i64,
    settings: Settings,
    seed: i64,
    c: Vec<Vec<f64>>,
    direction: Vec<Vec<f64>>,
    alpha: Vec<Vec<f64>>,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    target: f64,
    worst: f64,
    best_value: f64,
    best_choice: Option<Vec<usize>>,
    step: usize,
    pending: Vec<Sample>,
}

impl<O: Objective> Run<O> {
    /// The run `restart` of `seed`. Refuses settings with a zero count, no object, an object without an option, a
    /// negative cost, and a budget below the cheapest options ([`MixedError::BudgetTooSmall`]).
    pub fn new(
        objective: O,
        cost: &[Vec<i64>],
        budget: i64,
        unit: i64,
        settings: Settings,
        seed: i64,
        restart: usize,
    ) -> Result<Run<O>, MixedError> {
        settings.check()?;
        let groups = cost.len();
        if groups == 0 || cost.iter().any(Vec::is_empty) {
            return Err(MixedError::invalid("every object needs an option"));
        }
        if cost.iter().flatten().any(|&bytes| bytes < 0) {
            return Err(MixedError::invalid("a cost is negative"));
        }
        let seed = seed
            .wrapping_mul(1_000_003)
            .wrapping_add(i64::try_from(restart).unwrap_or(i64::MAX));
        let mut random = Random::new(seed);
        let c: Vec<Vec<f64>> = cost
            .iter()
            .map(|row| row.iter().map(|&bytes| bytes as f64).collect())
            .collect();
        let (mut minimum, mut maximum, mut ranges) = (0.0, 0.0, 0.0);
        let mut low = vec![0.0; groups];
        for (i, row) in c.iter().enumerate() {
            let lo = row.iter().copied().fold(f64::MAX, f64::min);
            let hi = row.iter().copied().fold(0.0, f64::max);
            low[i] = lo;
            minimum += lo;
            maximum += hi;
            ranges += hi - lo;
        }
        if (budget as f64) < minimum {
            return Err(MixedError::BudgetTooSmall(budget));
        }
        // The manifold must lie strictly between the cheapest and the dearest expectation to be reachable.
        let margin = 1e-4 * (maximum - minimum);
        let target = (budget as f64).max(minimum + margin).min(maximum - margin);
        let mean_range = (ranges / groups as f64).max(1.0);
        let direction: Vec<Vec<f64>> = c
            .iter()
            .zip(&low)
            .map(|(row, lo)| row.iter().map(|value| (value - lo) / mean_range).collect())
            .collect();
        let mut alpha: Vec<Vec<f64>> = c.iter().map(|row| vec![0.0; row.len()]).collect();
        for row in &mut alpha {
            for slot in row {
                *slot = INITIAL_SPREAD * random.next_gaussian();
            }
        }
        retract(&mut alpha, &c, &direction, target);
        let worst = manifold_error(&alpha, &c, target);
        Ok(Run {
            objective,
            cost: cost.to_vec(),
            budget,
            unit,
            settings,
            seed,
            m: zeros(&c),
            v: zeros(&c),
            c,
            direction,
            alpha,
            target,
            worst,
            best_value: f64::INFINITY,
            best_choice: None,
            step: 1,
            pending: Vec::new(),
        })
    }

    fn tau(&self) -> f64 {
        if self.settings.steps == 1 {
            TAU_START
        } else {
            TAU_START * (TAU_END / TAU_START).powf((self.step as f64 - 1.0) / (self.settings.steps as f64 - 1.0))
        }
    }

    /// The samples of a step.
    pub fn samples(&self) -> usize {
        self.settings.samples
    }

    /// True once every step is done.
    pub fn done(&self) -> bool {
        self.step > self.settings.steps
    }

    /// Draws the step's next sample, or, when every sample of the step is drawn, updates the logits. True once every
    /// step is done. The same as drawing every [`sample`](Self::sample) and then calling [`update`](Self::update).
    pub fn advance(&mut self) -> Result<bool, MixedError> {
        if self.done() {
            return Ok(true);
        }
        if self.pending.len() < self.settings.samples {
            let drawn = self.sample(self.pending.len())?;
            self.pending.push(drawn);
            return Ok(false);
        }
        let drawn = take(&mut self.pending);
        self.update(&drawn)
    }

    /// Draws sample `index` of the current step: Gumbel noise from its own stream (the run's seed, the step and the
    /// index), the knapsack forward, and the objective's gradient through the softmax backward. The samples of one step
    /// only read the logits.
    pub fn sample(&self, index: usize) -> Result<Sample, MixedError> {
        let tau = self.tau();
        let draw = (self.step * self.settings.samples + index + 1) as i64;
        let mut noise = Random::new(self.seed ^ NOISE_STRIDE.wrapping_mul(draw));
        let mut perturbed = zeros(&self.c);
        for (row, alpha) in perturbed.iter_mut().zip(&self.alpha) {
            for (slot, logit) in row.iter_mut().zip(alpha) {
                *slot = (logit - (-(noise.next_double() + 1e-300).ln() + 1e-300).ln()) / tau;
            }
        }
        let choice = knapsack::solve(&perturbed, &self.cost, self.budget, self.unit)?;
        let mut onehot = zeros(&self.c);
        for (row, &picked) in onehot.iter_mut().zip(&choice) {
            row[picked] = 1.0;
        }
        let value = self.objective.value(&onehot);
        let toward = self.objective.gradient(&onehot);
        let mut gradient = zeros(&self.c);
        for i in 0..perturbed.len() {
            let y = softmax(&perturbed[i]);
            let mut dot = 0.0;
            for k in 0..y.len() {
                dot += y[k] * toward[i][k];
            }
            for k in 0..y.len() {
                gradient[i][k] = y[k] * (toward[i][k] - dot) / tau / self.settings.samples as f64;
            }
        }
        Ok(Sample {
            value,
            choice,
            gradient,
        })
    }

    /// Ends the current step with its `samples`, in index order: sums their gradients, then the tangent projection,
    /// Adam, the retraction and the moment transport. True once every step is done.
    pub fn update(&mut self, samples: &[Sample]) -> Result<bool, MixedError> {
        if self.done() {
            return Ok(true);
        }
        if samples.len() != self.settings.samples {
            return Err(MixedError::invalid(format!(
                "a step ends with {} samples, not {}",
                self.settings.samples,
                samples.len()
            )));
        }
        let mut gradient = zeros(&self.c);
        for drawn in samples {
            if drawn.value < self.best_value {
                self.best_value = drawn.value;
                self.best_choice = Some(drawn.choice.clone());
            }
            for (row, add) in gradient.iter_mut().zip(&drawn.gradient) {
                for (slot, value) in row.iter_mut().zip(add) {
                    *slot += value;
                }
            }
        }
        let tangent = normal(&self.alpha, &self.c);
        project(&mut gradient, &tangent);
        let correction1 = 1.0 - BETA1.powf(self.step as f64);
        let correction2 = 1.0 - BETA2.powf(self.step as f64);
        let mut movement = zeros(&self.c);
        for (i, row) in movement.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                self.m[i][k] = BETA1 * self.m[i][k] + (1.0 - BETA1) * gradient[i][k];
                self.v[i][k] = BETA2 * self.v[i][k] + (1.0 - BETA2) * gradient[i][k] * gradient[i][k];
                *slot = self.m[i][k] / correction1 / ((self.v[i][k] / correction2).sqrt() + EPSILON);
            }
        }
        project(&mut movement, &tangent);
        for (row, step) in self.alpha.iter_mut().zip(&movement) {
            for (logit, change) in row.iter_mut().zip(step) {
                *logit -= self.settings.lr * change;
            }
        }
        retract(&mut self.alpha, &self.c, &self.direction, self.target);
        let moved = normal(&self.alpha, &self.c);
        project(&mut self.m, &moved);
        self.worst = self.worst.max(manifold_error(&self.alpha, &self.c, self.target));
        self.step += 1;
        Ok(self.done())
    }

    /// The run's assignment once every step is done.
    pub fn result(&self) -> Result<Outcome, MixedError> {
        if !self.done() {
            return Err(MixedError::invalid("the run has steps left"));
        }
        let p: Vec<Vec<f64>> = self.alpha.iter().map(|row| softmax(row)).collect();
        let log_p: Vec<Vec<f64>> = p
            .iter()
            .map(|row| row.iter().map(|value| value.max(1e-300).ln()).collect())
            .collect();
        let choice = knapsack::solve(&log_p, &self.cost, self.budget, self.unit)?;
        let mut last = zeros(&self.c);
        for (row, &picked) in last.iter_mut().zip(&choice) {
            row[picked] = 1.0;
        }
        let value = self.objective.value(&last);
        // The best assignment any sample produced is kept as well: the run has evaluated the objective at it.
        if let Some(best) = &self.best_choice
            && self.best_value < value
        {
            return Ok(Outcome {
                choice: best.clone(),
                p,
                value: self.best_value,
                max_manifold_error: self.worst,
            });
        }
        Ok(Outcome {
            choice,
            p,
            value,
            max_manifold_error: self.worst,
        })
    }
}

fn zeros(shape: &[Vec<f64>]) -> Vec<Vec<f64>> {
    shape.iter().map(|row| vec![0.0; row.len()]).collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p = vec![0.0; logits.len()];
    let mut sum = 0.0;
    for (slot, logit) in p.iter_mut().zip(logits) {
        *slot = (logit - max).exp();
        sum += *slot;
    }
    for slot in &mut p {
        *slot /= sum;
    }
    p
}

fn expected_cost(alpha: &[Vec<f64>], c: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for i in 0..alpha.len() {
        let p = softmax(&alpha[i]);
        for k in 0..p.len() {
            total += p[k] * c[i][k];
        }
    }
    total
}

fn manifold_error(alpha: &[Vec<f64>], c: &[Vec<f64>], target: f64) -> f64 {
    (expected_cost(alpha, c) - target).abs() / target
}

/// The normal of the manifold, `p (c - E_p[c])`.
fn normal(alpha: &[Vec<f64>], c: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut normal = Vec::with_capacity(alpha.len());
    for i in 0..alpha.len() {
        let p = softmax(&alpha[i]);
        let mut mean = 0.0;
        for k in 0..p.len() {
            mean += p[k] * c[i][k];
        }
        normal.push((0..p.len()).map(|k| p[k] * (c[i][k] - mean)).collect());
    }
    normal
}

/// Removes from `vector` its component along `normal`.
fn project(vector: &mut [Vec<f64>], normal: &[Vec<f64>]) {
    let (mut dot, mut squares) = (0.0, 0.0);
    for i in 0..vector.len() {
        for k in 0..vector[i].len() {
            dot += vector[i][k] * normal[i][k];
            squares += normal[i][k] * normal[i][k];
        }
    }
    if squares < 1e-300 {
        return;
    }
    let along = dot / squares;
    for i in 0..vector.len() {
        for k in 0..vector[i].len() {
            vector[i][k] -= along * normal[i][k];
        }
    }
}

/// Adds `t * direction` to `alpha` with `t` in `[-50, 50]` such that the expected bytes equal `target`.
fn retract(alpha: &mut [Vec<f64>], c: &[Vec<f64>], direction: &[Vec<f64>], target: f64) {
    let (mut low, mut high) = (-RETRACTION_RANGE, RETRACTION_RANGE);
    let mut trial = zeros(alpha);
    while high - low > RETRACTION_TOLERANCE * 1e-4 {
        let t = 0.5 * (low + high);
        for i in 0..alpha.len() {
            for k in 0..alpha[i].len() {
                trial[i][k] = alpha[i][k] + t * direction[i][k];
            }
        }
        if expected_cost(&trial, c) < target {
            low = t;
        } else {
            high = t;
        }
    }
    let t = 0.5 * (low + high);
    for i in 0..alpha.len() {
        for k in 0..alpha[i].len() {
            alpha[i][k] += t * direction[i][k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f(p) = sum over objects of (expected value - target)^2`: separable and quadratic in the probabilities.
    struct Quadratic {
        value: Vec<Vec<f64>>,
        target: Vec<f64>,
    }

    impl Quadratic {
        fn expected(&self, p: &[Vec<f64>], i: usize) -> f64 {
            p[i].iter()
                .zip(&self.value[i])
                .map(|(probability, value)| probability * value)
                .sum()
        }
    }

    impl Objective for Quadratic {
        fn value(&self, p: &[Vec<f64>]) -> f64 {
            (0..p.len())
                .map(|i| (self.expected(p, i) - self.target[i]).powi(2))
                .sum()
        }

        fn gradient(&self, p: &[Vec<f64>]) -> Vec<Vec<f64>> {
            (0..p.len())
                .map(|i| {
                    let along = 2.0 * (self.expected(p, i) - self.target[i]);
                    self.value[i].iter().map(|value| along * value).collect()
                })
                .collect()
        }
    }

    struct Instance {
        objective: Quadratic,
        cost: Vec<Vec<i64>>,
        budget: i64,
    }

    fn instance(n: usize, k: usize, seed: i64) -> Instance {
        let mut random = Random::new(seed);
        let mut value = vec![vec![0.0; k]; n];
        let mut target = vec![0.0; n];
        let mut cost = vec![vec![0i64; k]; n];
        let (mut minimum, mut maximum) = (0, 0);
        for i in 0..n {
            for j in 0..k {
                value[i][j] = random.next_double();
                cost[i][j] = 10 + 10 * j as i64 + i64::from(random.next_int_bounded(5).unwrap());
            }
            target[i] = random.next_double();
            minimum += cost[i].iter().min().unwrap();
            maximum += cost[i].iter().max().unwrap();
        }
        Instance {
            objective: Quadratic { value, target },
            cost,
            budget: (minimum + maximum) / 2,
        }
    }

    fn one_hot(choice: &[usize], options: usize) -> Vec<Vec<f64>> {
        choice
            .iter()
            .map(|&picked| (0..options).map(|k| if k == picked { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    fn brute_force(instance: &Instance) -> f64 {
        let (n, k) = (instance.cost.len(), instance.cost[0].len());
        let mut best = f64::INFINITY;
        for code in 0..k.pow(n as u32) {
            let (mut rest, mut bytes) = (code, 0);
            let mut pick = vec![0; n];
            for (slot, costs) in pick.iter_mut().zip(&instance.cost) {
                *slot = rest % k;
                rest /= k;
                bytes += costs[*slot];
            }
            if bytes <= instance.budget {
                best = best.min(instance.objective.value(&one_hot(&pick, k)));
            }
        }
        best
    }

    fn bytes_of(instance: &Instance, choice: &[usize]) -> i64 {
        choice.iter().enumerate().map(|(i, &k)| instance.cost[i][k]).sum()
    }

    #[test]
    fn the_expected_cost_stays_on_the_manifold_every_step() {
        let instance = instance(8, 3, 1);
        let settings = Settings::new(60, 0.1, 4, 1).unwrap();
        let outcome = run(&instance.objective, &instance.cost, instance.budget, 1, settings, 7, 0).unwrap();
        assert!(
            outcome.max_manifold_error <= 1e-6,
            "manifold error {}",
            outcome.max_manifold_error
        );
        let mut expected = 0.0;
        for (row, costs) in outcome.p.iter().zip(&instance.cost) {
            for (probability, bytes) in row.iter().zip(costs) {
                expected += probability * *bytes as f64;
            }
        }
        assert!((expected - instance.budget as f64).abs() <= 1e-6 * instance.budget as f64);
    }

    #[test]
    fn the_brute_force_optimum_of_a_separable_quadratic_is_reached_within_the_budget() {
        let mut reached = 0;
        for seed in 1..=6 {
            let instance = instance(7, 3, seed);
            let settings = Settings::new(200, 0.1, 8, 4).unwrap();
            let results = solve(&instance.objective, &instance.cost, instance.budget, 1, settings, seed).unwrap();
            let best = results
                .iter()
                .map(|outcome| outcome.value)
                .fold(f64::INFINITY, f64::min);
            let optimum = brute_force(&instance);
            assert!(best >= optimum - 1e-12, "better than the optimum?");
            for outcome in &results {
                assert!(bytes_of(&instance, &outcome.choice) <= instance.budget);
            }
            assert!(
                results.windows(2).all(|pair| pair[0].value <= pair[1].value),
                "best first"
            );
            if best <= optimum + 1e-9 {
                reached += 1;
            }
        }
        assert_eq!(reached, 6, "instances solved to the brute-force optimum");
    }

    #[test]
    fn the_same_seed_gives_the_same_assignment_and_the_restarts_differ() {
        let instance = instance(6, 3, 3);
        let settings = Settings::new(40, 0.1, 4, 2).unwrap();
        let again = || solve(&instance.objective, &instance.cost, instance.budget, 1, settings, 5).unwrap();
        assert_eq!(again(), again());
        let first = run(&instance.objective, &instance.cost, instance.budget, 1, settings, 5, 0).unwrap();
        let second = run(&instance.objective, &instance.cost, instance.budget, 1, settings, 5, 1).unwrap();
        assert_ne!(first.p, second.p);
    }

    #[test]
    fn samples_drawn_in_any_order_give_the_run_taken_a_draw_at_a_time() {
        let instance = instance(6, 3, 4);
        let settings = Settings::new(12, 0.1, 3, 1).unwrap();
        let whole = run(&instance.objective, &instance.cost, instance.budget, 1, settings, 9, 0).unwrap();
        let mut split = Run::new(&instance.objective, &instance.cost, instance.budget, 1, settings, 9, 0).unwrap();
        while !split.done() {
            // The step's samples are drawn last first, then handed over in index order.
            let mut drawn: Vec<(usize, Sample)> = (0..split.samples())
                .rev()
                .map(|index| (index, split.sample(index).unwrap()))
                .collect();
            drawn.sort_by_key(|(index, _)| *index);
            let drawn: Vec<Sample> = drawn.into_iter().map(|(_, sample)| sample).collect();
            split.update(&drawn).unwrap();
        }
        assert_eq!(split.result().unwrap(), whole);
    }

    #[test]
    fn what_a_run_cannot_start_from_is_refused() {
        let instance = instance(4, 3, 1);
        let objective = &instance.objective;
        let settings = Settings::default();
        let cheapest: i64 = instance.cost.iter().map(|row| row.iter().min().unwrap()).sum();
        assert_eq!(
            run(objective, &instance.cost, cheapest - 1, 1, settings, 1, 0).unwrap_err(),
            MixedError::BudgetTooSmall(cheapest - 1)
        );
        assert!(run(objective, &[], 10, 1, settings, 1, 0).is_err());
        assert!(run(objective, &[vec![]], 10, 1, settings, 1, 0).is_err());
        assert!(run(objective, &[vec![-1, 2]], 10, 1, settings, 1, 0).is_err());
        assert!(Settings::new(0, 0.1, 1, 1).is_err());
        assert!(Settings::new(1, 0.0, 1, 1).is_err());
        assert!(Settings::new(1, f64::NAN, 1, 1).is_err());
        let mut started = Run::new(objective, &instance.cost, instance.budget, 1, settings, 1, 0).unwrap();
        assert!(started.result().is_err());
        assert!(started.update(&[]).is_err());
    }
}

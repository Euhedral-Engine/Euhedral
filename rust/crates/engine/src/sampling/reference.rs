//! The test oracle: a straightforward selection that builds the candidate list, keeps the top-k with a heap, sorts
//! and draws step by step. It takes the unit draw as an argument so a test can give it the keyed draw of a position.

use super::config::{SamplingConfig, SamplingError};

#[derive(Debug)]
pub struct ReferenceSampler {
    config: SamplingConfig,
    vocabulary: usize,
}

impl ReferenceSampler {
    pub fn new(config: SamplingConfig, vocabulary: usize) -> Self {
        Self { config, vocabulary }
    }

    pub fn select(&self, logits: &[f32], unit: f64) -> Result<u32, SamplingError> {
        assert_eq!(logits.len(), self.vocabulary);
        if self.config.is_greedy() {
            return argmax(logits);
        }

        // Scale scores before applying the filters; top-k ties prefer the lower token id.
        let temperature = f64::from(self.config.temperature());
        let mut scores = vec![0.0_f64; self.vocabulary];
        let mut candidates: Vec<u32> = Vec::new();
        for (id, &logit) in (0..).zip(logits) {
            if logit.is_nan() || logit == f32::NEG_INFINITY {
                continue;
            }
            scores[id as usize] = f64::from(logit) / temperature;
            candidates.push(id);
        }
        if candidates.is_empty() {
            return Err(SamplingError::NoSelectableToken);
        }

        let top_k = self.config.top_k() as usize;
        if top_k > 0 && top_k < candidates.len() {
            candidates = retain_top_k(&candidates, top_k, &scores);
        }

        let apply_top_p = self.config.top_p() < 1.0;
        if apply_top_p {
            sort_by_priority(&mut candidates, &scores);
        }

        let probabilities = normalized(&candidates, &scores)?;
        let mut retained = candidates.len();
        if apply_top_p {
            let mut cumulative = 0.0;
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if cumulative >= f64::from(self.config.top_p()) {
                    retained = index + 1;
                    break;
                }
            }
        }
        draw(&candidates, &probabilities, retained, unit)
    }
}

fn argmax(logits: &[f32]) -> Result<u32, SamplingError> {
    let mut best: Option<(u32, f32)> = None;
    for (id, &logit) in (0..).zip(logits) {
        if logit.is_nan() || logit == f32::NEG_INFINITY {
            continue;
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((id, logit));
        }
    }
    best.map(|(id, _)| id).ok_or(SamplingError::NoSelectableToken)
}

fn retain_top_k(candidates: &[u32], top_k: usize, scores: &[f64]) -> Vec<u32> {
    let mut heap: Vec<u32> = Vec::with_capacity(top_k);
    for &id in candidates {
        if heap.len() < top_k {
            heap.push(id);
            let last = heap.len() - 1;
            sift_up(&mut heap, last, scores);
        } else if compare(id, heap[0], scores) < 0 {
            heap[0] = id;
            let size = heap.len();
            sift_down(&mut heap, size, 0, scores);
        }
    }
    heap
}

fn sift_up(heap: &mut [u32], mut index: usize, scores: &[f64]) {
    while index > 0 {
        let parent = (index - 1) / 2;
        if compare(heap[parent], heap[index], scores) >= 0 {
            return;
        }
        heap.swap(parent, index);
        index = parent;
    }
}

fn sift_down(heap: &mut [u32], size: usize, mut index: usize, scores: &[f64]) {
    loop {
        let left = 2 * index + 1;
        if left >= size {
            return;
        }
        let right = left + 1;
        let worse = if right < size && compare(heap[left], heap[right], scores) < 0 {
            right
        } else {
            left
        };
        if compare(heap[index], heap[worse], scores) >= 0 {
            return;
        }
        heap.swap(index, worse);
        index = worse;
    }
}

fn sort_by_priority(ids: &mut [u32], scores: &[f64]) {
    let count = ids.len();
    for root in (0..count / 2).rev() {
        sift_down(ids, count, root, scores);
    }
    for end in (1..count).rev() {
        ids.swap(0, end);
        sift_down(ids, end, 0, scores);
    }
}

fn compare(left: u32, right: u32, scores: &[f64]) -> i32 {
    let (l, r) = (scores[left as usize], scores[right as usize]);
    if l > r {
        -1
    } else if l < r {
        1
    } else {
        left.cmp(&right) as i32
    }
}

fn normalized(ids: &[u32], scores: &[f64]) -> Result<Vec<f64>, SamplingError> {
    let mut probabilities = vec![0.0; ids.len()];
    let infinite = ids.iter().filter(|&&id| scores[id as usize] == f64::INFINITY).count();
    let mut maximum = f64::NEG_INFINITY;
    for &id in ids {
        if scores[id as usize] > maximum {
            maximum = scores[id as usize];
        }
    }
    if infinite > 0 {
        for (probability, &id) in probabilities.iter_mut().zip(ids) {
            if scores[id as usize] == f64::INFINITY {
                *probability = 1.0 / infinite as f64;
            }
        }
        return Ok(probabilities);
    }
    let mut total = 0.0;
    for (probability, &id) in probabilities.iter_mut().zip(ids) {
        *probability = (scores[id as usize] - maximum).exp();
        total += *probability;
    }
    if total.is_nan() || total <= 0.0 || !total.is_finite() {
        return Err(SamplingError::NoNormalizableMass);
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

fn draw(ids: &[u32], probabilities: &[f64], retained: usize, unit: f64) -> Result<u32, SamplingError> {
    let mut mass = 0.0;
    let mut last_positive = None;
    for index in 0..retained {
        mass += probabilities[index];
        if probabilities[index] > 0.0 {
            last_positive = Some(ids[index]);
        }
    }
    let Some(last_positive) = last_positive else {
        return Err(SamplingError::FiltersRemovedAllMass);
    };
    if mass.is_nan() || mass <= 0.0 || !mass.is_finite() {
        return Err(SamplingError::FiltersRemovedAllMass);
    }
    let target = unit * mass;
    let mut cumulative = 0.0;
    for index in 0..retained {
        cumulative += probabilities[index];
        if probabilities[index] > 0.0 && target < cumulative {
            return Ok(ids[index]);
        }
    }
    Ok(last_positive)
}

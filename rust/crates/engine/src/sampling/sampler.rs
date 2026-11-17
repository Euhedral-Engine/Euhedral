//! The selection itself. The chosen token depends on the order of the floating-point operations and on the layout of
//! the heaps, so the comparisons are `>` on the raw float (never a total order), the heaps are sifted exactly as
//! written here, and sums run in array order.

use std::cmp::Ordering;

use super::config::{SamplingConfig, SamplingError};

/// Fewest best candidates a filtered selection ranks before it tests them. Part of the algorithm: a different
/// value tests different tokens and can select a different one.
const MIN_RANKED: usize = 64;

/// Which tokens a selection may choose.
pub trait TokenFilter {
    /// Whether the token may be chosen.
    fn allows(&self, id: u32) -> bool;

    /// Overwrites every disallowed entry of `row` with negative infinity. An implementation that knows its allowed
    /// set at once can do this without testing every token.
    fn mask_disallowed(&self, row: &mut [f32]) {
        for (id, logit) in (0..).zip(row.iter_mut()) {
            if !self.allows(id) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// A uniform draw in [0, 1) that depends only on `seed` and `position`: the SplitMix64 finalizer applied to
/// `seed + gamma * (position + 1)`, so neighbouring positions give independent draws.
pub fn unit_draw(seed: u64, position: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(position.wrapping_add(1)));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// The one exponential the probabilities use, so that it can be replaced as a whole.
fn exp(x: f64) -> f64 {
    x.exp()
}

/// The scratch rows and the selection loops. Kept apart from the BF16 conversion row so a conversion can be
/// selected from without moving the row out of its owner.
#[derive(Debug)]
struct Selector {
    config: SamplingConfig,
    vocabulary: usize,
    /// Scaled score of each candidate token, indexed by token id.
    scores: Vec<f64>,
    /// The candidate token ids, in the order the draw walks them.
    candidates: Vec<u32>,
    /// The probability of each candidate, indexed like `candidates`.
    probabilities: Vec<f64>,
    /// The best tokens of a filtered selection, best first.
    ranked: Vec<u32>,
}

impl Selector {
    fn new(config: SamplingConfig, vocabulary: usize) -> Result<Self, SamplingError> {
        if vocabulary == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        let stochastic = !config.is_greedy();
        let needed = needed(&config, vocabulary);
        let limit = if needed > 0 {
            vocabulary.min(MIN_RANKED.max(needed.saturating_mul(4)))
        } else {
            0
        };
        let rows = if stochastic { vocabulary } else { 0 };
        Ok(Self {
            config,
            vocabulary,
            scores: vec![0.0; rows],
            candidates: vec![0; rows],
            probabilities: vec![0.0; rows],
            ranked: vec![0; limit],
        })
    }

    fn is_greedy(&self) -> bool {
        self.config.is_greedy()
    }

    fn select(&mut self, row: &[f32], position: u64) -> Result<u32, SamplingError> {
        self.check_width(row.len())?;
        self.select_row(row, unit_draw(self.config.seed(), position))
    }

    fn select_filtered(
        &mut self,
        row: &mut [f32],
        filter: &dyn TokenFilter,
        position: u64,
    ) -> Result<u32, SamplingError> {
        self.check_width(row.len())?;
        let unit = unit_draw(self.config.seed(), position);
        let needed = needed(&self.config, self.vocabulary);
        if needed > 0 {
            let limit = self.ranked.len();
            let ranked = &mut self.ranked[..];
            let ranked_count = rank_best(row, limit, ranked);
            let mut found = 0;
            for index in 0..ranked_count {
                if found >= needed {
                    break;
                }
                if filter.allows(ranked[index]) {
                    ranked[found] = ranked[index];
                    found += 1;
                }
            }
            // Every selectable token was ranked when fewer than the limit exist, so the allowed ones are all found.
            if found == needed || (ranked_count < limit && found > 0) {
                if self.config.is_greedy() {
                    return Ok(ranked[0]);
                }
                let temperature = f64::from(self.config.temperature());
                for &id in &ranked[..found] {
                    self.scores[id as usize] = f64::from(row[id as usize]) / temperature;
                }
                self.candidates[..found].copy_from_slice(&ranked[..found]);
                return self.draw_from(found, unit);
            }
        }
        filter.mask_disallowed(row);
        self.select_row(row, unit)
    }

    fn check_width(&self, got: usize) -> Result<(), SamplingError> {
        if got == self.vocabulary {
            Ok(())
        } else {
            Err(SamplingError::RowWidth {
                expected: self.vocabulary,
                got,
            })
        }
    }

    /// Selection from an unfiltered row with the draw already made.
    fn select_row(&mut self, row: &[f32], unit: f64) -> Result<u32, SamplingError> {
        if self.is_greedy() {
            return argmax(row);
        }
        // Scale scores before applying the filters; top-k ties prefer the lower token id. Only the candidates'
        // entries of the reused score row are written and read.
        let top_k = self.config.top_k() as usize;
        let count = if top_k > 0 && top_k < self.vocabulary {
            self.scale_top_k(row, top_k)
        } else {
            self.scale_all(row)
        };
        if count == 0 {
            return Err(SamplingError::NoSelectableToken);
        }
        self.draw_from(count, unit)
    }

    /// Scales every selectable logit and lists the candidates in token-id order.
    fn scale_all(&mut self, row: &[f32]) -> usize {
        let temperature = f64::from(self.config.temperature());
        let mut count = 0;
        for (id, &logit) in (0..).zip(row) {
            if unselectable(logit) {
                continue;
            }
            self.scores[id as usize] = f64::from(logit) / temperature;
            self.candidates[count] = id;
            count += 1;
        }
        count
    }

    /// Retains the `top_k` best candidates in one pass, leaving them in `candidates` exactly as a worst-first heap
    /// built over every candidate in token-id order would. With no more selectable tokens than `top_k`, every
    /// candidate is retained in token-id order instead.
    ///
    /// A later candidate replaces the heap's worst entry only when it scores strictly higher: equal scores keep the
    /// lower token id, which is already in the heap. Dividing a non-NaN float by a positive, finite float
    /// temperature preserves both its order and its ties in double precision, so that test is made on the raw
    /// logit and only retained candidates are scaled.
    fn scale_top_k(&mut self, row: &[f32], top_k: usize) -> usize {
        let temperature = f64::from(self.config.temperature());
        let scores = &mut self.scores[..];
        let heap = &mut self.candidates[..];
        let mut heap_size = 0;
        let mut next = 0;
        while next < row.len() && heap_size < top_k {
            let id = next;
            next += 1;
            let logit = row[id];
            if unselectable(logit) {
                continue;
            }
            scores[id] = f64::from(logit) / temperature;
            heap[heap_size] = id as u32;
            sift_up_worst_first(heap, heap_size, scores);
            heap_size += 1;
        }
        // Only when another candidate follows the first `top_k` does the heap order apply.
        let mut exceeded = false;
        while next < row.len() && !exceeded {
            let id = next;
            next += 1;
            if unselectable(row[id]) {
                continue;
            }
            exceeded = true;
            replace_worst(row, id, temperature, scores, heap, heap_size);
        }
        if !exceeded {
            heap[..heap_size].sort_unstable();
            return heap_size;
        }
        // NaN and negative infinity never compare greater than a retained logit.
        let mut worst = row[heap[0] as usize];
        while next < row.len() {
            let id = next;
            next += 1;
            if row[id] > worst {
                replace_worst(row, id, temperature, scores, heap, heap_size);
                worst = row[heap[0] as usize];
            }
        }
        top_k
    }

    /// Normalises the first `count` candidates' scores, keeps the top-p prefix when the top-p is below one, and
    /// draws.
    fn draw_from(&mut self, count: usize, unit: f64) -> Result<u32, SamplingError> {
        let apply_top_p = self.config.top_p() < 1.0;
        let candidates = &mut self.candidates[..count];
        // Top-p uses the normalised top-k distribution, and the draw renormalises the retained prefix.
        if apply_top_p {
            sort_by_priority(candidates, &self.scores);
        }
        let probabilities = &mut self.probabilities[..count];
        normalize(candidates, &self.scores, probabilities)?;
        let mut retained = count;
        if apply_top_p {
            let top_p = f64::from(self.config.top_p());
            let mut cumulative = 0.0;
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if cumulative >= top_p {
                    retained = index + 1;
                    break;
                }
            }
        }
        draw(candidates, probabilities, retained, unit)
    }
}

/// Selects one token from one row of vocabulary logits, drawing with [`unit_draw`] of the seed and the position of
/// the token being chosen, so a position selects the same token however it is reached. NaN and negative infinity
/// are never selected; positive infinities share the probability mass equally.
///
/// One sampler serves one generation. It is not shared: its scratch rows are written by every selection, and all of
/// them are allocated when it is made.
#[derive(Debug)]
pub struct TokenSampler {
    selector: Selector,
    /// A BF16 row converted to f32.
    row: Vec<f32>,
}

impl TokenSampler {
    /// A sampler for rows of `vocabulary` logits.
    pub fn new(config: SamplingConfig, vocabulary: usize) -> Result<Self, SamplingError> {
        Ok(Self {
            selector: Selector::new(config, vocabulary)?,
            row: vec![0.0; vocabulary],
        })
    }

    pub fn vocabulary(&self) -> usize {
        self.selector.vocabulary
    }

    /// Whether every selection is the argmax of the row.
    pub fn is_greedy(&self) -> bool {
        self.selector.is_greedy()
    }

    /// Selects the token at `position` from `row`.
    pub fn select(&mut self, row: &[f32], position: u64) -> Result<u32, SamplingError> {
        self.selector.select(row, position)
    }

    /// Selects the token at `position` from `row` as if every token the filter rejects were removed, testing as few
    /// tokens as it can: the best candidates are tested in order until enough are allowed (one when greedy, top-k
    /// when sampling), and every token only when they are not, in which case the disallowed entries of `row` are
    /// overwritten with negative infinity.
    ///
    /// A greedy selection, and sampling with a top-p below one, select exactly what the masked row selects. With a
    /// top-p of one the same candidates and probabilities are drawn in priority order rather than in the masked
    /// row's heap order, so a seed may select another token of the same distribution.
    pub fn select_filtered(
        &mut self,
        row: &mut [f32],
        filter: &dyn TokenFilter,
        position: u64,
    ) -> Result<u32, SamplingError> {
        self.selector.select_filtered(row, filter, position)
    }

    /// Selects as [`TokenSampler::select`] and [`TokenSampler::select_filtered`] do from a row of BF16 logits, two
    /// little-endian bytes each, converted exactly to f32.
    pub fn select_bf16(
        &mut self,
        row: &[u8],
        position: u64,
        filter: Option<&dyn TokenFilter>,
    ) -> Result<u32, SamplingError> {
        let Self {
            selector,
            row: converted,
        } = self;
        if row.len() != selector.vocabulary * 2 {
            return Err(SamplingError::RowBytes {
                expected: selector.vocabulary * 2,
                got: row.len(),
            });
        }
        for (logit, pair) in converted.iter_mut().zip(row.as_chunks::<2>().0) {
            *logit = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
        }
        match filter {
            None => selector.select(converted, position),
            Some(filter) => selector.select_filtered(converted, filter, position),
        }
    }
}

/// How many allowed candidates a filtered selection needs before it may stop testing: one when greedy, the top-k
/// when it is below the vocabulary, else the whole row is needed and the row is masked.
fn needed(config: &SamplingConfig, vocabulary: usize) -> usize {
    let top_k = config.top_k() as usize;
    if config.is_greedy() {
        1
    } else if top_k > 0 && top_k < vocabulary {
        top_k
    } else {
        0
    }
}

fn unselectable(logit: f32) -> bool {
    logit.is_nan() || logit == f32::NEG_INFINITY
}

/// One comparison per logit. NaN never compares greater, negative infinity never beats the initial negative
/// infinity, and a strict comparison keeps the lowest token id among equal maxima.
fn argmax(row: &[f32]) -> Result<u32, SamplingError> {
    let mut best = None;
    let mut best_logit = f32::NEG_INFINITY;
    for (id, &logit) in (0..).zip(row) {
        if logit > best_logit {
            best_logit = logit;
            best = Some(id);
        }
    }
    best.ok_or(SamplingError::NoSelectableToken)
}

/// Writes the `limit` best selectable tokens to `ranked`, best first (higher logit, then lower token id), and
/// returns how many there are: fewer than `limit` only when the row has fewer selectable tokens.
fn rank_best(row: &[f32], limit: usize, ranked: &mut [u32]) -> usize {
    let mut size = 0;
    for (id, &logit) in (0..).zip(row) {
        if unselectable(logit) {
            continue;
        }
        if size < limit {
            ranked[size] = id;
            sift_up_ranked(ranked, size, row);
            size += 1;
        } else if logit > row[ranked[0] as usize] {
            // A later token with an equal logit has the higher id and ranks lower: it never enters.
            ranked[0] = id;
            sift_down_ranked(ranked, size, 0, row);
        }
    }
    for end in (1..size).rev() {
        ranked.swap(0, end);
        sift_down_ranked(ranked, end, 0, row);
    }
    size
}

/// Worst-first order of the ranking heap: a lower logit, or an equal logit and a higher id, is worse.
fn ranks_below(left: u32, right: u32, row: &[f32]) -> bool {
    let (l, r) = (row[left as usize], row[right as usize]);
    l < r || (l == r && left > right)
}

fn sift_up_ranked(heap: &mut [u32], mut index: usize, row: &[f32]) {
    while index > 0 {
        let parent = (index - 1) >> 1;
        if !ranks_below(heap[index], heap[parent], row) {
            return;
        }
        heap.swap(parent, index);
        index = parent;
    }
}

fn sift_down_ranked(heap: &mut [u32], size: usize, mut index: usize, row: &[f32]) {
    loop {
        let left = (index << 1) + 1;
        if left >= size {
            return;
        }
        let right = left + 1;
        let worse = if right < size && ranks_below(heap[right], heap[left], row) {
            right
        } else {
            left
        };
        if !ranks_below(heap[worse], heap[index], row) {
            return;
        }
        heap.swap(index, worse);
        index = worse;
    }
}

/// Replaces the heap's worst candidate with `id` when that scores strictly higher.
fn replace_worst(row: &[f32], id: usize, temperature: f64, scores: &mut [f64], heap: &mut [u32], heap_size: usize) {
    // Both logits are selectable, so this is the negation of `>`.
    if row[id] <= row[heap[0] as usize] {
        return;
    }
    scores[id] = f64::from(row[id]) / temperature;
    heap[0] = id as u32;
    sift_down_worst_first(heap, heap_size, 0, scores);
}

fn sift_up_worst_first(heap: &mut [u32], mut index: usize, scores: &[f64]) {
    while index > 0 {
        let parent = (index - 1) >> 1;
        if compare_priority(heap[parent], heap[index], scores) >= 0 {
            return;
        }
        heap.swap(parent, index);
        index = parent;
    }
}

fn sift_down_worst_first(heap: &mut [u32], heap_size: usize, mut index: usize, scores: &[f64]) {
    loop {
        let left = (index << 1) + 1;
        if left >= heap_size {
            return;
        }
        let right = left + 1;
        let worse_child = if right < heap_size && compare_priority(heap[left], heap[right], scores) < 0 {
            right
        } else {
            left
        };
        if compare_priority(heap[index], heap[worse_child], scores) >= 0 {
            return;
        }
        heap.swap(index, worse_child);
        index = worse_child;
    }
}

/// Orders the candidates best first. A worst-first heap produces best-first order without recursion.
fn sort_by_priority(ids: &mut [u32], scores: &[f64]) {
    let count = ids.len();
    for root in (0..count >> 1).rev() {
        sift_down_worst_first(ids, count, root, scores);
    }
    for end in (1..count).rev() {
        ids.swap(0, end);
        sift_down_worst_first(ids, end, 0, scores);
    }
}

/// Negative when the left token has priority: a higher score, then a lower id.
fn compare_priority(left: u32, right: u32, scores: &[f64]) -> i32 {
    let (l, r) = (scores[left as usize], scores[right as usize]);
    if l > r {
        -1
    } else if l < r {
        1
    } else {
        match left.cmp(&right) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

/// Writes each candidate's probability to `probabilities`, in candidate order. When any score is positive
/// infinity those candidates share the mass equally; otherwise the weights are `exp(score - max)` summed in order.
fn normalize(ids: &[u32], scores: &[f64], probabilities: &mut [f64]) -> Result<(), SamplingError> {
    probabilities.fill(0.0);
    let mut infinite = 0_u32;
    let mut maximum = f64::NEG_INFINITY;
    for &id in ids {
        let score = scores[id as usize];
        if score == f64::INFINITY {
            infinite += 1;
        }
        if score > maximum {
            maximum = score;
        }
    }

    if infinite > 0 {
        let share = 1.0 / f64::from(infinite);
        for (probability, &id) in probabilities.iter_mut().zip(ids) {
            if scores[id as usize] == f64::INFINITY {
                *probability = share;
            }
        }
        return Ok(());
    }

    let mut total = 0.0;
    for (probability, &id) in probabilities.iter_mut().zip(ids) {
        *probability = exp(scores[id as usize] - maximum);
        total += *probability;
    }
    if total.is_nan() || total <= 0.0 || !total.is_finite() {
        return Err(SamplingError::NoNormalizableMass);
    }
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
    Ok(())
}

/// Picks the first of the first `retained` candidates whose running mass passes `unit` times their total mass,
/// falling back to the last with positive probability.
fn draw(ids: &[u32], probabilities: &[f64], retained: usize, unit: f64) -> Result<u32, SamplingError> {
    let mut mass = 0.0;
    let mut last_positive = None;
    for (&id, &probability) in ids.iter().zip(probabilities).take(retained) {
        mass += probability;
        if probability > 0.0 {
            last_positive = Some(id);
        }
    }
    let Some(last_positive) = last_positive.filter(|_| mass > 0.0 && mass.is_finite()) else {
        return Err(SamplingError::FiltersRemovedAllMass);
    };

    let target = unit * mass;
    let mut cumulative = 0.0;
    for (&id, &probability) in ids.iter().zip(probabilities).take(retained) {
        cumulative += probability;
        if probability > 0.0 && target < cumulative {
            return Ok(id);
        }
    }
    Ok(last_positive)
}

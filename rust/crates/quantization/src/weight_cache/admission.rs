//! Whether another engine may start while others are scored: the bytes it would add against the memory that is free.
//! Engines are admitted in the order they were submitted, so a large one is not starved by small ones behind it.

use std::collections::HashMap;
use std::fmt;

use super::cache::Key;

/// What one engine needs: each object it uses (by key, with its bytes), and the bytes beyond its weights (its
/// sequences' state and its workspace).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Need {
    pub objects: HashMap<Key, u64>,
    pub other_bytes: u64,
}

impl Need {
    /// The bytes starting it adds when `resident` says which objects are already held: its other bytes and the objects
    /// not yet there.
    pub fn additional(&self, resident: impl Fn(&Key) -> bool) -> u64 {
        let mut bytes = self.other_bytes;
        for (key, object) in &self.objects {
            if !resident(key) {
                bytes = bytes.saturating_add(*object);
            }
        }
        bytes
    }

    pub fn weight_bytes(&self) -> u64 {
        self.objects.values().fold(0, |sum, object| sum.saturating_add(*object))
    }
}

/// Why an engine does not start now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The limit on engines at once is reached.
    Limit { running: usize, limit: usize },
    /// It adds more than is free.
    Short { additional: u64, available: i64 },
}

impl Refusal {
    /// The bytes by which the engine does not fit (0 for a refusal by the limit).
    pub fn short_by(&self) -> u64 {
        match *self {
            Refusal::Limit { .. } => 0,
            Refusal::Short { additional, available } => {
                u64::try_from(i128::from(additional) - i128::from(available)).unwrap_or(0)
            }
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Refusal::Limit { running, limit } => {
                write!(formatter, "{running} engines are running, the limit is {limit}")
            }
            Refusal::Short { additional, available } => write!(
                formatter,
                "the engine adds {additional} bytes and {available} are available: short by {}",
                self.short_by()
            ),
        }
    }
}

#[derive(Debug)]
pub struct Admission;

impl Admission {
    /// The bytes an engine may add now: the lower of the free bytes the device reports and the free bytes when the
    /// scorer opened less what the cache holds and the running engines' other bytes (an allocation that has been
    /// decided but not made does not show in the first), less `margin`. Negative when there is no room.
    pub fn available(free_now: i64, free_at_open: i64, cache_bytes: i64, running_other_bytes: i64, margin: i64) -> i64 {
        free_now
            .min(
                free_at_open
                    .saturating_sub(cache_bytes)
                    .saturating_sub(running_other_bytes),
            )
            .saturating_sub(margin)
    }

    /// Whether the next engine starts alongside `running` others: the limit `max_concurrent` (0 for none) allows one
    /// more and `additional` bytes are within what is [`available`](Self::available). With nothing running it always
    /// starts; a refusal says why and by how much.
    pub fn check(running: usize, max_concurrent: usize, additional: u64, available: i64) -> Result<(), Refusal> {
        if running == 0 {
            return Ok(());
        }
        if max_concurrent > 0 && running >= max_concurrent {
            return Err(Refusal::Limit {
                running,
                limit: max_concurrent,
            });
        }
        if i128::from(additional) <= i128::from(available) {
            Ok(())
        } else {
            Err(Refusal::Short { additional, available })
        }
    }

    pub fn admits(running: usize, max_concurrent: usize, additional: u64, available: i64) -> bool {
        Self::check(running, max_concurrent, additional, available).is_ok()
    }
}

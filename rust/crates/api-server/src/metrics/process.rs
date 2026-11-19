//! What the operating system says about this process: how long it has run, when it started, the CPU time it used and the
//! files it holds open. A reading the platform cannot answer is `None`, and the series built on it is left out of the
//! scrape rather than written as zero.

use std::time::{SystemTime, UNIX_EPOCH};

/// One reading of the process.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reading {
    /// Seconds since the process started.
    pub uptime_seconds: Option<f64>,
    /// When the process started, in seconds since the Unix epoch.
    pub start_time_seconds: Option<f64>,
    /// Seconds of CPU time the process used, user and system together.
    pub cpu_seconds: Option<f64>,
    /// Files (descriptors or handles) the process holds open now.
    pub open_files: Option<u64>,
    /// The most files the process may hold open, when the system sets a limit.
    pub max_files: Option<u64>,
}

/// Reads the process now.
pub fn read() -> Reading {
    let mut reading = platform::read();
    if reading.start_time_seconds.is_none()
        && let (Some(uptime), Ok(now)) = (reading.uptime_seconds, SystemTime::now().duration_since(UNIX_EPOCH))
    {
        reading.start_time_seconds = Some(now.as_secs_f64() - uptime);
    }
    reading
}

/// The fields of `/proc/self/stat` after the command name, split at spaces: `after[0]` is field 3 (the state).
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
fn stat_fields(stat: &str) -> Option<Vec<&str>> {
    // The command name is in parentheses and may hold spaces and parentheses itself: the last `)` ends it.
    let end = stat.rfind(')')?;
    Some(stat[end + 1..].split_whitespace().collect())
}

/// The soft limit of open files in the text of `/proc/self/limits`.
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
fn soft_open_files_limit(limits: &str) -> Option<u64> {
    let line = limits.lines().find(|line| line.starts_with("Max open files"))?;
    let soft = line.split_whitespace().nth(3)?;
    if soft == "unlimited" { None } else { soft.parse().ok() }
}

#[cfg(target_os = "linux")]
#[allow(
    clippy::disallowed_methods,
    reason = "the files of /proc are made by the kernel from memory when read: a read never waits on a device"
)]
mod platform {
    use std::fs::{read_dir, read_to_string};

    use super::{Reading, soft_open_files_limit, stat_fields};

    /// The unit of the times in `/proc`: fixed by the kernel's interface (`USER_HZ`), whatever the tick of the kernel.
    const TICKS_PER_SECOND: f64 = 100.0;

    pub(super) fn read() -> Reading {
        let mut reading = Reading::default();
        if let Some(stat) = read_to_string("/proc/self/stat").ok()
            && let Some(after) = stat_fields(&stat)
        {
            // Fields 14 and 15 (user and system time) and 22 (start time), counted from field 3.
            let ticks = |at: usize| after.get(at).and_then(|text| text.parse::<f64>().ok());
            if let (Some(user), Some(system)) = (ticks(11), ticks(12)) {
                reading.cpu_seconds = Some((user + system) / TICKS_PER_SECOND);
            }
            let booted = read_to_string("/proc/uptime").ok().and_then(|text| {
                text.split_whitespace()
                    .next()
                    .and_then(|first| first.parse::<f64>().ok())
            });
            if let (Some(started), Some(booted)) = (ticks(19), booted) {
                reading.uptime_seconds = Some((booted - started / TICKS_PER_SECOND).max(0.0));
            }
        }
        reading.open_files = read_dir("/proc/self/fd")
            .ok()
            // The directory's own descriptor is open while it is read.
            .map(|entries| (entries.count() as u64).saturating_sub(1));
        reading.max_files = read_to_string("/proc/self/limits")
            .ok()
            .and_then(|limits| soft_open_files_limit(&limits));
        reading
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;

    use super::Reading;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        /// In units of 100 ns.
        fn units(self) -> u64 {
            (u64::from(self.high) << 32) | u64::from(self.low)
        }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        fn GetProcessHandleCount(process: *mut c_void, count: *mut u32) -> i32;
    }

    /// Between 1601-01-01 and 1970-01-01, in units of 100 ns.
    const UNIX_EPOCH_UNITS: u64 = 116_444_736_000_000_000;
    const UNITS_PER_SECOND: f64 = 10_000_000.0;

    pub(super) fn read() -> Reading {
        let mut reading = Reading::default();
        let (mut creation, mut exit, mut kernel, mut user) = (
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
        );
        // SAFETY: the pseudo handle of the current process is always valid, and each pointer is to a live `FileTime`.
        let timed = unsafe { GetProcessTimes(GetCurrentProcess(), &mut creation, &mut exit, &mut kernel, &mut user) };
        if timed != 0 {
            reading.cpu_seconds = Some((kernel.units() + user.units()) as f64 / UNITS_PER_SECOND);
            if let Some(since_epoch) = creation.units().checked_sub(UNIX_EPOCH_UNITS) {
                reading.start_time_seconds = Some(since_epoch as f64 / UNITS_PER_SECOND);
            }
        }
        let mut handles = 0u32;
        // SAFETY: as above; `handles` is a live `u32`.
        if unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut handles) } != 0 {
            reading.open_files = Some(u64::from(handles));
        }
        reading
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod platform {
    use super::Reading;

    pub(super) fn read() -> Reading {
        Reading::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stat_fields_follow_a_command_name_with_spaces_and_parentheses() {
        let stat = "42 (a b) c) S 1 2 3";
        assert_eq!(stat_fields(stat), Some(vec!["S", "1", "2", "3"]));
        assert_eq!(stat_fields("no name"), None);
    }

    #[test]
    fn the_soft_limit_is_the_first_number_and_unlimited_is_none() {
        let text = "Limit Soft Limit Hard Limit Units\nMax open files            1024                 4096                 files\n";
        assert_eq!(soft_open_files_limit(text), Some(1024));
        let text = "Max open files            unlimited            unlimited            files\n";
        assert_eq!(soft_open_files_limit(text), None);
    }

    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn this_process_answers() {
        let reading = read();
        assert!(reading.cpu_seconds.is_some_and(|cpu| cpu >= 0.0));
        assert!(reading.open_files.is_some_and(|files| files > 0));
        assert!(reading.start_time_seconds.is_some_and(|start| start > 1.0e9));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_uptime_and_the_cpu_time_do_not_go_back() {
        let first = read();
        let mut spin = 0u64;
        for each in 0..2_000_000u64 {
            spin = spin.wrapping_add(each);
        }
        assert!(spin > 0);
        let second = read();
        assert!(second.uptime_seconds >= first.uptime_seconds);
        assert!(second.cpu_seconds >= first.cpu_seconds);
    }
}

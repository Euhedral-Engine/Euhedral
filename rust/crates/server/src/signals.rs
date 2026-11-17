//! The stop request: SIGTERM and SIGINT (on Windows Ctrl-C, Ctrl-Break and the console's close, logoff and shutdown
//! events). The first one asks the process to stop in order and gives the exit code (128 plus the signal's number); a
//! second one ends the process at once with the same code, without waiting for the stop in progress.

#[cfg(unix)]
use signal_hook::iterator::Signals as Delivered;
#[cfg(unix)]
use std::fmt::{self, Debug, Formatter};
#[cfg(unix)]
use std::io;

use crate::{EXIT_SIGINT, EXIT_SIGTERM};

/// The handlers, installed. [`wait`](Signals::wait) blocks the calling (main) thread until the first stop request.
#[cfg(unix)]
pub struct Signals {
    signals: Delivered,
}

#[cfg(unix)]
impl Debug for Signals {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Signals").finish_non_exhaustive()
    }
}

/// Installs the handlers.
///
/// Until the first signal arrives a second one cannot exist, so the order of registration makes the rule: for each
/// signal the handler that exits when the flag is already set comes first, then the one that wakes `wait`, then the one
/// that sets the flag.
#[cfg(unix)]
pub fn install() -> io::Result<Signals> {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use signal_hook::consts::{SIGINT, SIGTERM};
    use signal_hook::flag::{register, register_conditional_shutdown};

    let again = Arc::new(AtomicBool::new(false));
    register_conditional_shutdown(SIGTERM, i32::from(EXIT_SIGTERM), again.clone())?;
    register_conditional_shutdown(SIGINT, i32::from(EXIT_SIGINT), again.clone())?;
    let signals = Delivered::new([SIGTERM, SIGINT])?;
    register(SIGTERM, again.clone())?;
    register(SIGINT, again)?;
    Ok(Signals { signals })
}

#[cfg(unix)]
impl Signals {
    /// Blocks until the first stop request; returns its exit code.
    #[allow(clippy::disallowed_methods, reason = "the main thread waits for the stop request")]
    pub fn wait(&mut self) -> u8 {
        use signal_hook::consts::SIGINT;
        match self.signals.forever().next() {
            Some(SIGINT) => EXIT_SIGINT,
            _ => EXIT_SIGTERM,
        }
    }
}

#[cfg(windows)]
pub use console::{Signals, install};

/// The console's control events, through `SetConsoleCtrlHandler`: the handler runs on a thread Windows makes.
#[cfg(windows)]
mod console {
    use std::io;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::sync::{Mutex, OnceLock, PoisonError};
    use std::thread::park;

    use super::{EXIT_SIGINT, EXIT_SIGTERM};

    const CTRL_C_EVENT: u32 = 0;
    const CTRL_BREAK_EVENT: u32 = 1;
    const CTRL_CLOSE_EVENT: u32 = 2;
    const CTRL_LOGOFF_EVENT: u32 = 5;
    const CTRL_SHUTDOWN_EVENT: u32 = 6;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
        fn GetCurrentProcess() -> isize;
        fn TerminateProcess(process: isize, code: u32) -> i32;
    }

    static STOPPING: AtomicBool = AtomicBool::new(false);
    static SENDER: OnceLock<Mutex<Sender<u8>>> = OnceLock::new();

    /// The handlers, installed.
    #[derive(Debug)]
    pub struct Signals {
        requests: Receiver<u8>,
    }

    /// Installs the handler.
    pub fn install() -> io::Result<Signals> {
        let (sender, requests) = channel();
        if SENDER.set(Mutex::new(sender)).is_err() {
            return Err(io::Error::other("the console handler is installed already"));
        }
        // SAFETY: `handler` is an `extern "system"` function of the signature the API wants and lives for the whole
        // process; the call only registers it.
        let added = unsafe { SetConsoleCtrlHandler(Some(handler), 1) };
        if added == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Signals { requests })
    }

    impl Signals {
        /// Blocks until the first stop request; returns its exit code.
        #[allow(clippy::disallowed_methods, reason = "the main thread waits for the stop request")]
        pub fn wait(&mut self) -> u8 {
            self.requests.recv().unwrap_or(EXIT_SIGTERM)
        }
    }

    unsafe extern "system" fn handler(event: u32) -> i32 {
        let (code, closing) = match event {
            CTRL_C_EVENT | CTRL_BREAK_EVENT => (EXIT_SIGINT, false),
            CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT => (EXIT_SIGTERM, true),
            _ => return 0,
        };
        if STOPPING.swap(true, Ordering::AcqRel) {
            // The second event ends the process at once, from this thread: `exit` would wait for the runtime's
            // locks that the stop in progress may hold.
            // SAFETY: the pseudo handle of the current process is valid for the call, which takes no pointers.
            unsafe { TerminateProcess(GetCurrentProcess(), u32::from(code)) };
            return 1;
        }
        if let Some(sender) = SENDER.get() {
            sender.lock().unwrap_or_else(PoisonError::into_inner).send(code).ok();
        }
        if closing {
            // Windows ends the process when the handler of one of these events returns: stay until `main` exits.
            loop {
                park();
            }
        }
        1
    }
}

//! Holds a model artifact against change while an engine serves it, and makes Euhedral's own writers refuse one that is
//! held.
//!
//! The engine trusts the bytes of the file it serves and checks no checksum (the file must therefore not change under
//! it). [`ArtifactGuard`] is the engine's own handle on the file, opened before its other opens and kept for its whole
//! life: it holds the file's shared lock. On Windows it is also opened without write or delete sharing, so another
//! program's write, rename or delete of the file fails at its open. On Unix the lock is advisory: a program that does
//! not look at it can still write the file (no unprivileged mandatory lock exists), but Euhedral's writers do, through
//! [`WriteClaim`]; a delete only drops the name, and the engine keeps the file it has open.
//!
//! [`WriteClaim`] is the writers' side: before it writes an artifact, a tool takes the exclusive lock on the target,
//! which fails while any engine holds it, and keeps it until the write has ended.

use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

/// Why a file could not be held, or claimed for writing.
#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    /// A running engine holds the file (or, for a guard, a writer has it).
    #[error("{} is held by a running engine or a running writer: stop it before using this file", path.display())]
    Held { path: PathBuf },
    /// The file could not be opened or locked: the file system may not support file locks.
    #[error("cannot lock {}: {source}", path.display())]
    Lock {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// The engine's hold on an artifact file: the shared lock, on a handle of its own, until it is dropped.
#[derive(Debug)]
pub struct ArtifactGuard {
    /// Kept open for the lock; never read.
    _file: File,
}

impl ArtifactGuard {
    /// Holds `path`. Blocking: call it at startup, before the other opens of the file. Two guards of one file coexist.
    /// A file that is not there, or a file system without locks, is refused with the path and the error: the engine
    /// does not run on a file it cannot hold.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: one open and one lock at startup, before any frame runs"
    )]
    pub fn hold(path: &Path) -> Result<ArtifactGuard, GuardError> {
        let lock_error = |source| GuardError::Lock {
            path: path.to_path_buf(),
            source,
        };
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Others may read the file and nothing else: no write handle and no delete or rename while this lives.
            const FILE_SHARE_READ: u32 = 0x1;
            options.share_mode(FILE_SHARE_READ);
        }
        let file = options.open(path).map_err(lock_error)?;
        match file.try_lock_shared() {
            Ok(()) => Ok(ArtifactGuard { _file: file }),
            Err(TryLockError::WouldBlock) => Err(GuardError::Held {
                path: path.to_path_buf(),
            }),
            Err(TryLockError::Error(source)) => Err(lock_error(source)),
        }
    }
}

/// A writer's exclusive lock on the artifact it is about to replace, held until it is dropped.
///
/// On Windows an exclusive lock would also stop the writer's own other handles on the file (it reads and renames the
/// target through them), and the engine's guard already stops a write there at its open; so the claim only proves, at
/// its taking, that no engine holds the file, and holds nothing.
#[derive(Debug)]
pub struct WriteClaim {
    /// The locked file; `None` where there is nothing to hold: the target is not there, or the platform holds nothing.
    _file: Option<File>,
}

impl WriteClaim {
    /// Claims `target` for writing, or refuses with its path when a running engine holds it. A target that is not there
    /// is claimed trivially. A file system without locks admits the claim: no engine can run on it, because the engine
    /// refuses a file it cannot hold.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: one open and one lock before the writer's first frame"
    )]
    pub fn take(target: &Path) -> Result<WriteClaim, GuardError> {
        let file = match OpenOptions::new().read(true).open(target) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(WriteClaim { _file: None }),
            Err(source) => {
                return Err(GuardError::Lock {
                    path: target.to_path_buf(),
                    source,
                });
            }
        };
        match file.try_lock() {
            Ok(()) => Ok(WriteClaim {
                _file: if cfg!(windows) { None } else { Some(file) },
            }),
            Err(TryLockError::WouldBlock) => Err(GuardError::Held {
                path: target.to_path_buf(),
            }),
            Err(TryLockError::Error(error)) if error.kind() == io::ErrorKind::Unsupported => {
                Ok(WriteClaim { _file: None })
            }
            Err(TryLockError::Error(source)) => Err(GuardError::Lock {
                path: target.to_path_buf(),
                source,
            }),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "a test: it makes and removes its own files, and blocks on the calls it checks"
)]
mod tests {
    use std::fs;

    use super::*;

    fn artifact() -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.edrl");
        fs::write(&path, b"bytes").unwrap();
        (directory, path)
    }

    #[test]
    fn a_held_file_refuses_a_writer_and_names_the_path() {
        let (_directory, path) = artifact();
        let _guard = ArtifactGuard::hold(&path).unwrap();
        let refusal = WriteClaim::take(&path).unwrap_err();
        assert!(matches!(refusal, GuardError::Held { .. }));
        assert!(refusal.to_string().contains(&path.display().to_string()));
    }

    #[test]
    fn a_released_file_admits_a_writer() {
        let (_directory, path) = artifact();
        drop(ArtifactGuard::hold(&path).unwrap());
        WriteClaim::take(&path).unwrap();
    }

    #[test]
    fn two_engines_hold_one_file_together() {
        let (_directory, path) = artifact();
        let first = ArtifactGuard::hold(&path).unwrap();
        let second = ArtifactGuard::hold(&path).unwrap();
        drop(first);
        assert!(WriteClaim::take(&path).is_err(), "the second still holds the file");
        drop(second);
        WriteClaim::take(&path).unwrap();
    }

    #[test]
    fn a_file_a_writer_has_claimed_is_not_held_by_an_engine() {
        // On Windows a claim holds nothing, so only the Unix lock is observable.
        if cfg!(windows) {
            return;
        }
        let (_directory, path) = artifact();
        let _claim = WriteClaim::take(&path).unwrap();
        let refusal = ArtifactGuard::hold(&path).unwrap_err();
        assert!(matches!(refusal, GuardError::Held { .. }));
        assert!(refusal.to_string().contains(&path.display().to_string()));
    }

    #[test]
    fn a_missing_file_is_not_held_and_is_claimed_trivially() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("absent.edrl");
        let refusal = ArtifactGuard::hold(&path).unwrap_err();
        assert!(matches!(refusal, GuardError::Lock { .. }));
        assert!(refusal.to_string().contains(&path.display().to_string()));
        WriteClaim::take(&path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_deleted_held_file_stays_readable_through_the_guard_name() {
        use std::io::Read;

        let (_directory, path) = artifact();
        let guard = ArtifactGuard::hold(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let mut text = String::new();
        (&guard._file).read_to_string(&mut text).unwrap();
        assert_eq!(text, "bytes");
    }

    #[cfg(windows)]
    #[test]
    fn a_held_file_refuses_a_delete_a_rename_and_a_write() {
        let (directory, path) = artifact();
        let _guard = ArtifactGuard::hold(&path).unwrap();
        assert!(fs::remove_file(&path).is_err(), "delete");
        assert!(
            fs::rename(&path, directory.path().join("moved.edrl")).is_err(),
            "rename"
        );
        assert!(OpenOptions::new().write(true).open(&path).is_err(), "write");
        assert_eq!(fs::read(&path).unwrap(), b"bytes", "reads stay open");
    }
}

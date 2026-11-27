//! Holds a model artifact against change while an engine serves it, and makes Euhedral's own writers refuse one that is
//! held.
//!
//! The engine trusts the bytes of the file it serves and checks no checksum (the file must therefore not change under
//! it). [`ArtifactGuard`] is the engine's own handle on the file, taken at the start of the engine's build and kept for
//! its whole life: it holds the file's shared lock. Every later open of the artifact is checked against that handle
//! ([`ArtifactGuard::admit`]): a path that now names another file (a rename or replace after the lock) is refused, so the
//! engine never reads a file it does not hold. On Windows it is also opened without write or delete sharing, so another
//! program's write, rename or delete of the file fails at its open. On Unix the lock is advisory: a program that does
//! not look at it can still write the file (no unprivileged mandatory lock exists), but Euhedral's writers do, through
//! [`WriteClaim`]; a delete or rename only moves the name, and the engine keeps the file it holds (its later opens of
//! the name are refused).
//!
//! [`WriteClaim`] is the writers' side: before it writes an artifact, a tool takes the exclusive lock on the target,
//! which fails while any engine holds it, and keeps it until the write has ended.

use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

use same_file::Handle;

/// Why a file could not be held, or claimed for writing.
#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    /// A running engine holds the file (or, for a guard, a writer has it).
    #[error("{} is held by a running engine or a running writer: stop it before using this file", path.display())]
    Held { path: PathBuf },
    /// The engine holds no file of this path: it was asked to admit an open of a file it never held.
    #[error("{} is not a file the engine holds", path.display())]
    NotHeld { path: PathBuf },
    /// The path names another file than the one held: it was renamed over or replaced.
    #[error("{} is not the file the engine holds: it was renamed or replaced while the engine started", path.display())]
    Replaced { path: PathBuf },
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
    /// The path the guard was taken on, which every later open of the artifact names.
    path: PathBuf,
    /// Kept open for the lock; compared with, never read.
    file: File,
}

impl ArtifactGuard {
    /// Holds `path`. Blocking: call it at startup, before the other opens of the file. Two guards of one file coexist.
    /// A file that is not there, or a file system without locks, is refused with the path and the error: the engine
    /// does not run on a file it cannot hold. A path that names another file once the lock is taken (a writer renamed
    /// its output over it between the open and the lock) is refused as [`GuardError::Replaced`].
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
        Self::lock_opened(path, file)
    }

    /// Locks `file`, opened on `path`, and confirms the name still names it: the second half of [`hold`](Self::hold).
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: one open of the name the hold confirms, at startup, before any frame runs"
    )]
    fn lock_opened(path: &Path, file: File) -> Result<ArtifactGuard, GuardError> {
        let lock_error = |source| GuardError::Lock {
            path: path.to_path_buf(),
            source,
        };
        match file.try_lock_shared() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(GuardError::Held {
                    path: path.to_path_buf(),
                });
            }
            Err(TryLockError::Error(source)) => return Err(lock_error(source)),
        }
        let guard = ArtifactGuard {
            path: path.to_path_buf(),
            file,
        };
        // After the lock: a name that was replaced before it took effect names a file nobody holds.
        let named = File::open(path).map_err(lock_error)?;
        guard.admit(&named)?;
        Ok(guard)
    }

    /// The path the guard holds.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Admits `opened`, a handle the engine opened on [`path`](Self::path), when it is the file the guard holds, and
    /// refuses it as [`GuardError::Replaced`] when the name has come to name another file since the lock.
    pub fn admit(&self, opened: &File) -> Result<(), GuardError> {
        let identity = |source| GuardError::Lock {
            path: self.path.clone(),
            source,
        };
        let same = same_file(&self.file, opened).map_err(identity)?;
        match same {
            true => Ok(()),
            false => Err(GuardError::Replaced {
                path: self.path.clone(),
            }),
        }
    }
}

/// Admits `opened`, the engine's open of `path`, against the guard of `path` among `guards`: it must be the file held. A
/// path no guard holds is refused as [`GuardError::NotHeld`].
pub(crate) fn admit_held(guards: &[ArtifactGuard], path: &Path, opened: &File) -> Result<(), GuardError> {
    match guards.iter().find(|guard| guard.path() == path) {
        Some(guard) => guard.admit(opened),
        None => Err(GuardError::NotHeld {
            path: path.to_path_buf(),
        }),
    }
}

/// Whether both handles are the same file: the same device and index (the volume and file index on Windows).
fn same_file(first: &File, second: &File) -> io::Result<bool> {
    let first = Handle::from_file(first.try_clone()?)?;
    let second = Handle::from_file(second.try_clone()?)?;
    Ok(first == second)
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
        let file = match open_for_claim(target) {
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

/// Opens `target` for its exclusive lock. On Unix it is opened for writing where it can be: NFS emulates `flock` with
/// byte-range locks, and an exclusive one needs a handle open for writing. A file the user may not write is opened
/// read-only (the lock is then as the file system gives it, and the write that follows is refused at its own open).
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: one open before the writer's first frame"
)]
fn open_for_claim(target: &Path) -> io::Result<File> {
    #[cfg(unix)]
    match OpenOptions::new().read(true).write(true).open(target) {
        // A directory cannot be opened for writing either: the writer's own open of it says what is wrong.
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::PermissionDenied | io::ErrorKind::IsADirectory
            ) => {}
        opened => return opened,
    }
    OpenOptions::new().read(true).open(target)
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
    fn a_name_renamed_over_between_the_open_and_the_lock_is_refused_by_path() {
        let (directory, path) = artifact();
        let opened = File::open(&path).unwrap();
        let original = File::open(&path).unwrap();
        // A writer replaces the file and releases its claim: the open above is of the old file.
        let replacement = directory.path().join("replacement.edrl");
        fs::write(&replacement, b"other").unwrap();
        fs::rename(&replacement, &path).unwrap();
        let refusal = ArtifactGuard::lock_opened(&path, opened).unwrap_err();
        assert!(matches!(refusal, GuardError::Replaced { .. }));
        assert!(refusal.to_string().contains(&path.display().to_string()));
        // The lock on the original file, not on the replacement, was released.
        original.try_lock().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn an_open_of_a_name_replaced_after_the_hold_is_refused_by_path() {
        let (directory, path) = artifact();
        let guard = ArtifactGuard::hold(&path).unwrap();
        let replacement = directory.path().join("replacement.edrl");
        fs::write(&replacement, b"other").unwrap();
        fs::rename(&replacement, &path).unwrap();
        let refusal = guard.admit(&File::open(&path).unwrap()).unwrap_err();
        assert!(matches!(refusal, GuardError::Replaced { .. }));
        assert!(refusal.to_string().contains(&path.display().to_string()));
    }

    #[test]
    fn an_open_of_the_held_file_by_any_name_is_admitted() {
        let (directory, path) = artifact();
        let guard = ArtifactGuard::hold(&path).unwrap();
        guard.admit(&File::open(&path).unwrap()).unwrap();
        // Another name of the same file (a hard link) is the same file.
        let link = directory.path().join("link.edrl");
        fs::hard_link(&path, &link).unwrap();
        guard.admit(&File::open(&link).unwrap()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_claim_is_a_lock_on_a_handle_open_for_writing() {
        let (_directory, path) = artifact();
        let claim = WriteClaim::take(&path).unwrap();
        let file = claim._file.as_ref().unwrap();
        // The same length: nothing changes, and a handle open for reading only refuses it.
        file.set_len(5).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"bytes");
    }

    #[cfg(unix)]
    #[test]
    fn a_file_the_writer_cannot_write_is_still_claimed() {
        use std::os::unix::fs::PermissionsExt;

        let (_directory, path) = artifact();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o444)).unwrap();
        // A user who may write it anyway (root) never takes the fallback.
        if OpenOptions::new().write(true).open(&path).is_ok() {
            return;
        }
        let _claim = WriteClaim::take(&path).unwrap();
        assert!(matches!(ArtifactGuard::hold(&path), Err(GuardError::Held { .. })));
    }

    #[cfg(unix)]
    #[test]
    fn a_directory_output_is_claimed_through_the_read_only_open() {
        let directory = tempfile::tempdir().unwrap();
        let claim = WriteClaim::take(directory.path()).unwrap();
        assert!(claim._file.is_some());
    }

    #[test]
    fn an_open_of_a_path_no_guard_holds_is_refused_by_path() {
        let (directory, path) = artifact();
        let other = directory.path().join("other.edrl");
        fs::write(&other, b"bytes").unwrap();
        let guards = [ArtifactGuard::hold(&path).unwrap()];
        admit_held(&guards, &path, &File::open(&path).unwrap()).unwrap();
        let refusal = admit_held(&guards, &other, &File::open(&other).unwrap()).unwrap_err();
        assert!(matches!(refusal, GuardError::NotHeld { .. }));
        assert!(refusal.to_string().contains(&other.display().to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn a_deleted_held_file_stays_readable_through_the_guard_name() {
        use std::io::Read;

        let (_directory, path) = artifact();
        let guard = ArtifactGuard::hold(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let mut text = String::new();
        (&guard.file).read_to_string(&mut text).unwrap();
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

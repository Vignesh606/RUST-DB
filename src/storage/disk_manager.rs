//! On-disk page storage for RustDB.
//!
//! ## What this module does
//! [`DiskManager`] owns a single database file treated as a flat array of
//! fixed-size pages. It translates logical page identifiers ([`PageId`]) into
//! byte offsets in that file and performs the raw reads and writes that the
//! rest of the storage engine is built on: the buffer pool will cache pages,
//! and the B+ tree will index them. This is the lowest level of the storage
//! stack — nothing above it should touch the file directly, and nothing below
//! it knows about SQL, transactions, or rows.
//!
//! ## On-disk layout
//! The file is a contiguous run of pages. Page `N` occupies the byte range
//! `[N * PAGE_SIZE, (N + 1) * PAGE_SIZE)`. The page count is derived directly
//! from the file length (`file_len / PAGE_SIZE`), so the file can always be
//! reopened and its extent recovered *without* a header. Pages are allocated
//! strictly at the end of the file (append-only growth), which keeps the
//! invariant `file_len == num_pages * PAGE_SIZE` true at all times.
//!
//! ## Durability
//! Writes reach the OS page cache immediately but are flushed to the storage
//! device only when [`DiskManager::sync`] is called explicitly. Routine
//! [`DiskManager::write_page`] calls are *not* fsynced: durability policy is
//! intentionally left to the write-ahead log (Phase 5), which knows the
//! transactional ordering rules. An fsync per page write would be prohibitively
//! slow; deferring it keeps crash recovery correct without that cost.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Fixed size, in bytes, of every page managed by the storage engine.
///
/// 8 KiB matches PostgreSQL's default block size and is a good trade-off
/// between sequential-scan throughput and the cost of rewriting a dirty page.
pub const PAGE_SIZE: usize = 8192;

/// Logical identifier of a page within a database file.
///
/// A `u32` supports up to 2^32 pages (~35 TiB at 8 KiB pages), which is ample
/// for this engine. Widen to `u64` later if larger databases are needed.
pub type PageId = u32;

/// Errors produced by the storage layer.
#[derive(Debug)]
pub enum StorageError {
    /// An underlying I/O operation failed.
    Io(String),
    /// A page id was referenced that does not exist in the file.
    PageOutOfBounds {
        /// The offending page id.
        page_id: PageId,
        /// How many pages the file actually contains.
        num_pages: u64,
    },
    /// A buffer whose length did not equal [`PAGE_SIZE`] was supplied.
    InvalidPageSize {
        /// The incorrect length that was given.
        actual: usize,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(msg) => write!(f, "storage I/O error: {msg}"),
            StorageError::PageOutOfBounds { page_id, num_pages } => write!(
                f,
                "page {page_id} is out of bounds (file has {num_pages} pages)"
            ),
            StorageError::InvalidPageSize { actual } => write!(
                f,
                "page buffer must be exactly {PAGE_SIZE} bytes, got {actual}"
            ),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::Io(err.to_string())
    }
}

/// Result type for storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;

/// Manages raw, paged access to a single database file.
///
/// All file access is serialized through an internal [`Mutex`]. This guarantees
/// the correctness of the seek-then-read / seek-then-write sequences (which
/// would otherwise race on the shared OS file cursor) and makes `DiskManager`
/// cheaply `Sync` so it can be shared across threads by the buffer pool.
///
/// # Performance
/// The single I/O mutex means only one page read or write can be in flight at a
/// time. For a teaching database this is fine; a production engine would remove
/// it by using positioned I/O (`pread`/`pwrite`, exposed on Unix as
/// [`std::os::unix::fs::FileExt`] and on Windows via overlapped I/O) or an
/// asynchronous backend such as `io_uring`. Both bypass the file cursor and the
/// lock entirely. This is the most likely storage-layer bottleneck under load.
pub struct DiskManager {
    file: Mutex<File>,
    path: PathBuf,
}

impl DiskManager {
    /// Opens the database file at `path`, creating it if it does not exist.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        Ok(DiskManager {
            file: Mutex::new(file),
            path,
        })
    }

    /// Returns the number of pages currently stored in the file.
    ///
    /// Derived from the file length, so it is always consistent with what is on
    /// disk and survives reopen without any in-memory bookkeeping.
    pub fn num_pages(&self) -> Result<u64> {
        let file = self.file.lock().expect("disk manager mutex poisoned");
        let len = file.metadata()?.len();
        Ok(len / PAGE_SIZE as u64)
    }

    /// Allocates a new page at the end of the file (initially zeroed) and
    /// returns its id.
    pub fn allocate_page(&self) -> Result<PageId> {
        let mut file = self.file.lock().expect("disk manager mutex poisoned");
        let new_id = file.metadata()?.len() / PAGE_SIZE as u64;
        // Append a zeroed page to extend the file by exactly PAGE_SIZE bytes.
        let zeros = vec![0u8; PAGE_SIZE];
        file.seek(SeekFrom::End(0))?;
        file.write_all(&zeros)?;
        Ok(new_id as PageId)
    }

    /// Reads the full contents of `page_id` into a freshly owned buffer.
    pub fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        let mut file = self.file.lock().expect("disk manager mutex poisoned");
        let num_pages = file.metadata()?.len() / PAGE_SIZE as u64;
        if page_id as u64 >= num_pages {
            return Err(StorageError::PageOutOfBounds { page_id, num_pages });
        }
        file.seek(SeekFrom::Start(page_id as u64 * PAGE_SIZE as u64))?;
        let mut buf = vec![0u8; PAGE_SIZE];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Writes the full `PAGE_SIZE`-byte contents of `data` to `page_id`.
    ///
    /// The buffer length must equal [`PAGE_SIZE`]. Durability is *not* forced
    /// here; call [`DiskManager::sync`] when a flush to the device is required.
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> Result<()> {
        if data.len() != PAGE_SIZE {
            return Err(StorageError::InvalidPageSize { actual: data.len() });
        }
        let mut file = self.file.lock().expect("disk manager mutex poisoned");
        let num_pages = file.metadata()?.len() / PAGE_SIZE as u64;
        if page_id as u64 >= num_pages {
            return Err(StorageError::PageOutOfBounds { page_id, num_pages });
        }
        file.seek(SeekFrom::Start(page_id as u64 * PAGE_SIZE as u64))?;
        file.write_all(data)?;
        Ok(())
    }

    /// Forces all pending writes to the storage device (data and metadata).
    ///
    /// Called by the WAL / checkpoint code to establish durability boundaries.
    pub fn sync(&self) -> Result<()> {
        let file = self.file.lock().expect("disk manager mutex poisoned");
        file.sync_all()?;
        Ok(())
    }

    /// The path of the underlying database file, for diagnostics and logging.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// Builds a unique temp-file path scoped to the OS temp dir.
    ///
    /// We avoid the `tempfile` crate to keep the build dependency-free and
    /// working offline; uniqueness comes from pid + nanosecond timestamp +
    /// an in-process counter.
    fn temp_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        std::env::temp_dir().join(format!("rustdb_test_{pid}_{ts}_{n}.db"))
    }

    /// RAII guard that deletes a file when dropped, ignoring missing-file errors.
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn write_then_read_roundtrip() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");

        let page = dm.allocate_page().expect("allocate");
        assert_eq!(page, 0);

        let mut data = vec![0u8; PAGE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8; // 251 is prime → varied, non-trivial pattern
        }
        dm.write_page(page, &data).expect("write");

        let read_back = dm.read_page(page).expect("read");
        assert_eq!(read_back, data);
    }

    #[test]
    fn allocate_grows_num_pages() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");

        assert_eq!(dm.num_pages().expect("num_pages"), 0);
        let p0 = dm.allocate_page().expect("allocate");
        let p1 = dm.allocate_page().expect("allocate");
        let p2 = dm.allocate_page().expect("allocate");
        assert_eq!([p0, p1, p2], [0, 1, 2]);
        assert_eq!(dm.num_pages().expect("num_pages"), 3);
    }

    #[test]
    fn read_unallocated_page_errors() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");
        dm.allocate_page().expect("allocate");

        match dm.read_page(5) {
            Err(StorageError::PageOutOfBounds { page_id, .. }) => assert_eq!(page_id, 5),
            other => panic!("expected PageOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn write_wrong_size_errors() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");
        let page = dm.allocate_page().expect("allocate");

        match dm.write_page(page, &[0u8; 42]) {
            Err(StorageError::InvalidPageSize { actual }) => assert_eq!(actual, 42),
            other => panic!("expected InvalidPageSize, got {other:?}"),
        }
    }

    #[test]
    fn multiple_pages_are_independent() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");

        let ids: Vec<PageId> = (0..4).map(|_| dm.allocate_page().expect("allocate")).collect();
        for (i, &id) in ids.iter().enumerate() {
            let mut buf = vec![0u8; PAGE_SIZE];
            buf[0] = i as u8;
            buf[PAGE_SIZE - 1] = (i as u8).wrapping_mul(7);
            dm.write_page(id, &buf).expect("write");
        }
        for (i, &id) in ids.iter().enumerate() {
            let buf = dm.read_page(id).expect("read");
            assert_eq!(buf[0], i as u8);
            assert_eq!(buf[PAGE_SIZE - 1], (i as u8).wrapping_mul(7));
        }
    }

    #[test]
    fn persistence_across_reopen() {
        let path = temp_path();
        let _guard = TempFile(path.clone());

        let data = {
            let dm = DiskManager::open(&path).expect("open");
            let page = dm.allocate_page().expect("allocate");
            let mut data = vec![0u8; PAGE_SIZE];
            data[0] = 0xAB;
            data[PAGE_SIZE - 1] = 0xCD;
            dm.write_page(page, &data).expect("write");
            dm.sync().expect("sync");
            data
        };

        // Reopen a fresh DiskManager over the same file and verify the data.
        let dm = DiskManager::open(&path).expect("reopen");
        assert_eq!(dm.num_pages().expect("num_pages"), 1);
        let read_back = dm.read_page(0).expect("read");
        assert_eq!(read_back, data);
    }

    #[test]
    fn concurrent_access_is_safe() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = Arc::new(DiskManager::open(&path).expect("open"));

        // Pre-allocate a shared set of pages that all threads will hammer.
        const N: PageId = 8;
        for _ in 0..N {
            dm.allocate_page().expect("allocate");
        }

        let mut handles = Vec::new();
        for t in 0..4u8 {
            let dm = Arc::clone(&dm);
            handles.push(std::thread::spawn(move || {
                for i in 0..200u32 {
                    let page = (i % N as u32) as PageId;
                    let mut buf = vec![0u8; PAGE_SIZE];
                    buf[0] = t.wrapping_add(i as u8);
                    buf[1] = page as u8;
                    dm.write_page(page, &buf).expect("write");

                    // The page-mapping byte must always round-trip. Because all
                    // I/O is serialized through the mutex, we never observe a
                    // torn read, so this holds even under contention.
                    let got = dm.read_page(page).expect("read");
                    assert_eq!(got[1], page as u8);
                    assert_eq!(got.len(), PAGE_SIZE);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    #[test]
    fn sync_returns_ok() {
        let path = temp_path();
        let _guard = TempFile(path.clone());
        let dm = DiskManager::open(&path).expect("open");
        dm.allocate_page().expect("allocate");
        dm.sync().expect("sync");
    }
}

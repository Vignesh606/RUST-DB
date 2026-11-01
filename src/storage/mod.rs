//! Storage engine subsystem.
//!
//! Contains the lowest levels of the database: raw paged access to disk
//! ([`disk_manager`]), an LRU [`buffer_pool`] (Phase 1, upcoming), and a B+
//! tree [`btree`] index (Phase 1, upcoming). Everything above this layer —
//! the executor, transactions, and network — ultimately asks the storage
//! engine for pages.

pub mod disk_manager;

pub use disk_manager::{DiskManager, PageId, StorageError, PAGE_SIZE};

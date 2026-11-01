//! RustDB — a from-scratch relational database engine in Rust.
//!
//! The crate is built up incrementally, one subsystem at a time, in this order:
//!
//! 1. **Storage** — paged file I/O, an LRU buffer pool, and a B+ tree index.
//! 2. **SQL** — lexer, recursive-descent parser, and AST.
//! 3. **Planner** — logical plans and a cost-based optimizer.
//! 4. **Executor** — a volcano/iterator execution engine.
//! 5. **Transactions** — write-ahead log, 2PL locking, crash recovery.
//! 6. **Network** — PostgreSQL wire-protocol listener.
//!
//! Each module carries its own documentation explaining what it does and why.

pub mod storage;

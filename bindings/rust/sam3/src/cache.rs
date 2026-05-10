//! Cache configuration and stats types.

/// Cache options used when creating a [`Ctx`](crate::Ctx).
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheOpts {
    /// Number of image feature cache slots. `None` uses the C runtime default.
    pub image_slots: Option<u32>,
    /// Number of text feature cache slots. `None` uses the C runtime default.
    pub text_slots: Option<u32>,
    /// In-memory image cache budget in bytes. `None` uses the C runtime default.
    pub image_mem_budget_bytes: Option<usize>,
}

bitflags::bitflags! {
    /// Cache groups accepted by [`Ctx::cache_clear`](crate::Ctx::cache_clear).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CacheKind: u32 {
        /// Image encoder feature cache.
        const IMAGE = 1 << 0;
        /// Text encoder feature cache.
        const TEXT = 1 << 1;
    }
}

/// Cache hit/miss/eviction counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Image cache hits.
    pub image_hits: u64,
    /// Image cache misses.
    pub image_misses: u64,
    /// Image cache evictions.
    pub image_evictions: u64,
    /// Text cache hits.
    pub text_hits: u64,
    /// Text cache misses.
    pub text_misses: u64,
    /// Text cache evictions.
    pub text_evictions: u64,
}

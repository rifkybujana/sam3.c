use sam3::{CacheKind, CacheOpts, Ctx};

#[test]
fn new_with_cache_accepts_app_slot_counts() {
    let ctx = Ctx::new_with_cache(64, 64).expect("ctx");
    drop(ctx);
}

#[test]
fn new_with_cache_opts_accepts_defaults() {
    let ctx = Ctx::new_with_cache_opts(&CacheOpts::default()).expect("ctx");
    drop(ctx);
}

#[test]
fn cache_stats_fresh_ctx_is_zero() {
    let ctx = Ctx::new().expect("ctx");
    let stats = ctx.cache_stats();
    assert_eq!(stats.image_hits, 0);
    assert_eq!(stats.image_misses, 0);
    assert_eq!(stats.image_evictions, 0);
    assert_eq!(stats.text_hits, 0);
    assert_eq!(stats.text_misses, 0);
    assert_eq!(stats.text_evictions, 0);
}

#[test]
fn cache_clear_on_fresh_ctx_is_noop() {
    let mut ctx = Ctx::new().expect("ctx");
    ctx.cache_clear(CacheKind::empty());
}

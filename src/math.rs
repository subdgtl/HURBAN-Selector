/// Linear interpolation between values `a` and `b` for parameter `f`.
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + f * (b - a)
}

pub fn clamp(x: f32, min: f32, max: f32) -> f32 {
    // FIXME: clamp may eventually be stabilized in std
    // https://github.com/rust-lang/rust/issues/44095
    f32::max(min, f32::min(max, x))
}

/// Exponentially decay `source` to `target` over time. Framerate aware.
///
/// `smoothness` is a floating point number clamped between 0 amd 1.
/// It determines, what fraction of `source` still hasn't decayed towards
/// `target` after 1 second. `smoothness` equal to 0 essentially means
/// `source = target` while 1 means `source = source`
///
/// `delta` is previous frame's processing time in seconds.
///
/// http://www.rorydriscoll.com/2016/03/07/frame-rate-independent-damping-using-lerp/
pub fn decay(source: f32, target: f32, smoothness: f32, delta: f32) -> f32 {
    lerp(
        source,
        target,
        1.0 - clamp(smoothness, 0.0, 1.0).powf(delta),
    )
}

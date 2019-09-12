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
#[allow(dead_code)]
pub fn decay(source: f32, target: f32, smoothness: f32, delta: f32) -> f32 {
    lerp(
        source,
        target,
        1.0 - clamp(smoothness, 0.0, 1.0).powf(delta),
    )
}

/// Linear interpolation between values `source` and `target` for a
/// normalized `weight`.
pub fn lerp(source: f32, target: f32, weight: f32) -> f32 {
    source + weight * (target - source)
}

/// Smooth interpolation based on cubic bezier curve with adjustable
/// control points `p1` and `p2`.
///
/// Port and simplification of https://github.com/gre/bezier-easing
/// and
/// http://greweb.me/2012/02/bezier-curve-based-easing-functions-from-concept-to-implementation/
pub struct CubicBezierEasing {
    p1: [f32; 2],
    p2: [f32; 2],
}

impl CubicBezierEasing {
    /// Create a new cubic bezier curve with control points `p1` and `p2`.
    ///
    /// Control points `p0` and `p3` are [0.0,0.0] and [1.0,1.0],
    /// respectively.
    pub fn new(p1: [f32; 2], p2: [f32; 2]) -> Self {
        let r = 0.0..=1.0;

        assert!(
            r.contains(&p1[0]),
            "P1 must be between [0.0,0.0] and [1.0,1.0]"
        );
        assert!(
            r.contains(&p1[1]),
            "P1 must be between [0.0,0.0] and [1.0,1.0]"
        );
        assert!(
            r.contains(&p2[0]),
            "P2 must be between [0.0,0.0] and [1.0,1.0]"
        );
        assert!(
            r.contains(&p2[1]),
            "P2 must be between [0.0,0.0] and [1.0,1.0]"
        );

        Self { p1, p2 }
    }

    /// Get value for `x` in domain `0.0..1.0` on a cubic bezier
    /// curve.
    ///
    /// `x` is clamped to be between 0 and 1.
    pub fn apply(&self, x: f32) -> f32 {
        let x = clamp(x, 0.0, 1.0);
        if approx::relative_eq!(self.p1[0], self.p1[1])
            && approx::relative_eq!(self.p2[0], self.p2[1])
        {
            // Linear interpolation
            x
        } else {
            let bezier_t = self.compute_t(x);
            Self::compute_bezier(bezier_t, self.p1[1], self.p2[1])
        }
    }

    fn a(a1: f32, a2: f32) -> f32 {
        1.0 - 3.0 * a2 + 3.0 * a1
    }

    fn b(a1: f32, a2: f32) -> f32 {
        3.0 * a2 - 6.0 * a1
    }

    fn c(a1: f32) -> f32 {
        a1 * 3.0
    }

    /// Return x(t) given t, x1, and x2, or y(t) given t, y1, and y2.
    fn compute_bezier(t: f32, a1: f32, a2: f32) -> f32 {
        ((Self::a(a1, a2) * t + Self::b(a1, a2)) * t + Self::c(a1)) * t
    }

    /// Return dx/dt given t, x1, and x2, or dy/dt given t, y1, and y2.
    fn compute_slope(t: f32, a1: f32, a2: f32) -> f32 {
        3.0 * Self::a(a1, a2) * t * t + 2.0 * Self::b(a1, a2) * t + Self::c(a1)
    }

    /// Iteratively find approximation for parameter `t` along a cubic
    /// bezier curve for `x`.
    fn compute_t(&self, x: f32) -> f32 {
        // Newton Raphson iteration
        // https://en.wikipedia.org/wiki/Newton%27s_method

        // The more the prettier, with diminishing returns.
        // 1 iteration already looks very nice
        const N_ITERATIONS: u32 = 4;
        let mut t = x;

        for _ in 0..N_ITERATIONS {
            let slope = Self::compute_slope(t, self.p1[0], self.p2[0]);
            if slope == 0.0 {
                return t;
            }
            let current_x = Self::compute_bezier(t, self.p1[0], self.p2[0]) - x;
            t -= current_x / slope;
        }

        t
    }
}

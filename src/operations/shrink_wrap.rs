use std::iter;
use std::u32;

use crate::geometry::{self, Geometry};

pub struct ShrinkWrapParams<'a> {
    pub geometry: &'a Geometry,
    pub sphere_density: u32,
}

pub fn shrink_wrap(params: ShrinkWrapParams) -> Geometry {
    let (center, radius) = geometry::compute_bounding_sphere(iter::once(params.geometry));
    let mut sphere_geometry = geometry::uv_sphere(
        center.coords.into(),
        radius,
        params.sphere_density,
        params.sphere_density,
    );

    for vertex in sphere_geometry.vertices_mut() {
        if let Some(closest) = geometry::find_closest_point(vertex, &params.geometry) {
            vertex.coords = closest.coords;
        }
    }

    sphere_geometry
}

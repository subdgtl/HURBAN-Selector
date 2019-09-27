use nalgebra as na;
use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;
use std::collections::HashSet;

use crate::geometry::{Geometry, NormalStrategy, TriangleFace};

/// Check if all the vertices of geometry are referenced in geometry's faces
pub fn has_no_orphans(&geo: Geometry) -> bool {
    let mut used_vertices = HashSet::new();
    for face in &geo.faces {
        for vi in &face {
            used_vertices.insert(vi);
        }
    }
    used_vertices.len == &geo.vertices.len
}

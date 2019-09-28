use std::collections::HashSet;

use crate::geometry::Geometry;

/// Check if all the vertices of geometry are referenced in geometry's faces
pub fn has_no_orphans(geo: &Geometry) -> bool {
    let mut used_vertices = HashSet::new();
    for face in geo.triangle_faces_iter() {
        used_vertices.insert(face.vertices.0);
        used_vertices.insert(face.vertices.1);
        used_vertices.insert(face.vertices.2);
    }
    used_vertices.len() == geo.vertices().len()
}

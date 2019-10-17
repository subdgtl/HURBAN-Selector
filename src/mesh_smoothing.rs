use nalgebra::geometry::Point3;

use crate::geometry::{Geometry, Vertices};

pub fn simple_laplacian_smoothing(geometry: Geometry) -> Geometry {
    let vertex_to_vertex_topology = geometry.vertex_to_vertex_topology();
    let geometry_vertices = geometry.vertices();
    let vertices: Vertices = Vec::with_capacity(geometry_vertices.len());
    for (current_vertex_index, neighbors_indices) in vertex_to_vertex_topology.iter() {
        let mut average_position:Point3<f32> = Point3::origin();
        for neighbor_index in neighbors_indices {
            // TODO: unpack
            average_position += geometry_vertices[*neighbor_index] * 1.0 / neighbors_indices.len() as f32;
        }
        vertices.push(average_position);
    }
    Geometry::from_triangle_faces_with_vertices_and_normals(
        geometry.triangle_faces_iter().collect(),
        vertices,
        geometry.normals().to_vec(),
    )
}

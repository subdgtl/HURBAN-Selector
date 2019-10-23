use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::{Face, Geometry, TriangleFace};
use crate::mesh_topology_analysis::face_to_face_topology;

fn crawl_faces(
    current_face_index: u32,
    face_to_face: &HashMap<usize, SmallVec<[usize; 8]>>,
    available_faces: &mut Vec<u32>,
    separate_faces: &mut Vec<u32>,
) {
    separate_faces.push(current_face_index);

    if let Some(all_neighbors) = face_to_face.get(&cast_usize(current_face_index)) {
        let neighbors: Vec<u32> = all_neighbors
            .iter()
            .filter(|n| available_faces.iter().any(|a_f| *a_f == cast_u32(**n)))
            .map(|n| cast_u32(*n))
            .collect();
        let af: Vec<_> = available_faces
            .iter()
            .filter(|f| neighbors.iter().all(|n| n != *f))
            .map(|f| *f)
            .collect();
        *available_faces = af;
        for neighbor_index in neighbors {
            crawl_faces(
                neighbor_index,
                face_to_face,
                available_faces,
                separate_faces,
            );
        }
    }
}

/// #Split mesh into isolated geometries
/// Crawls the geometry to find continuous patches of geometry.
/// Returns a vector of new separated geometries.
#[allow(dead_code)]
pub fn separate_isolated_meshes(geometry: &Geometry) -> Vec<Geometry> {
    let face_to_face = face_to_face_topology(geometry);
    let triangular_faces: Vec<TriangleFace> = geometry
        .faces()
        .iter()
        .copied()
        .map(|face| match face {
            Face::Triangle(f) => f,
        })
        .collect();
    let vertices: Vec<_> = geometry.vertices().to_vec();
    let normals: Vec<_> = geometry.normals().to_vec();

    let mut available_faces: Vec<u32> = face_to_face.keys().map(|key| cast_u32(*key)).collect();

    let mut patches: Vec<Geometry> = Vec::new();

    while !available_faces.is_empty() {
        let mut separate_faces: Vec<u32> = Vec::new();

        if let Some(first_face_index) = available_faces.pop() {
            crawl_faces(
                first_face_index,
                &face_to_face,
                &mut available_faces,
                &mut separate_faces,
            );
        }
        patches.push(
            Geometry::from_triangle_faces_with_vertices_and_normals_remove_orphans(
                separate_faces
                    .iter()
                    .map(|face_index| triangular_faces[cast_usize(*face_index)])
                    .collect(),
                vertices.clone(),
                normals.clone(),
            ),
        );
    }

    patches
}

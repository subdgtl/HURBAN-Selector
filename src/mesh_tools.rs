use std::collections::{HashMap, HashSet};

use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;
use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::{Face, Geometry, TriangleFace};
use crate::mesh_topology_analysis::face_to_face_topology;

/// Weld similar (close enough) vertices into one and reuse such vertices in
/// connected faces
///
/// Weld is used to actually connect faces, which are connected merely visually
/// (often resulting in a watertight mesh) or to reduce number of vertices in
/// case when vertices are multiplied because vertices at the same coordinates
/// are not referenced by more faces but rather each face references its own
/// copy of the vertex.
///
/// Weld is one of the auto-fixes leading to a simplified, watertight or
/// true-to-its-genus mesh geometries.
pub fn weld(geometry: &Geometry, tolerance: f32) -> Geometry {
    // key = rounded vertex position with a tolerance (it's expected that the
    // same value will be shared by more close vertices)
    // value = actual positions of close vertices
    let mut vertex_proximity_map: HashMap<(u32, u32, u32), SmallVec<[usize; 8]>> = HashMap::new();
    for (current_vertex_index, vertex) in geometry.vertices().iter().enumerate() {
        let vertex_with_tolerance = (
            (vertex.x / tolerance).floor() as u32,
            (vertex.y / tolerance).floor() as u32,
            (vertex.z / tolerance).floor() as u32,
        );
        let close_vertices = vertex_proximity_map
            .entry(vertex_with_tolerance)
            .or_insert_with(SmallVec::new);
        close_vertices.push(current_vertex_index);
    }

    // All vertices sorted into clusters of positionally close items. These will
    // be later averaged into a single vertex
    let close_vertex_clusters = vertex_proximity_map.values();

    // key = original vertex index value = new (averaged) vertex index It is
    // expected that more keys will share the same value; more original vertices
    // will be replaced by a single averaged vertex
    let mut old_new_vertex_map: HashMap<usize, usize> = HashMap::new();
    for (new_vertex_index, old_vertex_indices) in close_vertex_clusters.clone().enumerate() {
        for old_vertex_index in old_vertex_indices {
            old_new_vertex_map.insert(*old_vertex_index, new_vertex_index);
        }
    }

    // Vertices of the new geometry averaged from the clusters of original
    // vertices
    let new_vertices = close_vertex_clusters.map(|old_vertex_indices| {
        old_vertex_indices
            .iter()
            .fold(Point3::origin(), |summed: Point3<f32>, old_vertex_index| {
                summed + geometry.vertices()[*old_vertex_index].coords
            })
            / old_vertex_indices.len() as f32
    });

    // New faces with renumbered vertex (and normal) indices. Some faces might
    // end up invalid (not referencing three distinct vertices). Those will be
    // removed as they don't affect the visual appearance of the mesh geometry.
    let new_faces = geometry
        .faces()
        .iter()
        .filter_map(|old_face| match old_face {
            Face::Triangle(f) => Some(Face::Triangle(TriangleFace::new(
                cast_u32(
                    *old_new_vertex_map
                        .get(&cast_usize(f.vertices.0))
                        .expect("Referencing non-existent vertex"),
                ),
                cast_u32(
                    *old_new_vertex_map
                        .get(&cast_usize(f.vertices.1))
                        .expect("Referencing non-existent vertex"),
                ),
                cast_u32(
                    *old_new_vertex_map
                        .get(&cast_usize(f.vertices.2))
                        .expect("Referencing non-existent vertex"),
                ),
            ))),
        })
        .filter(|new_face| match new_face {
            Face::Triangle(f) => f.vertices.0 != f.vertices.1 && f.vertices.0 != f.vertices.2,
        });

    // key = old vertex index
    // value = indices of all old normals being referenced by faces together
    // with the vertex
    //
    // The faces can reference vertices and normals in different ways. While the
    // vertices will be averaged using a straight-forward logic, it is unclear
    // which normals should be averaged to be matched with the new vertices.
    // Therefore it's important to collect all the normals associated with the
    // original vertices in clusters and averaging those.
    let mut old_vertex_normals_index_map: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();
    for face in geometry.faces() {
        match face {
            Face::Triangle(f) => {
                let vertex_indices: SmallVec<_> = SmallVec::from_buf([
                    (f.vertices.0, f.normals.0),
                    (f.vertices.1, f.normals.1),
                    (f.vertices.2, f.normals.2),
                ]);
                for (vertex_index, normal_index) in vertex_indices {
                    let associated_normals = &mut old_vertex_normals_index_map
                        .entry(vertex_index)
                        .or_insert_with(SmallVec::new);
                    if associated_normals
                        .iter()
                        .all(|value| *value != normal_index)
                    {
                        associated_normals.push(normal_index);
                    }
                }
            }
        }
    }

    // Associate old normals to the new averaged vertices
    let mut new_vertex_old_normals_index_map: Vec<SmallVec<[u32; 8]>> =
        vec![SmallVec::new(); new_vertices.len()];
    for (old_vertex_index, old_normals_indices) in old_vertex_normals_index_map {
        if let Some(new_vertex_index) = old_new_vertex_map.get(&cast_usize(old_vertex_index)) {
            new_vertex_old_normals_index_map[*new_vertex_index]
                .extend_from_slice(&old_normals_indices);
        };
    }

    // Calculate an average normal for each new (averaged) vertex
    let new_normals: Vec<Vector3<f32>> = new_vertex_old_normals_index_map
        .iter()
        .map(|old_normals_indices| {
            old_normals_indices
                .iter()
                .fold(Vector3::zeros(), |avg, o_n_i| {
                    avg + geometry.normals()[cast_usize(*o_n_i)]
                })
                / old_normals_indices.len() as f32
        })
        .collect();

    Geometry::from_faces_with_vertices_and_normals(new_faces, new_vertices, new_normals)
}

/// Crawls the geometry to find continuous patches of geometry.
/// Returns a vector of new separated geometries.
pub fn separate_isolated_meshes(geometry: &Geometry) -> Vec<Geometry> {
    let face_to_face = face_to_face_topology(geometry);
    let mut available_face_indices: HashSet<u32> = face_to_face.keys().copied().collect();
    let mut patches: Vec<Geometry> = Vec::new();

    while let Some(first_face_index) = available_face_indices.iter().next() {
        let connected_indices = crawl_faces(*first_face_index, &face_to_face);

        patches.push(
            Geometry::from_faces_with_vertices_and_normals_remove_orphans(
                connected_indices
                    .iter()
                    .map(|face_index| geometry.faces()[cast_usize(*face_index)]),
                geometry.vertices().to_vec(),
                geometry.normals().to_vec(),
            ),
        );

        for c in &connected_indices {
            available_face_indices.remove(c);
        }
    }

    patches
}

fn crawl_faces(
    start_face_index: u32,
    face_to_face: &HashMap<u32, SmallVec<[u32; 8]>>,
) -> HashSet<u32> {
    let mut index_stack = vec![start_face_index];
    index_stack.push(start_face_index);

    let mut connected_face_indices = HashSet::new();

    while let Some(current_face_index) = index_stack.pop() {
        if connected_face_indices.insert(current_face_index) {
            for neighbor in &face_to_face[&current_face_index] {
                index_stack.push(neighbor.clone());
            }
        }
    }

    connected_face_indices.shrink_to_fit();

    connected_face_indices
}

#[cfg(test)]
mod tests {
    use nalgebra::base::Vector3;
    use nalgebra::geometry::Point3;

    use crate::geometry::{self, Geometry, TriangleFace};
    use crate::mesh_analysis;

    use super::*;

    fn n(x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(x, y, z)
    }

    fn tessellated_triangle_geometry_after_welding() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let vertex_normals = vec![
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
        ];

        let faces = vec![
            TriangleFace::new_separate(0, 1, 3, 0, 1, 3),
            TriangleFace::new_separate(1, 4, 3, 1, 4, 3),
            TriangleFace::new_separate(1, 2, 4, 1, 2, 4),
            TriangleFace::new_separate(3, 4, 5, 3, 4, 5),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn tessellated_triangle_geometry_for_welding() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0), //0, 0
            Point3::new(0.0, -2.0, 0.0),  //1, 1
            Point3::new(-1.0, 0.0, 0.0),  //3, 2
            Point3::new(0.0, -2.0, 0.0),  //1, 3
            Point3::new(2.0, -2.0, 0.0),  //2, 4
            Point3::new(1.0, 0.0, 0.0),   //4, 5
            Point3::new(1.0, 0.0, 0.0),   //4, 6
            Point3::new(-1.0, 0.0, 0.0),  //3, 7
            Point3::new(0.0, -2.0, 0.0),  //1, 8
            Point3::new(-1.0, 0.0, 0.0),  //3, 9
            Point3::new(1.0, 0.0, 0.0),   //4, 10
            Point3::new(0.0, 2.0, 0.0),   //5, 11
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
            TriangleFace::new_separate(3, 4, 5, 0, 0, 0),
            TriangleFace::new_separate(6, 7, 8, 0, 0, 0),
            TriangleFace::new_separate(9, 10, 11, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn tessellated_triangle_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 3, 1, 0, 0, 0),
            TriangleFace::new_separate(1, 3, 4, 0, 0, 0),
            TriangleFace::new_separate(1, 4, 2, 0, 0, 0),
            TriangleFace::new_separate(3, 5, 4, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn tessellated_triangle_with_island_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 3, 1, 0, 0, 0),
            TriangleFace::new_separate(1, 3, 4, 0, 0, 0),
            TriangleFace::new_separate(1, 4, 2, 0, 0, 0),
            TriangleFace::new_separate(3, 5, 4, 0, 0, 0),
            TriangleFace::new_separate(6, 7, 8, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn triangular_island_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![TriangleFace::new_separate(0, 1, 2, 0, 0, 0)];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_tessellated_triangle() {
        let geometry = tessellated_triangle_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);

        assert!(mesh_analysis::are_similar(
            &calculated_geometries[0],
            &geometry
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_cube() {
        let geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);
        assert!(mesh_analysis::are_similar(
            &geometry,
            &calculated_geometries[0]
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_tessellated_triangle_with_island() {
        let geometry = tessellated_triangle_with_island_geometry();
        let geometry_triangle_correct = tessellated_triangle_geometry();
        let geometry_island_correct = triangular_island_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 2);

        if mesh_analysis::are_similar(&calculated_geometries[0], &geometry_triangle_correct) {
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[1],
                &geometry_island_correct
            ));
        } else {
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[1],
                &geometry_triangle_correct
            ));
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[0],
                &geometry_island_correct
            ));
        }
    }

    // FIXME: test on more geometries, check with updated visual similarity
    // comparators
    #[test]
    fn test_weld_tesselated_triangle() {
        let geometry = tessellated_triangle_geometry_for_welding();
        let geometry_after_welding_correct = tessellated_triangle_geometry_after_welding();

        // TODO: Vertex order fails
        let geometry_after_welding = weld(&geometry, 0.1);

        assert!(mesh_analysis::are_visually_similar(
            &geometry_after_welding_correct,
            &geometry_after_welding
        ));
    }
}
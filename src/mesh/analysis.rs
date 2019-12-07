use nalgebra as na;
use nalgebra::geometry::Point3;

use super::Mesh;

pub fn compute_bounding_sphere<'a, I>(meshes: I) -> (Point3<f32>, f32)
where
    I: IntoIterator<Item = &'a Mesh> + Clone,
{
    let centroid = compute_centroid(meshes.clone());
    let mut max_distance_squared = 0.0;

    for mesh in meshes {
        for vertex in &mesh.vertices {
            let distance_squared = na::distance_squared(&centroid, vertex);
            if distance_squared > max_distance_squared {
                max_distance_squared = distance_squared;
            }
        }
    }

    (centroid, max_distance_squared.sqrt())
}

pub fn compute_centroid<'a, I>(meshes: I) -> Point3<f32>
where
    I: IntoIterator<Item = &'a Mesh>,
{
    let mut vertex_count = 0;
    let mut centroid = Point3::origin();
    for mesh in meshes {
        vertex_count += mesh.vertices.len();
        for vertex in &mesh.vertices {
            let v = vertex - Point3::origin();
            centroid += v;
        }
    }

    centroid / (vertex_count as f32)
}

// FIXME: deprecate and remove, superseded by pull point to mesh
pub fn find_closest_point(position: &Point3<f32>, mesh: &Mesh) -> Option<Point3<f32>> {
    let vertices = mesh.vertices();
    if vertices.is_empty() {
        return None;
    }

    let mut closest = vertices[0];
    let mut closest_distance_squared = na::distance_squared(position, &closest);
    for point in &vertices[1..] {
        let distance_squared = na::distance_squared(position, &point);
        if distance_squared < closest_distance_squared {
            closest = *point;
            closest_distance_squared = distance_squared;
        }
    }

    Some(closest)
}

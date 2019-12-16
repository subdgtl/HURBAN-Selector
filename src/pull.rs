use std::f32;

use nalgebra;
use nalgebra::{Point3, Rotation3, Vector3};

use crate::convert::cast_usize;
use crate::geometry;
use crate::mesh::{Face, Mesh, UnorientedEdge};
use crate::plane::Plane;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulledPointWithDistance {
    pub point: Point3<f32>,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointOnLine {
    pub clamped: Point3<f32>,
    pub unclamped: Point3<f32>,
}

/// Pulls arbitrary point to an arbitrary line.
///
/// Returns both, a clamped point on the line and an unclamped points on an
/// endless ray.
///
/// https://stackoverflow.com/questions/3120357/get-closest-point-to-a-line
pub fn pull_point_to_line(
    point: &Point3<f32>,
    line_start: &Point3<f32>,
    line_end: &Point3<f32>,
) -> PointOnLine {
    if is_point_on_line_clamped(point, line_start, line_end) {
        return PointOnLine {
            clamped: *point,
            unclamped: *point,
        };
    }
    let start_to_point = point - line_start;
    let start_to_end = line_end - line_start;

    let square_magnitude_start_to_end = nalgebra::distance_squared(line_start, line_end);
    let start_to_point_dot_start_to_end = start_to_point.dot(&start_to_end);
    let line_parameter = start_to_point_dot_start_to_end / square_magnitude_start_to_end;

    let closest_point_on_line_unclamped =
        Point3::from(line_start.coords.lerp(&line_end.coords, line_parameter));
    let closest_point_on_line_clamped = if line_parameter < 0.0 {
        *line_start
    } else if line_parameter > 1.0 {
        *line_end
    } else {
        closest_point_on_line_unclamped
    };

    PointOnLine {
        clamped: closest_point_on_line_clamped,
        unclamped: closest_point_on_line_unclamped,
    }
}

/// Pulls arbitrary point to the closest point of a plane.
///
/// Cast a ray from the point perpendicular to the plane and calculate their
/// intersection.
#[allow(dead_code)]
pub fn pull_point_to_plane(point: &Point3<f32>, plane: &Plane) -> PulledPointWithDistance {
    if plane.contains_point(point) {
        PulledPointWithDistance {
            point: *point,
            distance: 0.0,
        }
    } else {
        ray_intersects_plane(point, &plane.normal(), plane)
            .expect("The normal is parallel to its plane")
    }
}

/// Pulls arbitrary point to the closest point of a mesh geometry.
///
/// Cast a ray from the point perpendicular to each mesh face and if there is an
/// intersection, add it to the list. Also measure a distance to the closest
/// point on each mesh geometry edge. Pick the closest point of them all as the
/// pulled point.
#[allow(dead_code)]
pub fn pull_point_to_mesh(
    point: &Point3<f32>,
    mesh: &Mesh,
    unoriented_edges: &[UnorientedEdge],
) -> PulledPointWithDistance {
    let vertices = mesh.vertices();
    let all_mesh_faces_with_normals = mesh.faces().iter().map(|Face::Triangle(t_f)| {
        let face_vertices = (
            &vertices[cast_usize(t_f.vertices.0)],
            &vertices[cast_usize(t_f.vertices.1)],
            &vertices[cast_usize(t_f.vertices.2)],
        );
        (
            face_vertices,
            geometry::compute_triangle_normal(face_vertices.0, face_vertices.1, face_vertices.2),
        )
    });

    let pulled_identity = PulledPointWithDistance {
        point: *point,
        distance: 0.0,
    };

    let mut pulled_points: Vec<PulledPointWithDistance> = Vec::new();
    // Pull to faces
    for (face_vertices, face_normal) in all_mesh_faces_with_normals {
        // If the point already lies in the face, it means it's already puled to the mesh.
        if is_point_in_triangle(point, face_vertices.0, face_vertices.1, face_vertices.2) {
            return pulled_identity;
        }
        // If triangle vertices are collinear, it's enough to pull to
        // triangle edges later on.
        if !geometry::are_points_collinear(face_vertices.0, face_vertices.1, face_vertices.2) {
            if let Some(intersection_point) = ray_intersects_triangle(
                point,
                &(-1.0 * face_normal),
                face_vertices.0,
                face_vertices.1,
                face_vertices.2,
            ) {
                pulled_points.push(intersection_point);
            }
        }
    }

    // Pull to edges
    for u_e in unoriented_edges {
        let closest_point = pull_point_to_line(
            point,
            &vertices[cast_usize(u_e.0.vertices.0)],
            &vertices[cast_usize(u_e.0.vertices.1)],
        );
        // If the point already lies in the edge, it means it's already puled to the mesh.
        if closest_point.clamped == *point {
            return pulled_identity;
        }
        pulled_points.push(PulledPointWithDistance {
            point: closest_point.clamped,
            distance: nalgebra::distance(point, &closest_point.clamped),
        });
    }

    let mut closest_pulled_point = pulled_points.pop().expect("No pulled point found");
    for current_pulled_point in pulled_points {
        if current_pulled_point.distance < closest_pulled_point.distance {
            closest_pulled_point = current_pulled_point;
        }
    }

    closest_pulled_point
}

/// Checks if a point lies in a triangle.
///
/// #Panics
/// Panics if triangle is collinear
///
/// https://math.stackexchange.com/questions/4322/check-whether-a-point-is-within-a-3d-triangle
fn is_point_in_triangle(
    point: &Point3<f32>,
    triangle_vertex0: &Point3<f32>,
    triangle_vertex1: &Point3<f32>,
    triangle_vertex2: &Point3<f32>,
) -> bool {
    // If the triangle is degenerated into a point, check if the point is equal
    // to the test point.
    if triangle_vertex0
        .coords
        .relative_eq(&triangle_vertex1.coords, 0.001, 0.001)
        && triangle_vertex0
            .coords
            .relative_eq(&triangle_vertex2.coords, 0.001, 0.001)
    {
        return point == triangle_vertex0;
    }

    // If the triangle is degenerated into a line, check if the point lies on
    // the line.
    if geometry::are_points_collinear(triangle_vertex0, triangle_vertex1, triangle_vertex2) {
        let dist01 = nalgebra::distance_squared(triangle_vertex0, triangle_vertex1);
        let dist02 = nalgebra::distance_squared(triangle_vertex0, triangle_vertex2);
        let dist12 = nalgebra::distance_squared(triangle_vertex1, triangle_vertex2);

        // Get the longest span of the three collinear points
        let (a, b) = if dist01 > dist02 && dist01 > dist12 {
            (triangle_vertex0, triangle_vertex1)
        } else if dist02 > dist01 && dist02 > dist12 {
            (triangle_vertex0, triangle_vertex2)
        } else {
            (triangle_vertex1, triangle_vertex2)
        };

        return is_point_on_line_clamped(point, a, b);
    }

    let plane = Plane::from_three_points(triangle_vertex0, triangle_vertex1, triangle_vertex2);

    // If the point is not on the triangle plane, it's also not on the triangle.
    if !plane.contains_point(&point) {
        return false;
    }

    let (horizontal_vertex0, horizontal_vertex1, horizontal_vertex2, horizontal_point) =
        if approx::relative_eq!(triangle_vertex0.z, triangle_vertex1.z)
            && approx::relative_eq!(triangle_vertex0.z, triangle_vertex2.z)
        {
            // In case the triangle already is horizontal, use the original
            // coordinates.
            (
                *triangle_vertex0,
                *triangle_vertex1,
                *triangle_vertex2,
                *point,
            )
        } else if let Some(rotation_to_horizontal) =
            Rotation3::rotation_between(&plane.normal(), &Vector3::new(0.0, 0.0, 1.0))
        {
            // In case the triangle isn't horizontal rotate the triangle to
            // become horizontal and rotate also the test point.
            (
                rotation_to_horizontal * triangle_vertex0,
                rotation_to_horizontal * triangle_vertex1,
                rotation_to_horizontal * triangle_vertex2,
                rotation_to_horizontal * point,
            )
        } else if let Some(rotation_to_horizontal) =
            Rotation3::rotation_between(&(-1.0 * plane.normal()), &Vector3::new(0.0, 0.0, 1.0))
        {
            // In case the triangle isn't horizontal rotate the triangle to
            // become horizontal and rotate also the test point.
            (
                rotation_to_horizontal * triangle_vertex0,
                rotation_to_horizontal * triangle_vertex1,
                rotation_to_horizontal * triangle_vertex2,
                rotation_to_horizontal * point,
            )
        } else {
            // The original triangle is not horizontal and it's not possible to
            // rotate it to become horizontal. This case should never happen but
            // if it does, there is currently no way to check if the point is
            // inside the triangle, therefore assume it's not. Early return
            // false because the following calculations won't produce valid
            // results.
            return false;
        };

    let barycentric_point = geometry::compute_barycentric_coords(
        horizontal_vertex0.xy(),
        horizontal_vertex1.xy(),
        horizontal_vertex2.xy(),
        horizontal_point.xy(),
    )
    .expect("Failed to calculate barycentric coords");
    barycentric_point.x >= 0.0
        && barycentric_point.x <= 1.0
        && barycentric_point.y >= 0.0
        && barycentric_point.y <= 1.0
        && barycentric_point.z >= 0.0
        && barycentric_point.z <= 1.0
        && approx::relative_eq!(
            barycentric_point.x + barycentric_point.y + barycentric_point.z,
            1.0
        )
}

/// The Möller–Trumbore ray-triangle intersection algorithm is a fast method for
/// calculating the intersection of a ray and a triangle in three dimensions
/// without the need of precomputation of the plane equation of the plane
/// containing the triangle.
///
/// #Panics
/// Panics if ray vector is zero.
///
/// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
/// http://webserver2.tecgraf.puc-rio.br/~mgattass/cg/trbRR/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
fn ray_intersects_triangle(
    ray_origin: &Point3<f32>,
    ray_vector: &Vector3<f32>,
    triangle_vertex0: &Point3<f32>,
    triangle_vertex1: &Point3<f32>,
    triangle_vertex2: &Point3<f32>,
) -> Option<PulledPointWithDistance> {
    assert!(ray_vector != &Vector3::zeros(), "Ray vector zero");

    let ray_vector_normalized = ray_vector.normalize();
    let edge_1_vector = triangle_vertex1 - triangle_vertex0;
    let edge_2_vector = triangle_vertex2 - triangle_vertex0;
    // If the ray is parallel to the triangle, a vector perpendicular to the ray
    // and one of the triangle edges
    let perpendicular_vector = ray_vector_normalized.cross(&edge_2_vector);
    // will be also perpendicular to the other triangle edge.
    let determinant = edge_1_vector.dot(&perpendicular_vector);
    // Which means the ray is parallel to the tested triangle.
    if approx::relative_eq!(determinant, 0.0) {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let tangent_vector = ray_origin - triangle_vertex0;
    let u_parameter = inverse_determinant * tangent_vector.dot(&perpendicular_vector);
    // The ray intersects the triangle plane outside of the triangle -> the ray
    // doesn't intersect the triangle
    if u_parameter < 0.0 || u_parameter > 1.0 {
        return None;
    }
    let q_vector = tangent_vector.cross(&edge_1_vector);
    let v_parameter = inverse_determinant * ray_vector_normalized.dot(&q_vector);
    // The ray intersects the triangle plane outside of the triangle -> the ray
    // doesn't intersect the triangle
    if v_parameter < 0.0 || u_parameter + v_parameter > 1.0 {
        return None;
    }
    // The t_parameter is the relative position of the intersection point on the
    // ray line.
    let t_parameter = inverse_determinant * edge_2_vector.dot(&q_vector);
    // Ray-triangle intersection
    if t_parameter > f32::EPSILON && t_parameter < 1.0 / f32::EPSILON {
        Some(PulledPointWithDistance {
            point: ray_origin + ray_vector_normalized.scale(t_parameter),
            distance: t_parameter.abs(),
        })
    } else {
        None
    }
}

/// Find an intersection or a ray and a plane.
///
/// #Panics
/// Panics if ray vector is zero.
///
/// https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Rust
/// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
fn ray_intersects_plane(
    ray_origin: &Point3<f32>,
    ray_vector: &Vector3<f32>,
    plane: &Plane,
) -> Option<PulledPointWithDistance> {
    assert!(ray_vector != &Vector3::zeros(), "Ray vector zero");

    let plane_normal = plane.normal();
    let ray_vector_normalized = ray_vector.normalize();
    let denominator = ray_vector_normalized.dot(&plane_normal);
    // The ray is parallel to the plane
    if approx::relative_eq!(denominator, 0.0) {
        None
    } else {
        let ray_to_plane_origin_vector = ray_origin - plane.origin();
        let t_parameter = ray_to_plane_origin_vector.dot(&plane_normal) / denominator;
        Some(PulledPointWithDistance {
            point: ray_origin - ray_vector_normalized.scale(t_parameter),
            distance: t_parameter.abs(),
        })
    }
}

/// Checks if a point lies on an infinite line.
fn is_point_on_line_unclamped(
    point: &Point3<f32>,
    line_start: &Point3<f32>,
    line_end: &Point3<f32>,
) -> bool {
    let line_start_to_point = point - line_start;
    let line_start_to_end = line_end - line_start;
    point == line_start
        || point == line_end
        || approx::relative_eq!(
            line_start_to_point
                .normalize()
                .dot(&line_start_to_end.normalize())
                .abs(),
            1.0
        )
}

/// Checks if a point lies on a line between two points.
fn is_point_on_line_clamped(
    point: &Point3<f32>,
    line_start: &Point3<f32>,
    line_end: &Point3<f32>,
) -> bool {
    let line_length_squared = nalgebra::distance_squared(line_start, line_end);
    point == line_start
        || point == line_end
        || is_point_on_line_unclamped(point, line_start, line_end)
            && nalgebra::distance_squared(line_start, point) <= line_length_squared
            && nalgebra::distance_squared(line_end, point) <= line_length_squared
}

#[cfg(test)]
mod tests {
    use crate::mesh::primitive;

    use super::*;

    #[test]
    fn test_pull_point_to_line_for_point_pulled_to_center() {
        let line_start = &Point3::new(-1.0, 0.0, 0.0);
        let line_end = &Point3::new(1.0, 0.0, 0.0);
        let test_point = Point3::new(0.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(0.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(0.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_start, line_end);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_pull_point_to_line_for_point_pulled_below_start() {
        let line_start = &Point3::new(-1.0, 0.0, 0.0);
        let line_end = &Point3::new(1.0, 0.0, 0.0);
        let test_point = Point3::new(-2.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(-1.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(-2.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_start, line_end);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_pull_point_to_line_for_point_pulled_beyond_end() {
        let line_start = &Point3::new(-1.0, 0.0, 0.0);
        let line_end = &Point3::new(1.0, 0.0, 0.0);
        let test_point = Point3::new(2.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(1.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(2.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_start, line_end);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_pull_point_to_line_for_point_on_the_line() {
        let line_start = &Point3::new(-1.0, 0.0, 0.0);
        let line_end = &Point3::new(1.0, 0.0, 0.0);
        let test_point = Point3::new(0.0, 0.0, 0.0);

        let point_on_line_clamped_correct = Point3::new(0.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(0.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_start, line_end);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_pull_point_to_plane_for_point_on_plane_and_horizontal_plane() {
        let test_point: Point3<f32> = Point3::origin();
        let test_plane =
            Plane::from_origin_and_normal(&Point3::origin(), &Vector3::new(0.0, 0.0, 1.0));

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert_eq!(point_pulled_to_plane_calculated.point, test_point);
        assert_eq!(point_pulled_to_plane_calculated.distance, 0.0);
    }

    #[test]
    fn test_pull_point_to_plane_for_point_on_plane_and_vertical_plane() {
        let test_point: Point3<f32> = Point3::origin();
        let test_plane =
            Plane::from_origin_and_normal(&Point3::origin(), &Vector3::new(1.0, 0.0, 0.0));

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert_eq!(point_pulled_to_plane_calculated.point, test_point);
        assert_eq!(point_pulled_to_plane_calculated.distance, 0.0);
    }

    #[test]
    fn test_pull_point_to_plane_for_point_above_plane_and_horizontal_plane() {
        let test_point = Point3::new(0.0, 0.0, 1.0);
        let test_plane =
            Plane::from_origin_and_normal(&Point3::origin(), &Vector3::new(0.0, 0.0, 1.0));

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert_eq!(
            point_pulled_to_plane_calculated.point,
            Point3::new(0.0, 0.0, 0.0)
        );
        assert_eq!(point_pulled_to_plane_calculated.distance, 1.0);
    }

    #[test]
    fn test_pull_point_to_plane_for_arbitrary_point_and_arbitrary_plane() {
        let test_point = Point3::new(2.5, 4.2, 1.8);
        let test_plane = Plane::from_origin_and_normal(&Point3::origin(), &test_point.coords);

        let distance_correct = nalgebra::distance(&test_point, &Point3::origin());

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert!(point_pulled_to_plane_calculated.point.coords.relative_eq(
            &Vector3::new(0.0, 0.0, 0.0),
            0.001,
            0.001,
        ));
        assert_eq!(point_pulled_to_plane_calculated.distance, distance_correct);
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_inside_left() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(-0.25, 0.0, 0.0);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        let point_on_mesh_correct = Point3::new(-1.0, 0.0, 0.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert!(approx::relative_eq!(
            pulled_point_on_mesh_calculated.point,
            point_on_mesh_correct,
        ));
        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_inside_top_front_right() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(0.25, 0.25, 0.25);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        // any of the following points on mesh would be correct
        let points_on_mesh_correct = vec![
            Point3::new(1.0, 0.25, 0.25),
            Point3::new(0.25, 1.0, 0.25),
            Point3::new(0.25, 0.25, 1.0),
        ];
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert!(points_on_mesh_correct
            .iter()
            .any(|p| *p == pulled_point_on_mesh_calculated.point));

        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_outside_top_front_right() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(1.25, 1.25, 1.25);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        // corner
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 1.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert_eq!(pulled_point_on_mesh_calculated.point, point_on_mesh_correct);
        assert_eq!(
            pulled_point_on_mesh_calculated.distance,
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
        );
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_outside_front_right() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(1.25, 1.25, 0.25);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        // on the edge
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 0.25);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert_eq!(pulled_point_on_mesh_calculated.point, point_on_mesh_correct);
        assert_eq!(
            pulled_point_on_mesh_calculated.distance,
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
        );
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_on_face() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(0.0, 0.0, 1.0);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert_eq!(pulled_point_on_mesh_calculated.point, test_point);
        assert_eq!(
            pulled_point_on_mesh_calculated.distance,
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
        );
    }

    #[test]
    fn test_pull_point_to_mesh_box_point_on_edge() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let test_point = Point3::new(1.0, 1.0, 0.0);
        let unoriented_edges: Vec<_> = mesh.unoriented_edges_iter().collect();

        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &mesh, &unoriented_edges);

        assert_eq!(pulled_point_on_mesh_calculated.point, test_point);
        assert_eq!(
            pulled_point_on_mesh_calculated.distance,
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
        );
    }

    #[test]
    fn test_is_point_in_triangle_returns_true_for_point_inside() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 0.0, 0.0);

        assert!(is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_outside() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 2.0, 0.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_above() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 0.0, 1.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_true_for_point_inside_xy() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.25, 0.25, 0.0);

        assert!(is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_outside_xy() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(1.0, 1.0, 0.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_above_xy() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.25, 0.25, 1.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_true_for_point_inside_xz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.25, 0.0, 0.25);

        assert!(is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_outside_xz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(1.0, 0.0, 1.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_above_xz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.25, 1.0, 0.25);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_true_for_point_inside_yz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.0, 0.25, 0.25);

        assert!(is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_outside_yz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(0.0, 1.0, 1.0);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    fn test_is_point_in_triangle_returns_false_for_point_above_yz() {
        let triangle_points = (
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, 0.0, 0.0),
        );

        let test_point = Point3::new(1.0, 0.25, 0.25);

        assert!(!is_point_in_triangle(
            &test_point,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2
        ));
    }

    #[test]
    #[should_panic = "Ray vector zero"]
    fn test_ray_intersects_triangle_panics_for_zero_vector_ray() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 1.0),
        );

        let ray_origin = Point3::origin();
        let ray_vector = Vector3::zeros();

        ray_intersects_triangle(
            &ray_origin,
            &ray_vector,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
        );
    }

    #[test]
    fn test_ray_intersects_triangle_for_horizontal_triangle_returns_point_inside() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_correct = Point3::new(0.25, 0.25, 0.0);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_triangle_correct);

        let point_on_triangle_calculated = ray_intersects_triangle(
            &ray_origin,
            &ray_vector,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
        )
        .expect("Point is not on the triangle.");

        assert_eq!(
            point_on_triangle_calculated.point,
            point_on_triangle_correct
        );
        assert_eq!(point_on_triangle_calculated.distance, distance_correct);
    }

    #[test]
    fn test_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_misses() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(1.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_calculated = ray_intersects_triangle(
            &ray_origin,
            &ray_vector,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
        );

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_parallel() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(1.0, 0.0, 0.0);

        let point_on_triangle_calculated = ray_intersects_triangle(
            &ray_origin,
            &ray_vector,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
        );

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_ray_intersects_triangle_for_arbitrary_triangle_returns_point_inside() {
        let triangle_points = (
            &Point3::new(0.268023, 0.8302, 0.392469),
            &Point3::new(-0.870844, -0.462665, 0.215034),
            &Point3::new(0.334798, -0.197734, -0.999972),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(-0.622709, 0.614937, -0.483823);

        let point_on_triangle_correct = Point3::new(0.07773773, 0.42011225, 0.11615828);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_triangle_correct);

        let point_on_triangle_calculated = ray_intersects_triangle(
            &ray_origin,
            &ray_vector,
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
        )
        .expect("Point is not on the triangle.");

        assert_eq!(
            point_on_triangle_calculated.point,
            point_on_triangle_correct
        );
        assert_eq!(point_on_triangle_calculated.distance, distance_correct);
    }

    #[test]
    fn test_ray_intersects_plane_seed_2() {
        let normal = Vector3::new(0.505588, 0.843833, -0.179794);
        let ray_vector = Vector3::new(-0.708348, 0.05881, -0.703409);
        let ray_origin = Point3::new(0.328762, 0.9441, 0.02429);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_correct = Point3::new(-3.4010215, 1.2537622, -3.6794877);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_plane_correct);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane)
            .expect("The ray doesn't intersect the plane");

        assert_eq!(point_on_plane_calculated.point, point_on_plane_correct);
        assert!(approx::relative_eq!(
            point_on_plane_calculated.distance,
            distance_correct
        ));
    }

    #[test]
    fn test_ray_intersects_plane_seed_12() {
        let normal = Vector3::new(-0.987928, 0.02117, -0.15346);
        let ray_vector = Vector3::new(0.515951, 0.520597, -0.680275);
        let ray_origin = Point3::new(0.758986, -0.648866, 0.053964);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_correct = Point3::new(-0.25097048, -1.667917, 1.3855792);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_plane_correct);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane)
            .expect("The ray doesn't intersect the plane");

        assert_eq!(point_on_plane_calculated.point, point_on_plane_correct);
        assert!(approx::relative_eq!(
            point_on_plane_calculated.distance,
            distance_correct
        ));
    }

    #[test]
    fn test_ray_intersects_plane_returns_none_because_ray_parallel_to_plane() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::new(1.0, 1.0, 0.0);
        let ray_origin = Point3::new(0.0, 0.0, 1.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane);

        assert_eq!(point_on_plane_calculated, None);
    }

    #[test]
    fn test_ray_intersects_plane_returns_self_for_point_on_plane() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::new(1.0, 1.0, 1.0);
        let ray_origin = Point3::new(1.0, 1.0, 0.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane)
            .expect("The ray doesn't intersect the plane");

        assert_eq!(point_on_plane_calculated.point, ray_origin);
        assert!(approx::relative_eq!(
            point_on_plane_calculated.distance,
            0.0,
        ));
    }

    #[test]
    #[should_panic = "Ray vector zero"]
    fn test_ray_intersects_plane_panics_for_zero_vector_ray() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::zeros();
        let ray_origin = Point3::new(1.0, 1.0, 0.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        ray_intersects_plane(&ray_origin, &ray_vector, &plane);
    }

    #[test]
    fn test_ray_intersects_plane_returns_none_for_parallel_ray() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::new(1.0, 1.0, 0.0);
        let ray_origin = Point3::new(1.0, 1.0, 1.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane);

        assert_eq!(point_on_plane_calculated, None);
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_true_for_point_on_start() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(-1.0, -1.0, -1.0);

        assert!(is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_true_for_point_on_end() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(1.0, 1.0, 1.0);

        assert!(is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_true_for_point_in_the_middle() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(0.0, 0.0, 0.0);

        assert!(is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_true_for_point_before() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(-2.0, -2.0, -2.0);

        assert!(is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_true_for_point_after() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(2.0, 2.0, 2.0);

        assert!(is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_unclamped_returns_false_for_point_elsewhere() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(3.0, 2.0, 1.0);

        assert!(!is_point_on_line_unclamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_true_for_point_on_start() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(-1.0, -1.0, -1.0);

        assert!(is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_true_for_point_on_end() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(1.0, 1.0, 1.0);

        assert!(is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_true_for_point_in_the_middle() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(0.0, 0.0, 0.0);

        assert!(is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_false_for_point_before() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(-2.0, -2.0, -2.0);

        assert!(!is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_false_for_point_after() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(2.0, 2.0, 2.0);

        assert!(!is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }

    #[test]
    fn test_is_point_on_line_clamped_returns_false_for_point_elsewhere() {
        let line_start = Point3::new(-1.0, -1.0, -1.0);
        let line_end = Point3::new(1.0, 1.0, 1.0);
        let test_point = Point3::new(3.0, 2.0, 1.0);

        assert!(!is_point_on_line_clamped(
            &test_point,
            &line_start,
            &line_end
        ));
    }
}

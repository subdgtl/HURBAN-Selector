use std::f32;

use nalgebra;
use nalgebra::base::Vector3;
use nalgebra::geometry::{Point2, Point3};

use crate::convert::cast_usize;
use crate::geometry::{self, Face, Geometry, Plane, UnorientedEdge};

/// Compute barycentric coordinates of point P in triangle A, B, C. Returns None
/// for degenerate triangles.
pub fn compute_barycentric_coords(
    a: Point2<f32>,
    b: Point2<f32>,
    c: Point2<f32>,
    p: Point2<f32>,
) -> Option<Point3<f32>> {
    let ab = b - a;
    let ac = c - a;
    let pa = a - p;
    let xs = Vector3::new(ac.x, ab.x, pa.x);
    let ys = Vector3::new(ac.y, ab.y, pa.y);
    let ortho = xs.cross(&ys);
    if f32::abs(ortho.z) < 1.0 {
        None
    } else {
        Some(Point3::new(
            1.0 - (ortho.x + ortho.y) / ortho.z,
            ortho.y / ortho.z,
            ortho.x / ortho.z,
        ))
    }
}

/// Checks if a point lies in a triangle.
///
/// https://math.stackexchange.com/questions/4322/check-whether-a-point-is-within-a-3d-triangle
// TODO: Orient the triangle and point to XY plane and then test
fn is_point_in_triangle(
    point: &Point3<f32>,
    triangle_vertices: (&Point3<f32>, &Point3<f32>, &Point3<f32>),
) -> bool {
    // If the triangle is degenerated into a point, check if the point is equal
    // to the test point
    if triangle_vertices.0 == triangle_vertices.1 && triangle_vertices.0 == triangle_vertices.2 {
        return point == triangle_vertices.0;
    }

    // If the triangle is degenerated into a line, check if the point lies on
    // the line
    if triangle_vertices.0 == triangle_vertices.1 {
        return is_point_on_line_clamped(point, triangle_vertices.0, triangle_vertices.2);
    }
    if triangle_vertices.0 == triangle_vertices.2 || triangle_vertices.1 == triangle_vertices.2 {
        return is_point_on_line_clamped(point, triangle_vertices.0, triangle_vertices.1);
    }

    let plane = Plane::from_three_points(
        triangle_vertices.0,
        triangle_vertices.1,
        triangle_vertices.2,
    );

    if is_point_on_plane(point, &plane) {
        let a = Point2::origin();
        let b = Point2::from((triangle_vertices.1 - triangle_vertices.0).xy());
        let c = Point2::from((triangle_vertices.2 - triangle_vertices.0).xy());
        let p = Point2::from((point - triangle_vertices.0).xy());
        let barycentric_point =
            compute_barycentric_coords(a, b, c, p).expect("The triangle is degenerate");

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
    } else {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulledPointWithDistance {
    point: Point3<f32>,
    distance: f32,
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
    triangle_vertices: (&Point3<f32>, &Point3<f32>, &Point3<f32>),
) -> Option<PulledPointWithDistance> {
    // TODO: test
    assert!(ray_vector != &Vector3::zeros(), "Ray vector zero");

    let ray_vector_normalized = ray_vector.normalize();
    let edge_1_vector = triangle_vertices.1 - triangle_vertices.0;
    let edge_2_vector = triangle_vertices.2 - triangle_vertices.0;
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
    let tangent_vector = ray_origin - triangle_vertices.0;
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

/// Test if an arbitrary point lies on a plane.
///
/// https://stackoverflow.com/questions/17227149/using-dot-product-to-determine-if-point-lies-on-a-plane
fn is_point_on_plane(point: &Point3<f32>, plane: &Plane) -> bool {
    let vector_from_plane_point_to_point = point - plane.origin();
    point == plane.origin()
        || approx::relative_eq!(
            vector_from_plane_point_to_point
                .normalize()
                .dot(&plane.normal()),
            0.0
        )
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
    // TODO: test
    assert!(ray_vector != &Vector3::zeros(), "Ray vector zero");

    let plane_normal = plane.normal();
    let ray_vector_normalized = ray_vector.normalize();
    let denominator = ray_vector_normalized.dot(&plane_normal);
    // The ray is parallel to the plane
    if approx::relative_eq!(denominator, 0.0) {
        None
    } else {
        let ray_to_plane_origin_vector = ray_origin - *plane.origin();
        let t_parameter = ray_to_plane_origin_vector.dot(&plane_normal) / denominator;
        Some(PulledPointWithDistance {
            point: ray_origin - ray_vector_normalized.scale(t_parameter),
            distance: t_parameter.abs(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointOnLine {
    clamped: Point3<f32>,
    unclamped: Point3<f32>,
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

/// Pulls arbitrary point to the closest point of a mesh geometry.
///
/// Cast a ray from the point perpendicular to each mesh face and if there is an
/// intersection, add it to the list. Also measure a distance to the closest
/// point on each mesh geometry edge. Pick the closest point of them all as the
/// pulled point.
#[allow(dead_code)]
pub fn pull_point_to_mesh(
    point: &Point3<f32>,
    geometry: &Geometry,
    unoriented_edges: &[UnorientedEdge],
) -> PulledPointWithDistance {
    let vertices = geometry.vertices();
    let all_mesh_faces_with_normals = geometry.faces().iter().map(|Face::Triangle(t_f)| {
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
    let points_pulled_to_faces =
        all_mesh_faces_with_normals.filter_map(|(face_vertices, face_normal)| {
            // TODO: test if the point is on the triangle first
            ray_intersects_triangle(point, &(-1.0 * face_normal), face_vertices)
        });

    let points_pulled_to_edges = unoriented_edges.iter().map(|u_e| {
        let closest_point = pull_point_to_line(
            point,
            &vertices[cast_usize(u_e.0.vertices.0)],
            &vertices[cast_usize(u_e.0.vertices.1)],
        );
        PulledPointWithDistance {
            point: closest_point.clamped,
            distance: nalgebra::distance(point, &closest_point.clamped),
        }
    });

    let mut all_pulled_points = points_pulled_to_faces.chain(points_pulled_to_edges);

    let mut closest_pulled_point = all_pulled_points.next().expect("No pulled point found");
    for current_pulled_point in all_pulled_points {
        if current_pulled_point.distance < closest_pulled_point.distance {
            closest_pulled_point = current_pulled_point;
        }
    }

    closest_pulled_point
}

/// Pulls arbitrary point to the closest point of a plane.
///
/// Cast a ray from the point perpendicular to the plane and calculate their
/// intersection.
#[allow(dead_code)]
pub fn pull_point_to_plane(point: &Point3<f32>, plane: &Plane) -> PulledPointWithDistance {
    if is_point_on_plane(point, &plane) {
        return PulledPointWithDistance {
            point: *point,
            distance: 0.0,
        };
    } else {
        ray_intersects_plane(point, &plane.normal(), plane)
            .expect("The normal is parallel to its plane")
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry;

    use super::*;

    #[test]
    fn test_vertex_tools_compute_barycentric_coords_for_point_inside() {
        let triangle_points = (
            Point2::new(0.0, 1.0),
            Point2::new(-0.866025, -0.5),
            Point2::new(0.866025, -0.5),
        );

        let test_point = Point2::new(0.0, 0.0);
        let barycentric_calculated = compute_barycentric_coords(
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
            test_point,
        )
        .expect("Could not calculate the barycentric coords");

        let barycentric_correct = Point3::new(0.333333, 0.333333, 0.333333);

        assert!(barycentric_calculated.coords.relative_eq(
            &barycentric_correct.coords,
            0.001,
            0.001
        ));
    }

    #[test]
    fn test_vertex_tools_compute_barycentric_coords_for_point_outside() {
        let triangle_points = (
            Point2::new(0.0, 1.0),
            Point2::new(-0.866025, -0.5),
            Point2::new(0.866025, -0.5),
        );

        let test_point = Point2::new(0.0, 2.0);
        let barycentric_calculated = compute_barycentric_coords(
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
            test_point,
        )
        .expect("Could not calculate the barycentric coords");

        let barycentric_correct = Point3::new(1.6666667, -0.33333334, -0.33333334);

        assert!(barycentric_calculated.coords.relative_eq(
            &barycentric_correct.coords,
            0.001,
            0.001
        ));
    }

    #[test]
    fn test_vertex_tools_is_point_in_triangle_returns_true_for_point_inside() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 0.0, 0.0);

        assert!(is_point_in_triangle(&test_point, triangle_points));
    }

    #[test]
    fn test_vertex_tools_is_point_in_triangle_returns_false_for_point_outside() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 2.0, 0.0);

        assert!(!is_point_in_triangle(&test_point, triangle_points));
    }

    #[test]
    fn test_vertex_tools_is_point_in_triangle_returns_false_for_point_above() {
        let triangle_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let test_point = Point3::new(0.0, 0.0, 1.0);

        assert!(!is_point_in_triangle(&test_point, triangle_points));
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_point_inside() {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_correct = Point3::new(0.25, 0.25, 0.0);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_triangle_correct);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points)
                .expect("Point is not on the triangle.");

        assert_eq!(
            point_on_triangle_calculated.point,
            point_on_triangle_correct
        );
        assert_eq!(point_on_triangle_calculated.distance, distance_correct);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_misses(
    ) {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(1.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points);

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_parallel(
    ) {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(1.0, 0.0, 0.0);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points);

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_arbitrary_triangle_returns_point_inside() {
        let face_points = (
            &Point3::new(0.268023, 0.8302, 0.392469),
            &Point3::new(-0.870844, -0.462665, 0.215034),
            &Point3::new(0.334798, -0.197734, -0.999972),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(-0.622709, 0.614937, -0.483823);

        let point_on_triangle_correct = Point3::new(0.07773773, 0.42011225, 0.11615828);
        let distance_correct = nalgebra::distance(&ray_origin, &point_on_triangle_correct);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points)
                .expect("Point is not on the triangle.");

        assert_eq!(
            point_on_triangle_calculated.point,
            point_on_triangle_correct
        );
        assert_eq!(point_on_triangle_calculated.distance, distance_correct);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_line_for_point_pulled_to_center() {
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
    fn test_vertex_tools_pull_point_to_line_for_point_pulled_below_start() {
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
    fn test_vertex_tools_pull_point_to_line_for_point_pulled_beyond_end() {
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
    fn test_vertex_tools_pull_point_to_line_for_point_on_the_line() {
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
    fn test_vertex_tools_pull_point_to_mesh_cube_point_inside_left() {
        let cube_geometry = geometry::cube_sharp_geometry([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(-0.25, 0.0, 0.0);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        let point_on_mesh_correct = Point3::new(-1.0, 0.0, 0.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(point_on_mesh_correct, pulled_point_on_mesh_calculated.point);
        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_inside_top_front_right() {
        let cube_geometry = geometry::cube_sharp_geometry([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(0.25, 0.25, 0.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // any of the following points on mesh would be correct
        let points_on_mesh_correct = vec![
            Point3::new(1.0, 0.25, 0.25),
            Point3::new(0.25, 1.0, 0.25),
            Point3::new(0.25, 0.25, 1.0),
        ];
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert!(points_on_mesh_correct
            .iter()
            .any(|p| *p == pulled_point_on_mesh_calculated.point));

        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_outside_top_front_right() {
        let cube_geometry = geometry::cube_sharp_geometry([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(1.25, 1.25, 1.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // corner
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 1.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(point_on_mesh_correct, pulled_point_on_mesh_calculated.point);
        assert_eq!(
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
            pulled_point_on_mesh_calculated.distance
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_outside_front_right() {
        let cube_geometry = geometry::cube_sharp_geometry([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(1.25, 1.25, 0.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // on the edge
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 0.25);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(point_on_mesh_correct, pulled_point_on_mesh_calculated.point);
        assert_eq!(
            nalgebra::distance(&test_point, &pulled_point_on_mesh_calculated.point),
            pulled_point_on_mesh_calculated.distance
        );
    }

    #[test]
    fn test_vertex_tools_is_point_on_plane_returns_true_for_point_on_plane() {
        let plane_origin = Point3::new(1.0, 0.0, 0.0);
        let plane_normal = Vector3::new(1.0, 0.0, 0.0);
        let plane = Plane::from_origin_and_normal(&plane_origin, &plane_normal);
        let test_point = Point3::new(1.0, 1.0, 1.0);

        assert!(is_point_on_plane(&test_point, &plane));
    }

    #[test]
    fn test_vertex_tools_is_point_on_plane_returns_false_for_point_elsewhere() {
        let plane_origin = Point3::new(1.0, 0.0, 0.0);
        let plane_normal = Vector3::new(1.0, 0.0, 0.0);
        let plane = Plane::from_origin_and_normal(&plane_origin, &plane_normal);
        let test_point = Point3::new(2.0, 1.0, 1.0);

        assert!(!is_point_on_plane(&test_point, &plane));
    }

    #[test]
    fn test_vertex_tools_ray_intersects_plane_seed_2() {
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
    fn test_vertex_tools_ray_intersects_plane_seed_12() {
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
    fn test_vertex_tools_ray_intersects_plane_returns_none_because_ray_parallel_to_plane() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::new(1.0, 1.0, 0.0);
        let ray_origin = Point3::new(0.0, 0.0, 1.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane);

        assert_eq!(point_on_plane_calculated, None);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_plane_returns_self_for_point_on_plane() {
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
    fn test_vertex_tools_ray_intersects_plane_panics_for_zero_vector_ray() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::zeros();
        let ray_origin = Point3::new(1.0, 1.0, 0.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        ray_intersects_plane(&ray_origin, &ray_vector, &plane);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_plane_returns_none_for_parallel_ray() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let ray_vector = Vector3::new(1.0, 1.0, 0.0);
        let ray_origin = Point3::new(1.0, 1.0, 1.0);

        let plane = Plane::from_origin_and_normal(&Point3::origin(), &normal);

        let point_on_plane_calculated = ray_intersects_plane(&ray_origin, &ray_vector, &plane);

        assert_eq!(point_on_plane_calculated, None);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_plane_for_point_on_plane_and_horizontal_plane() {
        let test_point: Point3<f32> = Point3::origin();
        let test_plane =
            Plane::from_origin_and_normal(&Point3::origin(), &Vector3::new(0.0, 0.0, 1.0));

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert_eq!(point_pulled_to_plane_calculated.point, test_point);
        assert_eq!(point_pulled_to_plane_calculated.distance, 0.0);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_plane_for_point_on_plane_and_vertical_plane() {
        let test_point: Point3<f32> = Point3::origin();
        let test_plane =
            Plane::from_origin_and_normal(&Point3::origin(), &Vector3::new(1.0, 0.0, 0.0));

        let point_pulled_to_plane_calculated = pull_point_to_plane(&test_point, &test_plane);

        assert_eq!(point_pulled_to_plane_calculated.point, test_point);
        assert_eq!(point_pulled_to_plane_calculated.distance, 0.0);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_plane_for_point_above_plane_and_horizontal_plane() {
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
    fn test_vertex_tools_pull_point_to_plane_for_arbitrary_point_and_arbitrary_plane() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_true_for_point_on_start() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_true_for_point_on_end() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_true_for_point_in_the_middle() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_true_for_point_before() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_true_for_point_after() {
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
    fn test_vertex_tools_pull_is_point_on_line_unclamped_returns_false_for_point_elsewhere() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_true_for_point_on_start() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_true_for_point_on_end() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_true_for_point_in_the_middle() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_false_for_point_before() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_false_for_point_after() {
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
    fn test_vertex_tools_pull_is_point_on_line_clamped_returns_false_for_point_elsewhere() {
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

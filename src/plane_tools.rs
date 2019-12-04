use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;

use crate::geometry::Plane;

/// Fit a plane to a collection of points. Fast, and accurate to within a few
/// degrees. Returns None if the points do not span a plane.
///
/// The resulting normal is flipped randomly as well as the rotation of the X
/// and Y vector around the normal is random. The result is, nevertheless,
/// correct.
///
/// https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
#[allow(dead_code)]
pub fn plane_fit(points: &[Point3<f32>]) -> Option<Plane> {
    let n = points.len();
    // Not enough points to calculate a plane.
    if n < 3 {
        return None;
    }

    let mut sum = Vector3::zeros();
    for p in points {
        sum += p.coords;
    }
    let centroid = sum * (1.0 / (n as f32));

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let r = p - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    let n_f32 = n as f32;

    xx /= n_f32;
    xy /= n_f32;
    xz /= n_f32;
    yy /= n_f32;
    yz /= n_f32;
    zz /= n_f32;

    let mut weighted_dir = Vector3::zeros();

    let det_x = yy * zz - yz * yz;
    let axis_dir = Vector3::new(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    let mut weight = det_x * det_x;
    if weighted_dir.dot(&axis_dir) < 0.0 {
        weight = -weight;
    }
    weighted_dir += axis_dir * weight;

    let det_y = xx * zz - xz * xz;
    let axis_dir = Vector3::new(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    let mut weight = det_y * det_y;
    if weighted_dir.dot(&axis_dir) < 0.0 {
        weight = -weight;
    }
    weighted_dir += axis_dir * weight;

    let det_z = xx * yy - xy * xy;
    let axis_dir = Vector3::new(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    let mut weight = det_z * det_z;
    if weighted_dir.dot(&axis_dir) < 0.0 {
        weight = -weight;
    }
    weighted_dir += axis_dir * weight;

    let normal = weighted_dir.normalize();
    if normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite() {
        Some(Plane::from_origin_and_normal(
            &Point3::from(centroid),
            &normal,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test compares calculated values to the result of a similar function
    /// in Grasshopper. The internal logic of the Grasshopper function is
    /// unknown, therefore the resulting plane is flipped or rotated around its
    /// normal differently. The origin of a plane fitted in Grasshopper is also
    /// located randomly.
    #[test]
    fn test_plane_tools_plane_fit_random_seed_1() {
        let points = vec![
            Point3::new(2.486686, 1.10744, 0.934021),
            Point3::new(8.758215, 7.961443, 0.440452),
            Point3::new(0.960045, 9.516898, 0.965176),
            Point3::new(9.142283, 0.918479, 1.504984),
            Point3::new(4.419898, 6.225771, 1.422512),
            Point3::new(9.717592, 4.613409, 1.807314),
            Point3::new(0.23115, 5.332513, 0.465347),
            Point3::new(6.13154, 3.183633, 0.808695),
            Point3::new(5.983811, 0.062063, 1.120022),
            Point3::new(2.671959, 3.858621, 0.598954),
            Point3::new(4.155326, 9.551214, 0.09266),
            Point3::new(6.821242, 9.279189, 1.344678),
            Point3::new(8.870291, 2.748636, 0.067714),
            Point3::new(2.329386, 7.446438, 1.592039),
            Point3::new(0.037465, 2.681375, 0.790804),
        ];

        let plane_calculated = plane_fit(&points).expect("Plane not created");

        let origin_correct = Point3::new(4.8477926, 4.965808, 0.9303582);
        let normal_correct = Vector3::new(-0.026102116, 0.009860026, 0.9996106);

        assert_eq!(*plane_calculated.origin(), origin_correct);
        assert_eq!(plane_calculated.normal(), normal_correct);
    }

    /// The test compares calculated values to the result of a similar function
    /// in Grasshopper. The internal logic of the Grasshopper function is
    /// unknown, therefore the resulting plane is flipped or rotated around its
    /// normal differently. The origin of a plane fitted in Grasshopper is also
    /// located randomly.
    #[test]
    fn test_plane_tools_plane_fit_random_seed_12() {
        let points = vec![
            Point3::new(9.95347, 3.383488, 0.311757),
            Point3::new(0.020258, 9.610504, 0.822709),
            Point3::new(1.911678, 0.818737, 0.044826),
            Point3::new(6.037565, 8.114436, 1.005404),
            Point3::new(0.27844, 4.986335, 1.130643),
            Point3::new(5.880629, 3.86344, 1.481233),
            Point3::new(8.124532, 0.28451, 0.996495),
            Point3::new(9.404667, 9.729843, 1.884917),
            Point3::new(8.900096, 6.551787, 1.322558),
            Point3::new(3.227978, 9.941923, 1.257427),
            Point3::new(1.905207, 7.259329, 0.053678),
            Point3::new(4.071256, 1.187643, 1.063247),
            Point3::new(3.860367, 5.956172, 1.921102),
            Point3::new(2.39709, 3.30792, 0.28436),
            Point3::new(0.301991, 2.241202, 1.657795),
            Point3::new(0.453235, 7.417713, 1.894652),
            Point3::new(6.14066, 1.697522, 0.516279),
            Point3::new(6.926976, 6.209543, 0.319568),
            Point3::new(4.112988, 7.534449, 0.266323),
            Point3::new(9.237105, 8.553053, 0.030565),
        ];

        let plane_calculated = plane_fit(&points).expect("Plane not created");

        let origin_correct = Point3::new(4.65731, 5.4324775, 0.913277);
        let normal_correct = Vector3::new(-0.021554187, 0.032416273, -0.999242);

        assert_eq!(*plane_calculated.origin(), origin_correct);
        assert_eq!(plane_calculated.normal(), normal_correct);
    }
}

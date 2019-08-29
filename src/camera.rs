use std::cmp::Ordering;
use std::f32;

use nalgebra::{Matrix4, Point3, Rotation3, Vector3};

use crate::math::clamp;

const TWO_PI: f32 = f32::consts::PI * 2.0;
const ZOOM_SPEED_BASE: f32 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraOptions {
    pub radius_max: f32,
    pub radius_min: f32,
    pub polar_angle_distance_min: f32,
    pub speed_pan: f32,
    pub speed_rotate: f32,
    pub speed_zoom: f32,
    pub speed_zoom_step: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    aspect_ratio: f32,
    radius: f32,
    azimuthal_angle: f32,
    polar_angle: f32,
    origin: Point3<f32>,
    up: Vector3<f32>,
    options: CameraOptions,
}

impl Camera {
    pub fn new(
        screen_size: [f32; 2],
        radius: f32,
        azimuthal_angle: f32,
        polar_angle: f32,
        options: CameraOptions,
    ) -> Camera {
        Camera {
            aspect_ratio: screen_size[0] / screen_size[1],
            radius: clamp(radius, options.radius_min, options.radius_max),
            azimuthal_angle: azimuthal_angle % TWO_PI,
            polar_angle: clamp(
                polar_angle,
                options.polar_angle_distance_min,
                f32::consts::PI - options.polar_angle_distance_min,
            ),
            origin: Point3::origin(),
            up: Vector3::z(),
            options,
        }
    }

    pub fn set_screen_size(&mut self, screen_size: [f32; 2]) {
        self.aspect_ratio = screen_size[0] / screen_size[1];
    }

    pub fn reset_origin(&mut self) {
        self.origin = Point3::origin();
    }

    pub fn pan_ground(&mut self, dx: f32, dy: f32) {
        let pan_factor = self.options.speed_pan * self.radius / self.options.radius_max;
        let ground_translation = Vector3::new(dx, dy, 0.0) * pan_factor;
        let camera_rotation =
            Rotation3::new(Vector3::z() * (self.azimuthal_angle - f32::consts::FRAC_PI_2));
        self.origin += camera_rotation * ground_translation;
    }

    pub fn pan_screen(&mut self, dx: f32, dy: f32) {
        let pan_factor = self.options.speed_pan * self.radius / self.options.radius_max;
        let ground_translation = Vector3::new(dx, dy, 0.0) * pan_factor;
        let camera_rotation =
            Rotation3::new(Vector3::z() * (self.azimuthal_angle - f32::consts::FRAC_PI_2));

        // Compute normal vector of the screen plane
        let eye = self.compute_eye();
        let normal = eye - self.origin;

        // Create rotation from XY plane to screen plane
        // Note that the vectors can theoretically be zero... just don't do anything in that case.
        if let Some(xy_to_screen_rotation) = Rotation3::rotation_between(&self.up, &normal) {
            self.origin += xy_to_screen_rotation * camera_rotation * ground_translation;
        }
    }

    pub fn rotate(&mut self, dtheta: f32, dphi: f32) {
        let dtheta = dtheta * self.options.speed_rotate;
        let dphi = dphi * self.options.speed_rotate;

        self.azimuthal_angle = (self.azimuthal_angle + dtheta) % TWO_PI;
        self.polar_angle = clamp(
            self.polar_angle + dphi,
            self.options.polar_angle_distance_min,
            f32::consts::PI - self.options.polar_angle_distance_min,
        );
    }

    pub fn zoom(&mut self, zoom_scale: f32) {
        let zoom_speed = ZOOM_SPEED_BASE.powf(self.options.speed_zoom * zoom_scale.abs());
        let new_radius = match zoom_scale.partial_cmp(&0.0) {
            Some(Ordering::Greater) => self.radius * zoom_speed,
            Some(Ordering::Less) => self.radius / zoom_speed,
            _ => self.radius,
        };

        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    pub fn zoom_step(&mut self, zoom_steps: i32) {
        let zoom_speed = ZOOM_SPEED_BASE.powf(self.options.speed_zoom_step);

        let mut new_radius = self.radius;
        match zoom_steps.cmp(&0) {
            Ordering::Greater => {
                for _ in 0..zoom_steps {
                    new_radius *= zoom_speed;
                }
            }
            Ordering::Less => {
                for _ in zoom_steps..0 {
                    new_radius /= zoom_speed;
                }
            }
            _ => (),
        }

        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    /// Attempt to fit a sphere into camera view frustrum.
    ///
    /// Camera options may affect the outcome. A too small
    /// `radius_max` or a too large `radius_min` may cause the result
    /// to be not zoomed out enough, or not zoomed in enough.
    pub fn zoom_to_fit_sphere(&mut self, sphere_origin: &Point3<f32>, sphere_radius: f32) {
        const MARGIN_MULTIPLIER: f32 = 1.005;

        let fovy = self.options.fovy;
        let fovx = fovy * self.aspect_ratio;
        let fov = fovy.min(fovx);

        // Compute the distance needed from the sphere for it to fit
        // inside the camera frustrum
        let new_radius = MARGIN_MULTIPLIER * sphere_radius / (fov / 2.0).tan();

        self.origin = *sphere_origin;
        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let eye = self.compute_eye();
        Matrix4::look_at_rh(&eye, &self.origin, &self.up)
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(
            self.aspect_ratio,
            self.options.fovy,
            self.options.znear,
            self.options.zfar,
        )
    }

    fn compute_eye(&self) -> Point3<f32> {
        let x = self.radius * self.azimuthal_angle.cos() * self.polar_angle.sin();
        let y = self.radius * self.azimuthal_angle.sin() * self.polar_angle.sin();
        let z = self.radius * self.polar_angle.cos();

        self.origin + Vector3::new(x, y, z)
    }
}

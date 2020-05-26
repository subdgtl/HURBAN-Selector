use std::cmp::Ordering;
use std::f32;

use nalgebra::{Matrix4, Point3, Rotation3, Vector3};

use crate::math::{clamp, TAU};

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

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    screen_width: u32,
    screen_height: u32,
    radius: f32,
    azimuthal_angle: f32,
    polar_angle: f32,
    origin: Point3<f32>,
    options: CameraOptions,
}

impl Camera {
    /// Creates a new camera with screen dimensions, radius, azimuthal and polar angle.
    ///
    /// Azimuthal angle is computed counter-clockwise from the unit vector
    /// `[1,0]` lying on the XY plane. Polar angle is the angle between the Z
    /// axis and the current camera position.
    pub fn new(
        screen_width: u32,
        screen_height: u32,
        radius: f32,
        azimuthal_angle: f32,
        polar_angle: f32,
        options: CameraOptions,
    ) -> Camera {
        Camera {
            screen_width,
            screen_height,
            radius: clamp(radius, options.radius_min, options.radius_max),
            azimuthal_angle: azimuthal_angle % TAU,
            polar_angle: clamp(
                polar_angle,
                options.polar_angle_distance_min,
                f32::consts::PI - options.polar_angle_distance_min,
            ),
            origin: Point3::origin(),
            options,
        }
    }

    pub fn position(&self) -> Point3<f32> {
        let x = self.radius * self.azimuthal_angle.cos() * self.polar_angle.sin();
        let y = self.radius * self.azimuthal_angle.sin() * self.polar_angle.sin();
        let z = self.radius * self.polar_angle.cos();

        self.origin + Vector3::new(x, y, z)
    }

    pub fn set_screen_dimensions(&mut self, screen_width: u32, screen_height: u32) {
        self.screen_width = screen_width;
        self.screen_height = screen_height;
    }

    pub fn screen_aspect_ratio(&self) -> f32 {
        self.screen_width as f32 / self.screen_height as f32
    }

    /// Pans the camera by changing the camera position against the ground plane
    /// (XY). `dx` and `dy` are in screen space.
    pub fn pan_ground(&mut self, dx: f32, dy: f32) {
        // FIXME: Raycast against ground plane for nicer movement.
        let pan_factor = self.options.speed_pan * self.radius / self.options.radius_max;
        let ground_translation = Vector3::new(dx, dy, 0.0) * pan_factor;
        let camera_rotation =
            Rotation3::new(Vector3::z() * (self.azimuthal_angle - f32::consts::FRAC_PI_2));
        self.origin += camera_rotation * ground_translation;
    }

    /// Pans the camera by changing the camera position against the screen
    /// plane. `dx` and `dy` are in screen space.
    pub fn pan_screen(&mut self, dx: f32, dy: f32) {
        let pan_factor = self.options.speed_pan * self.radius / self.options.radius_max;
        let ground_translation = Vector3::new(dx, dy, 0.0) * pan_factor;
        let camera_rotation =
            Rotation3::new(Vector3::z() * (self.azimuthal_angle - f32::consts::FRAC_PI_2));

        // Compute normal vector of the screen plane
        let eye = self.position();
        let normal = eye - self.origin;

        // Create rotation from XY plane to screen plane
        // Note that the vectors can theoretically be zero... just don't do anything in that case.
        if let Some(xy_to_screen_rotation) = Rotation3::rotation_between(&Vector3::z(), &normal) {
            self.origin += xy_to_screen_rotation * camera_rotation * ground_translation;
        }
    }

    /// Rotates the camera by changing azimuthal (theta) and polar (phi) angles.
    pub fn rotate(&mut self, dtheta: f32, dphi: f32) {
        let dtheta = dtheta * self.options.speed_rotate;
        let dphi = dphi * self.options.speed_rotate;

        self.azimuthal_angle = (self.azimuthal_angle + dtheta) % TAU;
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

    /// A sphere completely visible by this camera, no matter the rotation.
    pub fn visible_sphere(&self) -> (Point3<f32>, f32) {
        const MARGIN_MULTIPLIER: f32 = 1.005;
        let alpha = self.compute_visible_sphere_alpha();

        let sphere_radius = self.radius / MARGIN_MULTIPLIER * alpha.tan();

        (self.origin, sphere_radius)
    }

    /// Attempt to fit a sphere into camera view, no matter the rotation.
    ///
    /// Camera options may affect the outcome. A too small
    /// `radius_max` or a too large `radius_min` may cause the result
    /// to be not zoomed out enough, or not zoomed in enough.
    pub fn zoom_to_fit_visible_sphere(&mut self, sphere_origin: Point3<f32>, sphere_radius: f32) {
        const MARGIN_MULTIPLIER: f32 = 1.005;
        let alpha = self.compute_visible_sphere_alpha();

        // Compute the distance needed from the sphere for it to fit
        // inside the camera frustum
        let new_radius = MARGIN_MULTIPLIER * sphere_radius / alpha.tan();

        self.origin = sphere_origin;
        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position(), &self.origin, &Vector3::z())
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(
            self.screen_aspect_ratio(),
            self.options.fovy,
            self.options.znear,
            self.options.zfar,
        )
    }

    fn compute_visible_sphere_alpha(&self) -> f32 {
        let fovy = self.options.fovy;
        let fovx = fovy * self.screen_aspect_ratio();
        let fov = fovy.min(fovx);

        fov / 2.0
    }
}

pub use crate::logger::LogLevel;
pub use crate::renderer::{GpuBackend, Msaa, PresentMode};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::Point3;

use crate::camera::{Camera, CameraOptions};
use crate::convert::cast_usize;
use crate::input::InputManager;
use crate::interpreter::{Value, VarIdent};
use crate::mesh::{analysis, Mesh};
use crate::renderer::{DrawMeshMode, GpuMesh, GpuMeshId, Options as RendererOptions, Renderer};
use crate::session::{PollInterpreterResponseNotification, Session};
use crate::ui::Ui;

pub mod geometry;
pub mod importer;
pub mod renderer;

mod camera;
mod convert;
mod input;
mod interpreter;
mod interpreter_funcs;
mod interpreter_server;
mod logger;
mod math;
mod mesh;
mod plane;
mod platform;
mod pull;
mod session;
mod ui;

const CAMERA_INTERPOLATION_DURATION: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Whether to open a fullscreen window.
    pub fullscreen: bool,
    /// Which multi-sampling setting to use.
    pub msaa: Msaa,
    /// Whether to run with VSync or not.
    pub present_mode: PresentMode,
    /// Whether to select an explicit gpu backend for the renderer to use.
    pub gpu_backend: Option<GpuBackend>,
    /// Logging level for the editor.
    pub app_log_level: Option<logger::LogLevel>,
    /// Logging level for external libraries.
    pub lib_log_level: Option<logger::LogLevel>,
}

/// A unique identifier assigned to a value or subvalue for purposes
/// of displaying in the viewport.
///
/// Since we support value arrays, there can be multiple geometries
/// contained in a single value that all need to be treated separately
/// for pusposes of scene geometry analysis and rendering.
///
/// For simple values, the path is always `(var_ident, 0)`. For array
/// element values, the path is `(var_ident, array_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ValuePath(VarIdent, usize);

/// Initialize the window and run in infinite loop.
///
/// Will continue running until a close request is received from the
/// created window.
pub fn init_and_run(options: Options) -> ! {
    logger::init(options.app_log_level, options.lib_log_level);

    let event_loop = winit::event_loop::EventLoop::new();
    let window = if options.fullscreen {
        log::info!("Running in fullscreen mode, looking for compatible video modes...");
        let monitor = event_loop.primary_monitor();

        // TODO: needs testing whether the best video mode is always
        // given first on all systems
        if let Some(video_mode) = monitor.video_modes().next() {
            log::info!("Found fullscreen video mode: {}", video_mode);
            winit::window::WindowBuilder::new()
                .with_title("H.U.R.B.A.N. Selector")
                .with_fullscreen(Some(winit::window::Fullscreen::Exclusive(video_mode)))
                .build(&event_loop)
                .expect("Failed to create window")
        } else {
            log::info!("Didn't find compatible video mode, falling back to borderless");
            winit::window::WindowBuilder::new()
                .with_title("H.U.R.B.A.N. Selector")
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                .build(&event_loop)
                .expect("Failed to create window")
        }
    } else {
        log::info!("Running in windowed mode");
        winit::window::WindowBuilder::new()
            .with_title("H.U.R.B.A.N. Selector")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .build(&event_loop)
            .expect("Failed to create window")
    };

    let window_size = window.inner_size().to_physical(window.hidpi_factor());

    let mut session = Session::new();
    let mut input_manager = InputManager::new();
    let mut ui = Ui::new(&window);

    let mut camera = Camera::new(
        window_size,
        5.0,
        45f32.to_radians(),
        60f32.to_radians(),
        CameraOptions {
            radius_min: 1.0,
            radius_max: 10000.0,
            polar_angle_distance_min: 1f32.to_radians(),
            speed_pan: 10.0,
            speed_rotate: 0.005,
            speed_zoom: 0.01,
            speed_zoom_step: 1.0,
            fovy: 45f32.to_radians(),
            znear: 0.01,
            zfar: 1000.0,
        },
    );

    let mut renderer_draw_mesh_mode = DrawMeshMode::Shaded;
    let mut renderer = Renderer::new(
        &window,
        &camera.projection_matrix(),
        &camera.view_matrix(),
        ui.fonts(),
        RendererOptions {
            // FIXME: @Correctness Msaa X4 is the only value currently
            // working on all devices we tried. Once msaa capabilities
            // are queryable with wgpu `Limits`, we should have a
            // chain of options the renderer tries before giving up,
            // and this field should be renamed to `desired_msaa`.
            msaa: options.msaa,
            present_mode: options.present_mode,
            gpu_backend: options.gpu_backend,
        },
    );

    let mut scene_meshes: HashMap<ValuePath, Arc<Mesh>> = HashMap::new();
    let mut scene_gpu_mesh_ids: HashMap<ValuePath, GpuMeshId> = HashMap::new();

    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);

    let time_start = Instant::now();
    let mut time = time_start;

    let mut camera_interpolation: Option<CameraInterpolation> = None;

    // Since input manager needs to process events separately after imgui
    // handles them, this buffer with copies of events is needed.
    let mut input_events: Vec<winit::event::Event<_>> = Vec::with_capacity(16);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            winit::event::Event::EventsCleared => {
                let (duration_last_frame, _duration_running) = {
                    let now = Instant::now();
                    let duration_last_frame = now.duration_since(time);
                    let duration_running = now.duration_since(time_start);
                    time = now;

                    (duration_last_frame, duration_running)
                };

                ui.set_delta_time(duration_last_frame.as_secs_f32());

                let ui_frame = ui.prepare_frame(&window);
                input_manager.start_frame();

                for event in input_events.drain(..) {
                    input_manager.process_event(
                        &event,
                        ui_frame.want_capture_keyboard(),
                        ui_frame.want_capture_mouse(),
                    );
                }

                let input_state = input_manager.input_state();

                let [pan_ground_x, pan_ground_y] = input_state.camera_pan_ground;
                let [pan_screen_x, pan_screen_y] = input_state.camera_pan_screen;
                let [rotate_x, rotate_y] = input_state.camera_rotate;

                camera.pan_ground(pan_ground_x, pan_ground_y);
                camera.pan_screen(pan_screen_x, pan_screen_y);
                camera.rotate(rotate_x, rotate_y);
                camera.zoom(input_state.camera_zoom);
                camera.zoom_step(input_state.camera_zoom_steps);

                let ui_reset_viewport =
                    ui_frame.draw_viewport_settings_window(&mut renderer_draw_mesh_mode);
                ui_frame.draw_pipeline_window(&mut session);
                ui_frame.draw_operations_window(&mut session);

                if input_state.camera_reset_viewport || ui_reset_viewport {
                    camera_interpolation = Some(CameraInterpolation::new(
                        &camera,
                        scene_meshes.values().map(Arc::as_ref),
                        time,
                    ));
                }

                if input_state.close_requested {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                if let Some(logical_size) = input_state.window_resized {
                    let physical_size = logical_size.to_physical(window.hidpi_factor());
                    log::debug!(
                        "Window resized to new size: logical [{},{}], physical [{},{}]",
                        logical_size.width,
                        logical_size.height,
                        physical_size.width,
                        physical_size.height,
                    );

                    camera.set_window_size(physical_size);
                    renderer.set_window_size(physical_size);
                }

                session.poll_interpreter_response(|callback_value| match callback_value {
                    PollInterpreterResponseNotification::Add(var_ident, value) => match value {
                        Value::Mesh(mesh) => {
                            let gpu_mesh = GpuMesh::from_mesh(&mesh);
                            let gpu_mesh_id = renderer
                                .add_scene_mesh(&gpu_mesh)
                                .expect("Failed to upload scene mesh");

                            let path = ValuePath(var_ident, 0);

                            scene_meshes.insert(path, mesh);
                            scene_gpu_mesh_ids.insert(path, gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for (index, mesh) in mesh_array.iter().enumerate() {
                                let gpu_mesh = GpuMesh::from_mesh(&mesh);
                                let gpu_mesh_id = renderer
                                    .add_scene_mesh(&gpu_mesh)
                                    .expect("Failed to upload scene mesh");

                                let path = ValuePath(var_ident, index);

                                scene_meshes.insert(path, mesh);
                                scene_gpu_mesh_ids.insert(path, gpu_mesh_id);
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },
                    PollInterpreterResponseNotification::Remove(var_ident, value) => match value {
                        Value::Mesh(_) => {
                            let path = ValuePath(var_ident, 0);

                            scene_meshes.remove(&path);
                            let gpu_mesh_id = scene_gpu_mesh_ids
                                .remove(&path)
                                .expect("Gpu mesh ID was not tracked");

                            renderer.remove_scene_mesh(gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for index in 0..mesh_array.len() {
                                let path = ValuePath(var_ident, cast_usize(index));

                                scene_meshes.remove(&path);
                                let gpu_mesh_id = scene_gpu_mesh_ids
                                    .remove(&path)
                                    .expect("Gpu mesh ID was not tracked");

                                renderer.remove_scene_mesh(gpu_mesh_id);
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },
                });

                if let Some(interp) = camera_interpolation {
                    if interp.target_time > time {
                        let (sphere_origin, sphere_radius) = interp.update(time, &cubic_bezier);
                        camera.zoom_to_fit_visible_sphere(sphere_origin, sphere_radius);
                    } else {
                        camera
                            .zoom_to_fit_visible_sphere(interp.target_origin, interp.target_radius);
                        camera_interpolation = None;
                    }
                }

                let imgui_draw_data = ui_frame.render(&window);

                // Camera matrices have to be uploaded when either window
                // resizes or the camera moves. We do it every frame for
                // simplicity.
                renderer.set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());
                let mut render_pass = renderer.begin_render_pass();

                render_pass.draw_mesh(scene_gpu_mesh_ids.values(), renderer_draw_mesh_mode);
                render_pass.draw_ui(imgui_draw_data);

                render_pass.submit();
            }

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::RedrawRequested,
                ..
            } => {
                // We don't answer redraw requests and instead draw in
                // the "events cleared" for 2 reasons:
                //
                // 1) Doing it with VSync is challenging - redrawing
                //    at the OS's whim while knowing that each redraw
                //    will block until the monitor flips sounds
                //    dangerous. We could still redraw here only when
                //    we were running without VSync, but...
                //
                // 2) ImGui produces a draw list with `render()`. The
                //    drawlist shares the lifetime of the `Ui` frame
                //    context, which is dropped at the end of "events
                //    cleared". We could copy the draw list and stash
                //    it for our subsequent handling of redraw
                //    requests, but it contains raw pointers to the
                //    `Ui` frame context which I am not sure are alive
                //    (or even contain correct data) by the time we
                //    get here. We could also try to prolong the
                //    lifetime of the `Ui` by not dropping it, but
                //    this is Rust...
            }

            winit::event::Event::WindowEvent { .. } => {
                ui.handle_event(&window, &event);
                input_events.push(event.clone());
            }

            _ => (),
        }
    });
}

#[derive(Debug, Clone, Copy)]
struct CameraInterpolation {
    source_origin: Point3<f32>,
    source_radius: f32,
    target_origin: Point3<f32>,
    target_radius: f32,
    target_time: Instant,
}

impl CameraInterpolation {
    fn new<'a, I>(camera: &Camera, scene_meshes: I, time: Instant) -> Self
    where
        I: IntoIterator<Item = &'a Mesh> + Clone,
    {
        let (source_origin, source_radius) = camera.visible_sphere();
        let (target_origin, target_radius) = analysis::compute_bounding_sphere(scene_meshes);

        CameraInterpolation {
            source_origin,
            source_radius,
            target_origin,
            target_radius,
            target_time: time + CAMERA_INTERPOLATION_DURATION,
        }
    }

    fn update(&self, time: Instant, easing: &math::CubicBezierEasing) -> (Point3<f32>, f32) {
        let duration_left = self.target_time.duration_since(time).as_secs_f32();
        let whole_duration = CAMERA_INTERPOLATION_DURATION.as_secs_f32();
        let t = easing.apply(1.0 - duration_left / whole_duration);

        let sphere_origin = Point3::from(
            self.source_origin
                .coords
                .lerp(&self.target_origin.coords, t),
        );
        let sphere_radius = math::lerp(self.source_radius, self.target_radius, t);

        (sphere_origin, sphere_radius)
    }
}

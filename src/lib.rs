pub use crate::logger::LogLevel;
pub use crate::renderer::{GpuBackend, GpuPowerPreference, Msaa};
pub use crate::ui::Theme;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::Point3;

use crate::bounding_box::BoundingBox;
use crate::camera::{Camera, CameraOptions};
use crate::convert::cast_usize;
use crate::input::InputManager;
use crate::interpreter::{Value, VarIdent};
use crate::mesh::Mesh;
use crate::notifications::{NotificationLevel, Notifications};
use crate::renderer::{DrawMeshMode, GpuMesh, GpuMeshHandle, Options as RendererOptions, Renderer};
use crate::session::{PollInterpreterResponseNotification, Session};
use crate::ui::{ScreenshotOptions, Ui};

pub mod geometry;
pub mod importer;
pub mod renderer;

mod bounding_box;
mod camera;
mod convert;
mod input;
mod interpreter;
mod interpreter_funcs;
mod interpreter_server;
mod logger;
mod math;
mod mesh;
mod notifications;
mod plane;
mod pull;
mod session;
mod ui;

const CAMERA_INTERPOLATION_DURATION: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// What theme to use.
    pub theme: Theme,
    /// Whether to open a fullscreen window.
    pub fullscreen: bool,
    /// Which multi-sampling setting to use.
    pub msaa: Msaa,
    /// Whether to run with VSync or not.
    pub vsync: bool,
    /// Whether to select an explicit gpu backend for the renderer to use.
    pub gpu_backend: Option<GpuBackend>,
    /// Whether to select an explicit power preference profile for the renderer
    /// to use when choosing a GPU.
    pub gpu_power_preference: Option<GpuPowerPreference>,
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
/// for purposes of scene geometry analysis and rendering.
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
        let monitor = event_loop.primary_monitor();

        // Exclusive fullscreen on macOS has 2 problems:
        // - winit does not report correct DPI once the window
        //   switches to fullscreen,
        // - wgpu on vulkan on metal can not allocate a large enough
        //   backbuffer.
        //
        // Neither of these happen on borderless fullscreen on macOS.
        // FIXME: Fix these issues in winit and wgpu.
        #[cfg(target_os = "macos")]
        {
            log::info!("Running in fullscreen mode on macOS, opening borderless fullscreen");
            winit::window::WindowBuilder::new()
                .with_title("H.U.R.B.A.N. Selector")
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                .build(&event_loop)
                .expect("Failed to create window")
        }

        #[cfg(not(target_os = "macos"))]
        {
            log::info!("Running in fullscreen mode, looking for exclusive fullscreen video modes");
            if let Some(video_mode) = monitor.video_modes().next() {
                log::info!(
                    "Found video mode: {}, opening exclusive fullscreen",
                    video_mode,
                );
                winit::window::WindowBuilder::new()
                    .with_title("H.U.R.B.A.N. Selector")
                    .with_fullscreen(Some(winit::window::Fullscreen::Exclusive(video_mode)))
                    .build(&event_loop)
                    .expect("Failed to create window")
            } else {
                log::info!("Didn't find compatible video mode, opening borderless fullscreen");
                winit::window::WindowBuilder::new()
                    .with_title("H.U.R.B.A.N. Selector")
                    .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                    .build(&event_loop)
                    .expect("Failed to create window")
            }
        }
    } else {
        log::info!("Running in windowed mode");
        winit::window::WindowBuilder::new()
            .with_title("H.U.R.B.A.N. Selector")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .build(&event_loop)
            .expect("Failed to create window")
    };

    let initial_window_size = window.inner_size().to_physical(window.hidpi_factor());
    let initial_window_width = initial_window_size.width.round() as u32;
    let initial_window_height = initial_window_size.height.round() as u32;
    let initial_window_aspect_ratio =
        initial_window_size.width as f32 / initial_window_size.height as f32;

    let mut session = Session::new();
    let mut input_manager = InputManager::new();

    let notification_ttl = Duration::from_secs(5);
    let notifications = Rc::new(RefCell::new(Notifications::with_ttl(notification_ttl)));

    let mut ui = Ui::new(&window, options.theme);

    let mut camera = Camera::new(
        initial_window_aspect_ratio,
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

    let mut screenshot_modal_open = false;
    let mut screenshot_options = ScreenshotOptions {
        width: initial_window_width,
        height: initial_window_height,
        transparent: true,
    };

    let clear_color = match options.theme {
        Theme::Dark => [0.1, 0.1, 0.1, 1.0],
        Theme::Funky => [1.0, 1.0, 1.0, 1.0],
    };
    let mut renderer_draw_mesh_mode = DrawMeshMode::Shaded;
    let mut renderer = Renderer::new(
        &window,
        initial_window_width,
        initial_window_height,
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
            vsync: options.vsync,
            backend: options.gpu_backend,
            power_preference: options.gpu_power_preference,
        },
    );

    let mut scene_meshes: HashMap<ValuePath, Arc<Mesh>> = HashMap::new();
    let mut scene_gpu_mesh_handles: HashMap<ValuePath, GpuMeshHandle> = HashMap::new();

    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);

    let mut camera_interpolation: Option<CameraInterpolation> = None;
    // Since input manager needs to process events separately after imgui
    // handles them, this buffer with copies of events is needed.
    let mut input_events: Vec<winit::event::Event<_>> = Vec::with_capacity(16);

    let time_start = Instant::now();
    let mut time = time_start;

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

                if input_state.open_screenshot_options {
                    screenshot_modal_open = true;
                }

                let [pan_ground_x, pan_ground_y] = input_state.camera_pan_ground;
                let [pan_screen_x, pan_screen_y] = input_state.camera_pan_screen;
                let [rotate_x, rotate_y] = input_state.camera_rotate;

                camera.pan_ground(pan_ground_x, pan_ground_y);
                camera.pan_screen(pan_screen_x, pan_screen_y);
                camera.rotate(rotate_x, rotate_y);
                camera.zoom(input_state.camera_zoom);
                camera.zoom_step(input_state.camera_zoom_steps);

                let ui_reset_viewport = ui_frame.draw_viewport_settings_window(
                    &mut screenshot_modal_open,
                    &mut renderer_draw_mesh_mode,
                );
                let reset_viewport = input_state.camera_reset_viewport || ui_reset_viewport;

                let window_size = window.inner_size().to_physical(window.hidpi_factor());
                let take_screenshot = ui_frame.draw_screenshot_window(
                    &mut screenshot_modal_open,
                    &mut screenshot_options,
                    window_size.width.round() as u32,
                    window_size.height.round() as u32,
                );
                ui_frame.draw_notifications_window(&notifications.borrow());

                ui_frame.draw_pipeline_window(&mut session);
                ui_frame.draw_operations_window(&mut session);

                if reset_viewport {
                    camera_interpolation = Some(CameraInterpolation::new(
                        &camera,
                        scene_meshes.values().map(Arc::as_ref),
                        time,
                    ));
                }

                let screenshot_render_target = if take_screenshot {
                    Some(renderer.add_offscreen_render_target(
                        screenshot_options.width,
                        screenshot_options.height,
                    ))
                } else {
                    None
                };

                if input_state.close_requested {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                if let Some(logical_size) = input_state.window_resized {
                    let physical_size = logical_size.to_physical(window.hidpi_factor());
                    log::debug!(
                        "Window resized to new physical size {}x{}",
                        physical_size.width,
                        physical_size.height,
                    );

                    let aspect_ratio = physical_size.width as f32 / physical_size.height as f32;
                    let width = physical_size.width.round() as u32;
                    let height = physical_size.height.round() as u32;

                    // While it can't be queried, 16 is usually the minimal
                    // dimension of certain types of textures. Creating anything
                    // smaller currently crashes most of our GPU backend/driver
                    // combinations.
                    if width >= 16 && height >= 16 {
                        screenshot_options.width = width;
                        screenshot_options.height = height;
                        camera.set_viewport_aspect_ratio(aspect_ratio);
                        renderer.set_window_size(width, height);
                    } else {
                        log::warn!("Ignoring new window physical size {}x{}", width, height);
                    }
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
                            scene_gpu_mesh_handles.insert(path, gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for (index, mesh) in mesh_array.iter_refcounted().enumerate() {
                                let gpu_mesh = GpuMesh::from_mesh(&mesh);
                                let gpu_mesh_id = renderer
                                    .add_scene_mesh(&gpu_mesh)
                                    .expect("Failed to upload scene mesh");

                                let path = ValuePath(var_ident, index);

                                scene_meshes.insert(path, mesh);
                                scene_gpu_mesh_handles.insert(path, gpu_mesh_id);
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },
                    PollInterpreterResponseNotification::Remove(var_ident, value) => match value {
                        Value::Mesh(_) => {
                            let path = ValuePath(var_ident, 0);

                            scene_meshes.remove(&path);
                            let gpu_mesh_id = scene_gpu_mesh_handles
                                .remove(&path)
                                .expect("Gpu mesh ID was not tracked");

                            renderer.remove_scene_mesh(gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for index in 0..mesh_array.len() {
                                let path = ValuePath(var_ident, cast_usize(index));

                                scene_meshes.remove(&path);
                                let gpu_mesh_id = scene_gpu_mesh_handles
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
                notifications.borrow_mut().update(time);

                let imgui_draw_data = ui_frame.render(&window);

                let mut window_command_buffer = renderer.begin_command_buffer(clear_color);
                window_command_buffer
                    .set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());
                window_command_buffer.draw_meshes_to_primary_render_target(
                    scene_gpu_mesh_handles.values(),
                    renderer_draw_mesh_mode,
                );
                window_command_buffer.blit_primary_render_target_to_backbuffer();
                window_command_buffer.draw_ui_to_backbuffer(imgui_draw_data);
                window_command_buffer.submit();

                if let Some(screenshot_render_target) = screenshot_render_target {
                    log::info!(
                        "Capturing screenshot with dimensions {}x{} and transparency {}",
                        screenshot_options.width,
                        screenshot_options.height,
                        screenshot_options.transparent,
                    );

                    let screenshot_aspect_ratio =
                        screenshot_options.width as f32 / screenshot_options.height as f32;

                    let mut screenshot_camera = camera.clone();
                    screenshot_camera.set_viewport_aspect_ratio(screenshot_aspect_ratio);

                    let screenshot_clear_color = if screenshot_options.transparent {
                        [0.0; 4]
                    } else {
                        clear_color
                    };

                    let mut screenshot_command_buffer =
                        renderer.begin_command_buffer(screenshot_clear_color);
                    screenshot_command_buffer.set_camera_matrices(
                        &screenshot_camera.projection_matrix(),
                        &screenshot_camera.view_matrix(),
                    );
                    screenshot_command_buffer.draw_meshes_to_offscreen_render_target(
                        &screenshot_render_target,
                        scene_gpu_mesh_handles.values(),
                        renderer_draw_mesh_mode,
                    );
                    screenshot_command_buffer.submit();

                    let screenshot_notifications = Rc::clone(&notifications);
                    renderer.offscreen_render_target_data(
                        &screenshot_render_target,
                        move |width, height, data| {
                            let actual_data_len = data.len();
                            let expected_data_len = cast_usize(width)
                                * cast_usize(height)
                                * cast_usize(mem::size_of::<[u8; 4]>());
                            if expected_data_len != actual_data_len {
                                log::error!(
                                    "Screenshot data is {} bytes, but was expected to be {} bytes",
                                    actual_data_len,
                                    expected_data_len,
                                );

                                return;
                            }

                            if let Some(mut path) = dirs::picture_dir() {
                                path.push(format!(
                                    "hurban_selector-{}.png",
                                    chrono::Local::now().format("%Y-%m-%d-%H-%M-%S"),
                                ));

                                let file = File::create(&path).unwrap();
                                let mut png_encoder = png::Encoder::new(file, width, height);
                                png_encoder.set_color(png::ColorType::RGBA);
                                png_encoder.set_depth(png::BitDepth::Eight);

                                png_encoder
                                    .write_header()
                                    .expect("Failed to write png header")
                                    .write_image_data(data)
                                    .expect("Failed to write png data");

                                let path_str = path.to_string_lossy();
                                log::info!("Screenshot saved in {}", path_str);
                                screenshot_notifications.borrow_mut().push(
                                    time,
                                    NotificationLevel::Info,
                                    format!("Screenshot saved in {}", path_str),
                                );
                            } else {
                                log::error!("Failed to find picture directory");
                                screenshot_notifications.borrow_mut().push(
                                    time,
                                    NotificationLevel::Warn,
                                    "Failed to find picture directory",
                                );
                            }
                        },
                    );

                    renderer.remove_offscreen_render_target(screenshot_render_target);
                }
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
        I: Iterator<Item = &'a Mesh>,
    {
        let (source_origin, source_radius) = camera.visible_sphere();
        let bounding_box_iter = scene_meshes.map(|mesh| mesh.bounding_box());

        let (target_origin, target_radius) = match BoundingBox::union(bounding_box_iter) {
            Some(bounding_box) => (bounding_box.center(), bounding_box.diagonal().norm() / 2.0),
            None => (Point3::origin(), 1.0),
        };

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

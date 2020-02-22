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

use image::{GenericImageView, Pixel};
use nalgebra::{Point3, Vector2, Vector3};

use crate::bounding_box::BoundingBox;
use crate::camera::{Camera, CameraOptions};
use crate::convert::cast_usize;
use crate::input::InputManager;
use crate::interpreter::{Value, VarIdent};
use crate::mesh::Mesh;
use crate::notifications::{NotificationLevel, Notifications};
use crate::plane::Plane;
use crate::project::ProjectStatus;
use crate::renderer::{
    DirectionalLight, DrawMeshMode, GpuMesh, GpuMeshHandle, Options as RendererOptions, Renderer,
};
use crate::session::{PollNotification, Session};
use crate::ui::{OverwriteModalTrigger, SaveModalResult, ScreenshotOptions, Ui};

pub mod geometry;
pub mod importer;
pub mod renderer;

mod analytics;
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
mod project;
mod pull;
mod session;
mod ui;

const DURATION_CAMERA_INTERPOLATION: Duration = Duration::from_millis(1000);
const DURATION_NOTIFICATION: Duration = Duration::from_millis(5000);
const DURATION_AUTORUN_DELAY: Duration = Duration::from_millis(100);
const BASE_WINDOW_TITLE: &str = "H.U.R.B.A.N. selector";

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

#[cfg(not(feature = "dist"))]
#[derive(Debug, Clone, Copy)]
enum RendererDebugView {
    Off,
    ShadowMap,
}

#[cfg(not(feature = "dist"))]
impl RendererDebugView {
    pub fn cycle(self) -> Self {
        match self {
            RendererDebugView::Off => RendererDebugView::ShadowMap,
            RendererDebugView::ShadowMap => RendererDebugView::Off,
        }
    }
}

/// Initialize the window and run in infinite loop.
///
/// Will continue running until a close request is received from the
/// created window.
pub fn init_and_run(options: Options) -> ! {
    logger::init(options.app_log_level, options.lib_log_level);

    let event_loop = winit::event_loop::EventLoop::new();
    let icon_file = include_bytes!("../icons/64x64.ico");
    let icon = load_icon(icon_file);
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
                .with_title(BASE_WINDOW_TITLE)
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
                    .with_title(BASE_WINDOW_TITLE)
                    .with_fullscreen(Some(winit::window::Fullscreen::Exclusive(video_mode)))
                    .with_window_icon(Some(icon))
                    .build(&event_loop)
                    .expect("Failed to create window")
            } else {
                log::info!("Didn't find compatible video mode, opening borderless fullscreen");
                winit::window::WindowBuilder::new()
                    .with_title(BASE_WINDOW_TITLE)
                    .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                    .with_window_icon(Some(icon))
                    .build(&event_loop)
                    .expect("Failed to create window")
            }
        }
    } else {
        log::info!("Running in windowed mode");

        winit::window::WindowBuilder::new()
            .with_title(BASE_WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .with_window_icon(Some(icon))
            .build(&event_loop)
            .expect("Failed to create window")
    };

    let initial_window_size = window.inner_size().to_physical(window.hidpi_factor());
    let initial_window_width = initial_window_size.width.round() as u32;
    let initial_window_height = initial_window_size.height.round() as u32;
    let initial_window_aspect_ratio =
        initial_window_size.width as f32 / initial_window_size.height as f32;

    let mut session = Session::new();
    session.set_autorun_delay(Some(DURATION_AUTORUN_DELAY));
    let mut input_manager = InputManager::new();

    let notifications = Rc::new(RefCell::new(Notifications::with_ttl(DURATION_NOTIFICATION)));

    let mut ui = Ui::new(&window, options.theme);

    let mut project_status = project::ProjectStatus::default();

    change_window_title(&window, &project_status);

    let mut camera = Camera::new(
        initial_window_aspect_ratio,
        5.0,
        270_f32.to_radians(),
        60_f32.to_radians(),
        CameraOptions {
            radius_min: 1.0,
            radius_max: 10000.0,
            polar_angle_distance_min: 1_f32.to_radians(),
            speed_pan: 10.0,
            speed_rotate: 0.005,
            speed_zoom: 0.01,
            speed_zoom_step: 1.0,
            fovy: 45_f32.to_radians(),
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

    #[cfg(not(feature = "dist"))]
    let mut renderer_debug_view = RendererDebugView::Off;
    let mut renderer_draw_mesh_mode = DrawMeshMode::Shaded;
    let mut renderer = Renderer::new(
        &window,
        initial_window_width,
        initial_window_height,
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
            flat_shading_color: [0.0, 0.0, 0.0, 0.1],
        },
    );

    let mut scene_bounding_box: BoundingBox<f32> = BoundingBox::unit();
    let mut scene_meshes: HashMap<ValuePath, Arc<Mesh>> = HashMap::new();
    let mut scene_gpu_mesh_handles: HashMap<ValuePath, GpuMeshHandle> = HashMap::new();

    let mut ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
    let mut ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
    let mut ground_plane_gpu_mesh_handle = Some(
        renderer
            .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
            .expect("Failed to add ground plane mesh"),
    );

    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);
    let mut camera_interpolation: Option<CameraInterpolation> = None;
    // Since input manager needs to process events separately after imgui
    // handles them, this buffer with copies of events is needed.
    let mut input_events: Vec<winit::event::Event<_>> = Vec::with_capacity(16);

    let time_start = Instant::now();
    let mut time = time_start;

    #[allow(clippy::cognitive_complexity)]
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

                #[cfg(not(feature = "dist"))]
                {
                    if input_state.debug_view_cycle {
                        renderer_debug_view = renderer_debug_view.cycle();
                        log::debug!("Cycled debug view to {:?}", renderer_debug_view);
                    }
                }

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

                let menu_status = ui_frame.draw_menu_window(
                    &mut screenshot_modal_open,
                    &mut renderer_draw_mesh_mode,
                    &mut project_status,
                );
                let reset_viewport =
                    input_state.camera_reset_viewport || menu_status.reset_viewport;

                if let Some(prevent_overwrite_modal_trigger) = menu_status.prevent_overwrite_modal {
                    project_status.prevent_overwrite_status = match prevent_overwrite_modal_trigger
                    {
                        OverwriteModalTrigger::NewProject => {
                            Some(crate::project::NextAction::NewProject)
                        }
                        OverwriteModalTrigger::OpenProject => {
                            Some(crate::project::NextAction::OpenProject)
                        }
                    }
                }

                if menu_status.new_project {
                    scene_meshes.clear();

                    for (_, gpu_mesh_handle) in scene_gpu_mesh_handles.drain() {
                        renderer.remove_scene_mesh(gpu_mesh_handle);
                    }

                    scene_bounding_box =
                        BoundingBox::union(scene_meshes.values().map(|mesh| mesh.bounding_box()))
                            .unwrap_or_else(BoundingBox::unit);

                    ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                    ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                    renderer.remove_scene_mesh(
                        ground_plane_gpu_mesh_handle
                            .take()
                            .expect("Ground plane must always be present"),
                    );
                    ground_plane_gpu_mesh_handle = Some(
                        renderer
                            .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                            .expect("Failed to add ground plane mesh"),
                    );

                    let current_autorun_delay = session.autorun_delay();
                    session = Session::new();
                    session.set_autorun_delay(current_autorun_delay);

                    project_status.path = None;
                    project_status.changed_since_last_save = false;

                    change_window_title(&window, &project_status);
                }

                if let Some(save_path) = menu_status.save_path {
                    log::info!("Saving project at {}", save_path);

                    let stmts = session.stmts().to_vec();
                    let project = project::Project { version: 1, stmts };

                    match project::save(&save_path, project) {
                        Ok(_) => {
                            project_status.save(&save_path);

                            change_window_title(&window, &project_status);
                        }
                        Err(err) => {
                            log::error!("{}", err);
                            project_status.error = Some(err);
                        }
                    }
                }

                if let Some(open_path) = menu_status.open_path {
                    log::info!("Opening new project at {}", open_path);

                    match project::open(&open_path) {
                        Ok(project) => {
                            scene_meshes.clear();

                            for (_, gpu_mesh_handle) in scene_gpu_mesh_handles.drain() {
                                renderer.remove_scene_mesh(gpu_mesh_handle);
                            }

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|mesh| mesh.bounding_box()),
                            )
                            .unwrap_or_else(BoundingBox::unit);

                            ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                            ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                            renderer.remove_scene_mesh(
                                ground_plane_gpu_mesh_handle
                                    .take()
                                    .expect("Ground plane must always be present"),
                            );
                            ground_plane_gpu_mesh_handle = Some(
                                renderer
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                                    .expect("Failed to add ground plane mesh"),
                            );

                            let current_autorun_delay = session.autorun_delay();
                            session = Session::new();
                            session.set_autorun_delay(current_autorun_delay);

                            for stmt in project.stmts {
                                session.push_prog_stmt(time, stmt);
                            }

                            project_status.path = Some(open_path);
                            project_status.changed_since_last_save = false;

                            change_window_title(&window, &project_status);
                        }
                        Err(err) => {
                            log::error!("{}", err);
                            project_status.error = Some(err);
                        }
                    };
                }

                if project_status.error.is_some()
                    && ui_frame.draw_error_modal(&project_status.error)
                {
                    project_status.error = None;
                }

                let window_size = window.inner_size().to_physical(window.hidpi_factor());
                let take_screenshot = ui_frame.draw_screenshot_window(
                    &mut screenshot_modal_open,
                    &mut screenshot_options,
                    window_size.width.round() as u32,
                    window_size.height.round() as u32,
                );
                ui_frame.draw_notifications_window(&notifications.borrow());

                if ui_frame.draw_pipeline_window(time, &mut session) {
                    project_status.changed_since_last_save = true;

                    change_window_title(&window, &project_status);
                }

                if ui_frame.draw_operations_window(time, &mut session, DURATION_AUTORUN_DELAY) {
                    project_status.changed_since_last_save = true;

                    change_window_title(&window, &project_status);
                }

                if let Some(prevent_overwrite_status) =
                    project_status.prevent_overwrite_status.clone()
                {
                    match ui_frame.draw_prevent_overwrite_modal() {
                        SaveModalResult::Cancel => {
                            project_status.prevent_overwrite_status = None;
                        }
                        SaveModalResult::DontSave => match prevent_overwrite_status {
                            project::NextAction::Exit => {
                                *control_flow = winit::event_loop::ControlFlow::Exit
                            }
                            project::NextAction::NewProject => {
                                project_status.new_requested = true;
                            }
                            project::NextAction::OpenProject => {
                                project_status.open_requested = true
                            }
                        },
                        SaveModalResult::Save => {
                            let save_path = match project_status.path.clone() {
                                Some(project_path) => Some(project_path),
                                None => ui_frame.draw_save_dialog(),
                            };

                            if let Some(save_path) = save_path {
                                let stmts = session.stmts().to_vec();
                                let project = project::Project { version: 1, stmts };

                                match project::save(&save_path, project) {
                                    Ok(_) => match prevent_overwrite_status {
                                        project::NextAction::Exit => {
                                            *control_flow = winit::event_loop::ControlFlow::Exit
                                        }
                                        project::NextAction::NewProject => {
                                            project_status.save(&save_path);
                                            project_status.new_requested = true;
                                        }
                                        project::NextAction::OpenProject => {
                                            project_status.save(&save_path);
                                            project_status.open_requested = true
                                        }
                                    },
                                    Err(err) => {
                                        log::error!("{}", err);

                                        project_status.error = Some(err);
                                    }
                                }
                            }

                            project_status.prevent_overwrite_status = None;
                        }
                        SaveModalResult::Nothing => {}
                    }
                }

                if reset_viewport {
                    camera_interpolation =
                        Some(CameraInterpolation::new(&camera, &scene_bounding_box, time));
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
                    if project_status.changed_since_last_save {
                        project_status.prevent_overwrite_status = Some(project::NextAction::Exit);
                    } else {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
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

                session.poll(time, |callback_value| match callback_value {
                    PollNotification::Add(var_ident, value) => match value {
                        Value::Mesh(mesh) => {
                            let gpu_mesh = GpuMesh::from_mesh(&mesh);
                            let gpu_mesh_id = renderer
                                .add_scene_mesh(&gpu_mesh, false)
                                .expect("Failed to upload scene mesh");

                            let path = ValuePath(var_ident, 0);

                            scene_meshes.insert(path, mesh);
                            scene_gpu_mesh_handles.insert(path, gpu_mesh_id);

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|mesh| mesh.bounding_box()),
                            )
                            .unwrap_or_else(BoundingBox::unit);

                            ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                            ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                            renderer.remove_scene_mesh(
                                ground_plane_gpu_mesh_handle
                                    .take()
                                    .expect("Ground plane must always be present"),
                            );
                            ground_plane_gpu_mesh_handle = Some(
                                renderer
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                                    .expect("Failed to add ground plane mesh"),
                            );
                        }
                        Value::MeshArray(mesh_array) => {
                            for (index, mesh) in mesh_array.iter_refcounted().enumerate() {
                                let gpu_mesh = GpuMesh::from_mesh(&mesh);
                                let gpu_mesh_id = renderer
                                    .add_scene_mesh(&gpu_mesh, false)
                                    .expect("Failed to upload scene mesh");

                                let path = ValuePath(var_ident, index);

                                scene_meshes.insert(path, mesh);
                                scene_gpu_mesh_handles.insert(path, gpu_mesh_id);
                            }

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|mesh| mesh.bounding_box()),
                            )
                            .unwrap_or_else(BoundingBox::unit);

                            ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                            ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                            renderer.remove_scene_mesh(
                                ground_plane_gpu_mesh_handle
                                    .take()
                                    .expect("Ground plane must always be present"),
                            );
                            ground_plane_gpu_mesh_handle = Some(
                                renderer
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                                    .expect("Failed to add ground plane mesh"),
                            );
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },
                    PollNotification::Remove(var_ident, value) => match value {
                        Value::Mesh(_) => {
                            let path = ValuePath(var_ident, 0);

                            scene_meshes.remove(&path);
                            let gpu_mesh_id = scene_gpu_mesh_handles
                                .remove(&path)
                                .expect("Gpu mesh ID was not tracked");

                            renderer.remove_scene_mesh(gpu_mesh_id);

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|mesh| mesh.bounding_box()),
                            )
                            .unwrap_or_else(BoundingBox::unit);

                            ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                            ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                            renderer.remove_scene_mesh(
                                ground_plane_gpu_mesh_handle
                                    .take()
                                    .expect("Ground plane must always be present"),
                            );
                            ground_plane_gpu_mesh_handle = Some(
                                renderer
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                                    .expect("Failed to add ground plane mesh"),
                            );
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

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|mesh| mesh.bounding_box()),
                            )
                            .unwrap_or_else(BoundingBox::unit);

                            ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
                            ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
                            renderer.remove_scene_mesh(
                                ground_plane_gpu_mesh_handle
                                    .take()
                                    .expect("Ground plane must always be present"),
                            );
                            ground_plane_gpu_mesh_handle = Some(
                                renderer
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh), true)
                                    .expect("Failed to add ground plane mesh"),
                            );
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
                window_command_buffer.set_light(&compute_light(
                    &ground_plane_mesh_bounding_box,
                    &scene_bounding_box,
                    &camera,
                ));
                window_command_buffer
                    .set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());

                window_command_buffer.draw_meshes_to_primary_render_target(
                    scene_gpu_mesh_handles.values(),
                    renderer_draw_mesh_mode,
                    true,
                );
                window_command_buffer.draw_meshes_to_primary_render_target(
                    ground_plane_gpu_mesh_handle.iter(),
                    DrawMeshMode::FlatWithShadows,
                    false,
                );

                #[cfg(not(feature = "dist"))]
                match renderer_debug_view {
                    RendererDebugView::Off => {
                        window_command_buffer.blit_primary_render_target_to_backbuffer();
                    }
                    RendererDebugView::ShadowMap => {
                        window_command_buffer.blit_shadow_map_to_backbuffer();
                    }
                }

                #[cfg(feature = "dist")]
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
                    screenshot_command_buffer.set_light(&compute_light(
                        &ground_plane_mesh_bounding_box,
                        &scene_bounding_box,
                        &camera,
                    ));
                    screenshot_command_buffer.set_camera_matrices(
                        &screenshot_camera.projection_matrix(),
                        &screenshot_camera.view_matrix(),
                    );

                    // For screenshots, we don't need to cast shadows, and we
                    // don't render the ground on purpose.
                    screenshot_command_buffer.draw_meshes_to_offscreen_render_target(
                        &screenshot_render_target,
                        scene_gpu_mesh_handles.values(),
                        renderer_draw_mesh_mode,
                        false,
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
    fn new(camera: &Camera, bounding_box: &BoundingBox<f32>, time: Instant) -> Self {
        let (source_origin, source_radius) = camera.visible_sphere();
        let (target_origin, target_radius) =
            if approx::relative_eq!(bounding_box.minimum_point(), bounding_box.maximum_point()) {
                (Point3::origin(), 1.0)
            } else {
                (bounding_box.center(), bounding_box.diagonal().norm() / 2.0)
            };

        CameraInterpolation {
            source_origin,
            source_radius,
            target_origin,
            target_radius,
            target_time: time + DURATION_CAMERA_INTERPOLATION,
        }
    }

    fn update(&self, time: Instant, easing: &math::CubicBezierEasing) -> (Point3<f32>, f32) {
        let duration_left = self.target_time.duration_since(time).as_secs_f32();
        let whole_duration = DURATION_CAMERA_INTERPOLATION.as_secs_f32();
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

fn load_icon(contents: &[u8]) -> winit::window::Icon {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::load_from_memory(contents).expect("Failed to load icon contents.");
        let (width, height) = image.dimensions();
        let mut rgba = Vec::with_capacity((width * height) as usize * 4);

        for (_, _, pixel) in image.pixels() {
            rgba.extend_from_slice(&pixel.to_rgba().0);
        }

        (rgba, width, height)
    };

    winit::window::Icon::from_rgba(icon_rgba, icon_width, icon_height)
        .expect("Failed to create icon.")
}

fn compute_light(
    ground_plane_bounding_box: &BoundingBox<f32>,
    scene_bounding_box: &BoundingBox<f32>,
    camera: &Camera,
) -> DirectionalLight {
    let (camera_origin, camera_radius) = camera.visible_sphere();
    let camera_radius_vector = Vector3::new(camera_radius, camera_radius, camera_radius) * 5.0;
    let camera_bounding_box = BoundingBox::new(
        &(camera_origin - camera_radius_vector),
        &(camera_origin + camera_radius_vector),
    );
    let bounding_box = BoundingBox::intersection(
        [
            *ground_plane_bounding_box,
            *scene_bounding_box,
            camera_bounding_box,
        ]
        .iter()
        .copied(),
    )
    .unwrap_or(camera_bounding_box);

    let diagonal_length = bounding_box.diagonal().norm() * 1.2;
    // We need to skew the directional vector so that it is never equal to the Z
    // axis and it is possible to compute a view matrix from it. The decision is
    // arbitrary as long as it is consistent, but we skew it in the Y axis only
    // so that the light looks slightly in the positive Y direction.
    let skew = diagonal_length * 0.01;
    DirectionalLight {
        position: bounding_box.center() + Vector3::new(0.0, 0.0, diagonal_length / 2.0),
        direction: Vector3::new(0.0, skew, -diagonal_length).normalize(),
        min_range: 0.1,
        max_range: diagonal_length,
        width: diagonal_length,
    }
}

fn compute_ground_plane_mesh(scene_bounding_box: &BoundingBox<f32>) -> Mesh {
    let dimension = f32::max(1000.0, scene_bounding_box.diagonal().norm() * 100.0);
    mesh::primitive::create_mesh_plane(
        Plane::new(
            &Point3::origin(),
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 1.0, 0.0),
        ),
        Vector2::new(dimension, dimension),
    )
}

fn change_window_title(window: &winit::window::Window, project_status: &ProjectStatus) {
    use std::path::Path;

    let filename = match &project_status.path {
        Some(project_path) => Path::new(project_path)
            .file_name()
            .expect("Failed to parse file name of the project.")
            .to_str()
            .expect("Project file name isn't valid UTF-8."),
        None => "unsaved project",
    };
    let title = if project_status.changed_since_last_save {
        format!("{} - *{}", BASE_WINDOW_TITLE, filename)
    } else {
        format!("{} - {}", BASE_WINDOW_TITLE, filename)
    };

    window.set_title(&title);
}

pub use crate::logger::LogLevel;
pub use crate::renderer::{GpuBackend, GpuPowerPreference, Msaa};

use std::borrow::Cow;
use std::collections::HashMap;
use std::f32;
use std::fs::File;
use std::io::BufWriter;
use std::mem;
use std::path::PathBuf;
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
    DirectionalLight, GpuMesh, GpuMeshHandle, Material, Options as RendererOptions, Renderer,
};
use crate::session::{PollNotification, Session};
use crate::ui::{OverwriteModalTrigger, SaveModalResult, Ui};

pub mod geometry;
pub mod importer;
pub mod renderer;

mod analytics;
mod bounding_box;
mod camera;
mod convert;
mod exporter;
mod imgui_winit_support;
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

static IMAGE_DATA_ICON: &[u8] = include_bytes!("../icons/64x64.ico");
static IMAGE_DATA_SCHEME: &[u8] = include_bytes!("../resources/scheme.png");
static IMAGE_DATA_LOGOS_BLACK: &[u8] = include_bytes!("../resources/logos.png");
static IMAGE_DATA_LOGOS_WHITE: &[u8] = include_bytes!("../resources/logos_white.png");
static IMAGE_DATA_SUBDIGITAL_LOGO: &[u8] = include_bytes!("../resources/subdigital_grey.png");

const DURATION_CAMERA_INTERPOLATION: Duration = Duration::from_millis(1000);
const DURATION_NOTIFICATION: Duration = Duration::from_millis(5000);
const DURATION_AUTORUN_DELAY: Duration = Duration::from_millis(100);
const BASE_WINDOW_TITLE: &str = "H.U.R.B.A.N. selector";

#[derive(Debug, Clone, Copy, PartialEq, clap::Clap)]
#[clap(name = "HURBAN selector", version, author)]
pub struct Options {
    /// Theme for the editor.
    #[clap(long, arg_enum, env = "HS_THEME", default_value = "dark")]
    pub theme: Theme,
    /// GPU backend to use for rendering.
    ///
    /// If not chosen, the renderer will pick a default based on the current
    /// platform.
    #[clap(long, arg_enum, env = "HS_GPU_BACKEND")]
    pub gpu_backend: Option<GpuBackend>,
    /// Power preference for selecting a GPU.
    #[clap(
        long,
        arg_enum,
        env = "HS_GPU_POWER_PREFERENCE",
        default_value = "default"
    )]
    pub gpu_power_preference: GpuPowerPreference,
    /// Level of multi-sampling based anti-aliasing to use in rendering.
    #[clap(long, arg_enum, env = "HS_GPU_MSAA", default_value = "disabled")]
    pub gpu_msaa: Msaa,
    /// Logging level for the editor.
    #[clap(long, arg_enum, env = "HS_LOG_LEVEL_APP", default_value = "info")]
    pub log_level_app: LogLevel,
    /// Logging level for external libraries.
    #[clap(long, arg_enum, env = "HS_LOG_LEVEL_LIB", default_value = "warn")]
    pub log_level_lib: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Clap)]
pub enum Theme {
    Dark,
    Light,
}

/// The draw mode applied to a group of objects in the viewport. Not always a
/// 1:1 mapping with renderer materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportDrawMode {
    Wireframe,
    Shaded,
    ShadedWireframe,
    ShadedWireframeXray,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenshotOptions {
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
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
    logger::init(options.log_level_app, options.log_level_lib);

    let event_loop = winit::event_loop::EventLoop::new();

    let (img_icon, width_icon, height_icon) = decode_image_rgba8_unorm(IMAGE_DATA_ICON);
    let (img_scheme, width_scheme, height_scheme) = decode_image_rgba8_unorm(IMAGE_DATA_SCHEME);
    let (img_logos_black, width_logos_black, height_logos_black) =
        decode_image_rgba8_unorm(IMAGE_DATA_LOGOS_BLACK);
    let (img_logos_white, width_logos_white, height_logos_white) =
        decode_image_rgba8_unorm(IMAGE_DATA_LOGOS_WHITE);
    let (img_subdigital_logo, width_subdigital_logo, height_subdigital_logo) =
        decode_image_rgba8_unorm(IMAGE_DATA_SUBDIGITAL_LOGO);

    let icon = winit::window::Icon::from_rgba(img_icon, width_icon, height_icon)
        .expect("Failed to create icon.");

    let window = winit::window::WindowBuilder::new()
        .with_title(BASE_WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .with_maximized(true)
        .with_window_icon(Some(icon))
        .build(&event_loop)
        .expect("Failed to create window");

    // FIXME: Remove this workaround for `WindowBuilder::with_maximized` bug:
    // https://github.com/rust-windowing/winit/issues/1510
    #[cfg(target_os = "windows")]
    {
        window.set_maximized(true);
    }

    let initial_window_size = window.inner_size();
    let initial_window_width = initial_window_size.width;
    let initial_window_height = initial_window_size.height;

    let mut session = Session::new();
    session.set_autorun_delay(Some(DURATION_AUTORUN_DELAY));

    let mut input_manager = InputManager::new();
    let mut notifications = Notifications::with_ttl(DURATION_NOTIFICATION);
    let mut ui = Ui::new(&window, options.theme);
    let mut project_status = project::ProjectStatus::default();

    change_window_title(&window, &project_status);

    let mut screenshot_modal_open = false;
    let mut screenshot_options = ScreenshotOptions {
        width: initial_window_width,
        height: initial_window_height,
        transparent: true,
    };

    let mut about_modal_open = false;

    let clear_color = match options.theme {
        Theme::Dark => [0.1, 0.1, 0.1, 1.0],
        Theme::Light => [1.0, 1.0, 1.0, 1.0],
    };

    #[cfg(not(feature = "dist"))]
    let mut renderer_debug_view = RendererDebugView::Off;
    let mut viewport_draw_mode = ViewportDrawMode::ShadedWireframe;
    let mut viewport_draw_used_values = true;
    let mut renderer = Renderer::new(
        &window,
        initial_window_width,
        initial_window_height,
        ui.fonts(),
        RendererOptions {
            backend: options.gpu_backend,
            power_preference: options.gpu_power_preference,
            msaa: options.gpu_msaa,
            flat_material_color: [0.0, 0.0, 0.0, 0.1],
            // FIXME: These different alphas are to workaround a blending bug in
            // the renderer. Fix the blending bug.
            transparent_matcap_shaded_material_alpha: match options.theme {
                Theme::Dark => 0.5,
                Theme::Light => 0.15,
            },
        },
    );

    let tex_scheme = renderer.add_ui_texture_rgba8_unorm(width_scheme, height_scheme, &img_scheme);
    let tex_logos_black = renderer.add_ui_texture_rgba8_unorm(
        width_logos_black,
        height_logos_black,
        &img_logos_black,
    );
    let tex_logos_white = renderer.add_ui_texture_rgba8_unorm(
        width_logos_white,
        height_logos_white,
        &img_logos_white,
    );
    let tex_subdigital_logo = renderer.add_ui_texture_rgba8_unorm(
        width_subdigital_logo,
        height_subdigital_logo,
        &img_subdigital_logo,
    );

    let mut scene_bounding_box: BoundingBox<f32> = BoundingBox::unit();
    let mut scene_meshes: HashMap<ValuePath, (bool, Arc<Mesh>)> = HashMap::new();
    let mut scene_gpu_mesh_handles: HashMap<ValuePath, (bool, GpuMeshHandle)> = HashMap::new();

    let mut ground_plane_mesh = compute_ground_plane_mesh(&scene_bounding_box);
    let mut ground_plane_mesh_bounding_box = ground_plane_mesh.bounding_box();
    let mut ground_plane_gpu_mesh_handle = Some(
        renderer
            .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh))
            .expect("Failed to add ground plane mesh"),
    );

    let initial_camera_radius_max = compute_scene_camera_radius(scene_bounding_box);
    let mut camera = Camera::new(
        initial_window_width,
        initial_window_height,
        5.0,
        270_f32.to_radians(),
        60_f32.to_radians(),
        CameraOptions {
            radius_min: 0.001 * initial_camera_radius_max,
            radius_max: initial_camera_radius_max,
            polar_angle_distance_min: 1_f32.to_radians(),
            speed_pan: 10.0,
            speed_rotate: 0.005,
            speed_zoom: 0.01,
            speed_zoom_step: 1.0,
            fovy: 45_f32.to_radians(),
            znear: 0.001 * initial_camera_radius_max,
            zfar: 2.0 * initial_camera_radius_max,
        },
    );

    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);
    let mut camera_interpolation: Option<CameraInterpolation> = None;

    let time_start = Instant::now();
    let mut time = time_start;

    let mut ui_want_capture_mouse = false;
    let mut ui_want_capture_keyboard = false;

    #[allow(clippy::cognitive_complexity)]
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            winit::event::Event::NewEvents(_) => {
                let now = Instant::now();
                let duration_last_frame = now.duration_since(time);
                time = now;

                ui.set_delta_time(duration_last_frame.as_secs_f32());

                input_manager.start_frame();
            }
            winit::event::Event::MainEventsCleared => {
                // Poll at the beginning of event processing, so that the
                // pipeline UI is not lagging one frame behind.
                session.poll(time, |callback_value| match callback_value {
                    PollNotification::UsedValueAdded(var_ident, value) => match value {
                        Value::Mesh(mesh) => {
                            let gpu_mesh = GpuMesh::from_mesh(&mesh);
                            let gpu_mesh_id = renderer
                                .add_scene_mesh(&gpu_mesh)
                                .expect("Failed to upload scene mesh");

                            let path = ValuePath(var_ident, 0);

                            scene_meshes.insert(path, (true, mesh));
                            scene_gpu_mesh_handles.insert(path, (true, gpu_mesh_id));
                        }
                        Value::MeshArray(mesh_array) => {
                            for (index, mesh) in mesh_array.iter_refcounted().enumerate() {
                                let gpu_mesh = GpuMesh::from_mesh(&mesh);
                                let gpu_mesh_id = renderer
                                    .add_scene_mesh(&gpu_mesh)
                                    .expect("Failed to upload scene mesh");

                                let path = ValuePath(var_ident, index);

                                scene_meshes.insert(path, (true, mesh));
                                scene_gpu_mesh_handles.insert(path, (true, gpu_mesh_id));
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },

                    PollNotification::UsedValueRemoved(var_ident, value) => match value {
                        Value::Mesh(_) => {
                            let path = ValuePath(var_ident, 0);

                            scene_meshes.remove(&path);
                            let gpu_mesh_id = scene_gpu_mesh_handles
                                .remove(&path)
                                .expect("Gpu mesh ID was not tracked")
                                .1;

                            renderer.remove_scene_mesh(gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for index in 0..mesh_array.len() {
                                let path = ValuePath(var_ident, cast_usize(index));

                                scene_meshes.remove(&path);
                                let gpu_mesh_id = scene_gpu_mesh_handles
                                    .remove(&path)
                                    .expect("Gpu mesh ID was not tracked")
                                    .1;

                                renderer.remove_scene_mesh(gpu_mesh_id);
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },

                    PollNotification::UnusedValueAdded(var_ident, value) => match value {
                        Value::Mesh(mesh) => {
                            let gpu_mesh = GpuMesh::from_mesh(&mesh);
                            let gpu_mesh_id = renderer
                                .add_scene_mesh(&gpu_mesh)
                                .expect("Failed to upload scene mesh");

                            let path = ValuePath(var_ident, 0);

                            scene_meshes.insert(path, (false, mesh));
                            scene_gpu_mesh_handles.insert(path, (false, gpu_mesh_id));
                        }
                        Value::MeshArray(mesh_array) => {
                            for (index, mesh) in mesh_array.iter_refcounted().enumerate() {
                                let gpu_mesh = GpuMesh::from_mesh(&mesh);
                                let gpu_mesh_id = renderer
                                    .add_scene_mesh(&gpu_mesh)
                                    .expect("Failed to upload scene mesh");

                                let path = ValuePath(var_ident, index);

                                scene_meshes.insert(path, (false, mesh));
                                scene_gpu_mesh_handles.insert(path, (false, gpu_mesh_id));
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },

                    PollNotification::UnusedValueRemoved(var_ident, value) => match value {
                        Value::Mesh(_) => {
                            let path = ValuePath(var_ident, 0);

                            scene_meshes.remove(&path);
                            let gpu_mesh_id = scene_gpu_mesh_handles
                                .remove(&path)
                                .expect("Gpu mesh ID was not tracked")
                                .1;

                            renderer.remove_scene_mesh(gpu_mesh_id);
                        }
                        Value::MeshArray(mesh_array) => {
                            for index in 0..mesh_array.len() {
                                let path = ValuePath(var_ident, cast_usize(index));

                                scene_meshes.remove(&path);
                                let gpu_mesh_id = scene_gpu_mesh_handles
                                    .remove(&path)
                                    .expect("Gpu mesh ID was not tracked")
                                    .1;

                                renderer.remove_scene_mesh(gpu_mesh_id);
                            }
                        }
                        _ => (/* Ignore other values, we don't display them in the viewport */),
                    },

                    PollNotification::FinishedSuccessfully => {
                        scene_bounding_box = BoundingBox::union(
                            scene_meshes
                                .values()
                                .filter(|(used, _)| viewport_draw_used_values || !used)
                                .map(|(_, mesh)| mesh.bounding_box()),
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
                                .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh))
                                .expect("Failed to add ground plane mesh"),
                        );

                        let camera_radius_max = compute_scene_camera_radius(scene_bounding_box);
                        camera.set_radius_min(0.001 * camera_radius_max);
                        camera.set_radius_max(camera_radius_max);
                        camera.set_znear(0.001 * camera_radius_max);
                        camera.set_zfar(2.0 * camera_radius_max);

                        notifications.push(
                            time,
                            NotificationLevel::Info,
                            "Execution of the Operation pipeline finished successfully.",
                        );
                    }

                    PollNotification::FinishedWithError(error_message) => {
                        notifications.push(
                            time,
                            NotificationLevel::Error,
                            format!(
                                "Execution of the Operation pipeline finished with error: {}",
                                error_message
                            ),
                        );
                    }
                });

                let input_state = input_manager.input_state();
                let ui_frame = ui.prepare_frame(&window);

                #[cfg(not(feature = "dist"))]
                {
                    if input_state.debug_view_cycle {
                        renderer_debug_view = renderer_debug_view.cycle();
                        log::debug!("Cycled debug view to {:?}", renderer_debug_view);
                    }
                }

                if !session.interpreter_busy() {
                    if input_state.prog_run_requested && session.autorun_delay().is_none() {
                        session.interpret();
                    }

                    if input_state.prog_pop_requested {
                        session.pop_prog_stmt(time);
                    }
                }

                if input_state.open_screenshot_options {
                    screenshot_modal_open = true;
                }

                if let Some(([old_x, old_y], [new_x, new_y])) = input_state.camera_pan_ground {
                    camera.pan_ground(old_x, old_y, new_x, new_y);
                }
                if let Some(([old_x, old_y], [new_x, new_y])) = input_state.camera_pan_screen {
                    camera.pan_screen(old_x, old_y, new_x, new_y);
                }
                camera.rotate(input_state.camera_rotate[0], input_state.camera_rotate[1]);
                camera.zoom(input_state.camera_zoom);
                camera.zoom_step(input_state.camera_zoom_steps);

                let menu_status = ui_frame.draw_menu_window(
                    time,
                    &mut screenshot_modal_open,
                    &mut about_modal_open,
                    &mut viewport_draw_mode,
                    &mut viewport_draw_used_values,
                    &mut project_status,
                    &mut session,
                    &mut notifications,
                );

                ui_frame.draw_subdigital_logo(
                    tex_subdigital_logo,
                    width_subdigital_logo,
                    height_subdigital_logo,
                );

                if menu_status.viewport_draw_used_values_changed {
                    scene_bounding_box = BoundingBox::union(
                        scene_meshes
                            .values()
                            .filter(|(used, _)| viewport_draw_used_values || !used)
                            .map(|(_, mesh)| mesh.bounding_box()),
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
                            .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh))
                            .expect("Failed to add ground plane mesh"),
                    );
                }

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

                    for (_, (_, gpu_mesh_handle)) in scene_gpu_mesh_handles.drain() {
                        renderer.remove_scene_mesh(gpu_mesh_handle);
                    }

                    scene_bounding_box = BoundingBox::union(
                        scene_meshes
                            .values()
                            .filter(|(used, _)| viewport_draw_used_values || !used)
                            .map(|(_, mesh)| mesh.bounding_box()),
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
                            .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh))
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
                    log::info!("Saving project at {}", save_path.to_string_lossy());

                    let stmts = session.stmts().to_vec();
                    let project = project::Project { version: 1, stmts };

                    match project::save(&save_path, project) {
                        Ok(save_path) => {
                            let save_path = save_path
                                .as_os_str()
                                .to_str()
                                .expect("Failed to convert save path to str.");

                            project_status.save(&save_path);
                            change_window_title(&window, &project_status);
                            notifications.push(
                                time,
                                NotificationLevel::Info,
                                format!("Project saved as {}", &save_path),
                            );
                        }
                        Err(err) => {
                            log::error!("{}", err);
                            project_status.error = Some(err);
                        }
                    }
                }

                if let Some(open_path) = menu_status.open_path {
                    log::info!("Opening new project at {}", open_path.to_string_lossy());

                    match project::open(&open_path) {
                        Ok(project) => {
                            scene_meshes.clear();

                            for (_, gpu_mesh_handle) in scene_gpu_mesh_handles.drain() {
                                renderer.remove_scene_mesh(gpu_mesh_handle.1);
                            }

                            scene_bounding_box = BoundingBox::union(
                                scene_meshes.values().map(|(_, mesh)| mesh.bounding_box()),
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
                                    .add_scene_mesh(&GpuMesh::from_mesh(&ground_plane_mesh))
                                    .expect("Failed to add ground plane mesh"),
                            );

                            let current_autorun_delay = session.autorun_delay();
                            session = Session::new();
                            session.set_autorun_delay(current_autorun_delay);

                            for stmt in project.stmts {
                                session.push_prog_stmt(time, stmt);
                            }

                            project_status.path = Some(PathBuf::from(&open_path));
                            project_status.changed_since_last_save = false;

                            change_window_title(&window, &project_status);

                            notifications.push(
                                time,
                                NotificationLevel::Info,
                                format!("Opened project {}", &open_path.to_string_lossy()),
                            );
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

                let window_size = window.inner_size();
                let take_screenshot = ui_frame.draw_screenshot_window(
                    &mut screenshot_modal_open,
                    &mut screenshot_options,
                    window_size.width,
                    window_size.height,
                );

                let (tex_logos, width_logos, height_logos) = match options.theme {
                    Theme::Light => (tex_logos_black, width_logos_black, height_logos_black),
                    Theme::Dark => (tex_logos_white, width_logos_white, height_logos_white),
                };
                ui_frame.draw_about_window(
                    &mut about_modal_open,
                    tex_scheme,
                    width_scheme,
                    height_scheme,
                    tex_logos,
                    width_logos,
                    height_logos,
                );

                ui_frame.draw_notifications_window(&notifications);

                if ui_frame.draw_pipeline_window(time, &mut session) {
                    project_status.changed_since_last_save = true;

                    change_window_title(&window, &project_status);
                }

                if ui_frame.draw_operations_window(
                    time,
                    &mut session,
                    &mut notifications,
                    DURATION_AUTORUN_DELAY,
                ) {
                    project_status.changed_since_last_save = true;

                    change_window_title(&window, &project_status);
                }

                if let Some(prevent_overwrite_status) = project_status.prevent_overwrite_status {
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
                                None => {
                                    let path_string = tinyfiledialogs::save_file_dialog_with_filter(
                                        "Save",
                                        project::DEFAULT_NEW_FILENAME,
                                        project::EXTENSION_FILTER,
                                        project::EXTENSION_DESCRIPTION,
                                    );

                                    path_string.map(PathBuf::from)
                                }
                            };

                            if let Some(save_path) = save_path {
                                let stmts = session.stmts().to_vec();
                                let project = project::Project { version: 1, stmts };

                                match project::save(&save_path, project) {
                                    Ok(save_path) => match prevent_overwrite_status {
                                        project::NextAction::Exit => {
                                            *control_flow = winit::event_loop::ControlFlow::Exit
                                        }
                                        project::NextAction::NewProject => {
                                            let save_path = save_path
                                                .as_os_str()
                                                .to_str()
                                                .expect("Failed to convert save path to str.");

                                            project_status.save(&save_path);
                                            project_status.new_requested = true;
                                        }
                                        project::NextAction::OpenProject => {
                                            let save_path = save_path
                                                .as_os_str()
                                                .to_str()
                                                .expect("Failed to convert save path to str.");

                                            project_status.save(&save_path);
                                            project_status.open_requested = true
                                        }
                                    },
                                    Err(err) => {
                                        log::error!("Project save failed: {}", err);

                                        project_status.error = Some(err);
                                    }
                                }
                            }

                            project_status.prevent_overwrite_status = None;
                        }
                        SaveModalResult::Nothing => (),
                    }
                }

                if input_state.camera_reset_viewport || menu_status.reset_viewport {
                    camera_interpolation =
                        Some(CameraInterpolation::new(&camera, &scene_bounding_box, time));
                }

                if menu_status.export_obj {
                    let suggested_filename = match &project_status.path {
                        Some(path) => match path.file_stem() {
                            Some(file_stem) => {
                                Cow::Owned(format!("{}.obj", file_stem.to_string_lossy()))
                            }
                            None => Cow::Borrowed("export.obj"),
                        },
                        None => Cow::Borrowed("export.obj"),
                    };
                    if let Some(path) = tinyfiledialogs::save_file_dialog_with_filter(
                        "Export OBJ",
                        &suggested_filename,
                        &["*.obj"],
                        "Wavefront (.obj)",
                    ) {
                        // FIXME: The session can not provide a name, if the
                        // viewport contains an object constructed by a func
                        // that was already removed from the program. For this
                        // reason we are exporting just the stringified var
                        // ident, if the name is not present.
                        //
                        // Note that this viewport vs session desync will also
                        // affect features like viewport picking. We have to
                        // assume all values in the viewport can potentially be
                        // stale.
                        //
                        // What do we do?
                        let unused_values_iter = scene_meshes
                            .iter()
                            .filter(|(_, (used, _))| !used)
                            .map(|(value_path, (_, mesh))| {
                                if value_path.1 == 0 {
                                    // Do not suffix zero mesh-array index
                                    let name = match session
                                        .var_decl_stmt_index_and_var_name_for_ident(value_path.0)
                                    {
                                        Some((_, name)) => Cow::Borrowed(name),
                                        None => Cow::Owned(value_path.0.to_string()),
                                    };

                                    (name, mesh.as_ref())
                                } else {
                                    // Suffix mesh-array index if nonzero
                                    let name = match session
                                        .var_decl_stmt_index_and_var_name_for_ident(value_path.0)
                                    {
                                        Some((_, name)) => {
                                            Cow::Owned(format!("{} [{}]", name, value_path.1))
                                        }
                                        None => Cow::Owned(format!(
                                            "{} [{}]",
                                            value_path.0, value_path.1,
                                        )),
                                    };

                                    (name, mesh.as_ref())
                                }
                            });

                        let file = File::create(&path).expect("Failed to create OBJ file");
                        let mut writer = BufWriter::new(file);

                        match exporter::export_obj(&mut writer, unused_values_iter, f32::DIGITS) {
                            Ok(()) => {
                                log::info!("OBJ exported to: {}", path);
                                notifications.push(
                                    time,
                                    NotificationLevel::Info,
                                    format!("OBJ exported to: {}", path),
                                );
                            }
                            Err(err) => {
                                log::error!("OBJ export failed: {}", err);
                                notifications.push(
                                    time,
                                    NotificationLevel::Error,
                                    "OBJ export failed",
                                );
                            }
                        }
                    }
                }

                if input_state.close_requested {
                    if project_status.changed_since_last_save {
                        project_status.prevent_overwrite_status = Some(project::NextAction::Exit);
                    } else {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                }

                if let Some(physical_size) = input_state.window_resized {
                    let width = physical_size.width;
                    let height = physical_size.height;
                    log::debug!("Window resized to new physical size {}x{}", width, height);

                    // While it can't be queried, 16 is usually the minimal
                    // dimension of certain types of textures. Creating anything
                    // smaller currently crashes most of our GPU backend/driver
                    // combinations.
                    if width >= 16 && height >= 16 {
                        screenshot_options.width = width;
                        screenshot_options.height = height;
                        camera.set_screen_dimensions(width, height);
                        renderer.set_window_size(width, height);
                    } else {
                        log::warn!("Ignoring new window physical size {}x{}", width, height);
                    }
                }

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
                notifications.update(time);

                // -- Draw to offscreen render target for screenshots --

                if take_screenshot {
                    log::info!(
                        "Capturing screenshot with dimensions {}x{} and transparency {}",
                        screenshot_options.width,
                        screenshot_options.height,
                        screenshot_options.transparent,
                    );

                    let screenshot_render_target = renderer.add_offscreen_render_target(
                        screenshot_options.width,
                        screenshot_options.height,
                    );

                    let mut screenshot_camera = camera.clone();
                    screenshot_camera
                        .set_screen_dimensions(screenshot_options.width, screenshot_options.height);

                    let screenshot_clear_color = if screenshot_options.transparent {
                        [0.0; 4]
                    } else {
                        clear_color
                    };

                    let mut screenshot_command_buffer = renderer.begin_command_buffer(
                        screenshot_clear_color,
                        Some(&screenshot_render_target),
                        false,
                    );
                    screenshot_command_buffer.set_light(&compute_scene_light(scene_bounding_box));
                    screenshot_command_buffer.set_camera_matrices(
                        &screenshot_camera.projection_matrix(),
                        &screenshot_camera.view_matrix(),
                    );

                    // For screenshots, we don't need to cast shadows, and we
                    // don't render the ground on purpose.
                    match viewport_draw_mode {
                        ViewportDrawMode::Wireframe => {
                            screenshot_command_buffer.draw_meshes_to_render_target(
                                scene_gpu_mesh_handles
                                    .values()
                                    .filter(|(used, _)| viewport_draw_used_values || !used)
                                    .map(|(used, handle)| {
                                        if *used {
                                            (handle, Material::TransparentMatcapShaded, false)
                                        } else {
                                            (handle, Material::Edges, true)
                                        }
                                    }),
                            );
                        }
                        ViewportDrawMode::Shaded => {
                            screenshot_command_buffer.draw_meshes_to_render_target(
                                scene_gpu_mesh_handles
                                    .values()
                                    .filter(|(used, _)| viewport_draw_used_values || !used)
                                    .map(|(used, handle)| {
                                        if *used {
                                            (handle, Material::TransparentMatcapShaded, false)
                                        } else {
                                            (handle, Material::MatcapShaded, true)
                                        }
                                    }),
                            );
                        }
                        ViewportDrawMode::ShadedWireframe => {
                            screenshot_command_buffer.draw_meshes_to_render_target(
                                scene_gpu_mesh_handles
                                    .values()
                                    .filter(|(used, _)| viewport_draw_used_values || !used)
                                    .map(|(used, handle)| {
                                        if *used {
                                            (handle, Material::TransparentMatcapShaded, false)
                                        } else {
                                            (handle, Material::MatcapShadedEdges, true)
                                        }
                                    }),
                            );
                        }
                        ViewportDrawMode::ShadedWireframeXray => {
                            screenshot_command_buffer.draw_meshes_to_render_target(
                                scene_gpu_mesh_handles
                                    .values()
                                    .filter(|(used, _)| viewport_draw_used_values || !used)
                                    .map(|(used, handle)| {
                                        if *used {
                                            (handle, Material::TransparentMatcapShaded, false)
                                        } else {
                                            (handle, Material::MatcapShaded, true)
                                        }
                                    }),
                            );

                            screenshot_command_buffer.draw_meshes_to_render_target(
                                scene_gpu_mesh_handles
                                    .values()
                                    .filter(|(used, _)| !used)
                                    .map(|(_, handle)| (handle, Material::EdgesXray, false)),
                            );
                        }
                    }

                    screenshot_command_buffer.submit();

                    let mapping = renderer.offscreen_render_target_data(&screenshot_render_target);
                    let (width, height) = mapping.dimensions();
                    let data = mapping.data();

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
                    }

                    if let Some(mut path) = dirs::picture_dir() {
                        path.push(format!(
                            "hurban_selector-{}.png",
                            chrono::Local::now().format("%Y-%m-%d-%H-%M-%S"),
                        ));

                        let file = File::create(&path).expect("Failed to create PNG file");
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
                        notifications.push(
                            time,
                            NotificationLevel::Info,
                            format!("Screenshot saved in {}", path_str),
                        );
                    } else {
                        log::error!("Failed to find picture directory");
                        notifications.push(
                            time,
                            NotificationLevel::Warn,
                            "Failed to find picture directory",
                        );
                    }

                    renderer.remove_offscreen_render_target(screenshot_render_target);
                }

                // -- Draw to viewport --

                let imgui_draw_data = ui_frame.render(&window);

                let mut window_command_buffer =
                    renderer.begin_command_buffer(clear_color, None, true);
                window_command_buffer.set_light(&compute_scene_light(scene_bounding_box));
                window_command_buffer
                    .set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());

                match viewport_draw_mode {
                    ViewportDrawMode::Wireframe => {
                        window_command_buffer.draw_meshes_to_render_target(
                            scene_gpu_mesh_handles
                                .values()
                                .filter(|(used, _)| viewport_draw_used_values || !used)
                                .map(|(used, handle)| {
                                    if *used {
                                        (handle, Material::TransparentMatcapShaded, false)
                                    } else {
                                        (handle, Material::Edges, true)
                                    }
                                }),
                        );
                    }
                    ViewportDrawMode::Shaded => {
                        window_command_buffer.draw_meshes_to_render_target(
                            scene_gpu_mesh_handles
                                .values()
                                .filter(|(used, _)| viewport_draw_used_values || !used)
                                .map(|(used, handle)| {
                                    if *used {
                                        (handle, Material::TransparentMatcapShaded, false)
                                    } else {
                                        (handle, Material::MatcapShaded, true)
                                    }
                                }),
                        );
                    }
                    ViewportDrawMode::ShadedWireframe => {
                        window_command_buffer.draw_meshes_to_render_target(
                            scene_gpu_mesh_handles
                                .values()
                                .filter(|(used, _)| viewport_draw_used_values || !used)
                                .map(|(used, handle)| {
                                    if *used {
                                        (handle, Material::TransparentMatcapShaded, false)
                                    } else {
                                        (handle, Material::MatcapShadedEdges, true)
                                    }
                                }),
                        );
                    }
                    ViewportDrawMode::ShadedWireframeXray => {
                        window_command_buffer.draw_meshes_to_render_target(
                            scene_gpu_mesh_handles
                                .values()
                                .filter(|(used, _)| viewport_draw_used_values || !used)
                                .map(|(used, handle)| {
                                    if *used {
                                        (handle, Material::TransparentMatcapShaded, false)
                                    } else {
                                        (handle, Material::MatcapShaded, true)
                                    }
                                }),
                        );

                        window_command_buffer.draw_meshes_to_render_target(
                            scene_gpu_mesh_handles
                                .values()
                                .filter(|(used, _)| !used)
                                .map(|(_, handle)| (handle, Material::EdgesXray, false)),
                        );
                    }
                }

                window_command_buffer.draw_meshes_to_render_target(
                    ground_plane_gpu_mesh_handle
                        .iter()
                        .map(|handle| (handle, Material::FlatWithShadows, false)),
                );

                #[cfg(not(feature = "dist"))]
                match renderer_debug_view {
                    RendererDebugView::Off => {
                        window_command_buffer.blit_render_target_to_swap_chain();
                    }
                    RendererDebugView::ShadowMap => {
                        window_command_buffer.blit_shadow_map_to_swap_chain();
                    }
                }

                #[cfg(feature = "dist")]
                window_command_buffer.blit_render_target_to_swap_chain();

                window_command_buffer.draw_ui_to_swap_chain(imgui_draw_data);
                window_command_buffer.submit();
            }

            winit::event::Event::RedrawRequested(_) => {
                // FIXME: @Correctness We don't answer redraw requests and
                // instead draw in the "events cleared", because imgui ui
                // containing the draw list does not live long
                // enough. Refactoring to on-demand rendering could unlock some
                // energy-savings in the future. Also note that winit coalesces
                // `RedrawRequested` events, so having `wgpu::PresentMode::Fifo`
                // *shouldn't* mean we pile up redraws if the OS notifies us to
                // redraw.
            }

            _ => (),
        }

        // Note: this is just best effort basis. If imgui is only one event
        // ahead, it still might not report `want_capture_mouse` and
        // `want_capture_keyboard` correctly. New winit does not implement
        // `Clone` for events though, so buffering would be more complicated. In
        // practice, latent ui capture state shouldn't be an issue.
        ui.process_event(&event, &window);
        ui_want_capture_keyboard = ui.want_capture_keyboard();
        ui_want_capture_mouse = ui.want_capture_mouse();

        input_manager.process_event(&event, ui_want_capture_keyboard, ui_want_capture_mouse);
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

fn decode_image_rgba8_unorm(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let image = image::load_from_memory(data).expect("Failed to decode image.");
    let (width, height) = image.dimensions();
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);

    for (_, _, pixel) in image.pixels() {
        rgba.extend_from_slice(&pixel.to_rgba().0);
    }

    (rgba, width, height)
}

fn compute_scene_camera_radius(scene_bounding_box: BoundingBox<f32>) -> f32 {
    scene_bounding_box.diagonal().norm() * 10.0
}

fn compute_scene_light(scene_bounding_box: BoundingBox<f32>) -> DirectionalLight {
    // Extend the bounding box to always contain a point with Z=0 so that we can
    // cast shadows on the ground plane.
    let scene_center = scene_bounding_box.center();
    let point_on_ground = Point3::new(scene_center.x, scene_center.y, 0.0);

    let bounding_box = BoundingBox::from_points(
        [
            scene_bounding_box.minimum_point(),
            scene_bounding_box.maximum_point(),
            point_on_ground,
        ]
        .iter()
        .copied(),
    )
    .expect("Must produce a bounding box for non-empty iterator");

    let diagonal = bounding_box.diagonal() * 1.1;
    let width = diagonal.x.max(diagonal.y);
    let height = diagonal.z;

    DirectionalLight {
        position: bounding_box.center() + Vector3::new(0.0, 0.0, height / 2.0),
        direction: Vector3::new(0.0, 0.0, -height),
        min_range: 0.001,
        max_range: height,
        width,
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
    let join_str = if project_status.changed_since_last_save {
        " - *"
    } else {
        " - "
    };

    window.set_title(&[BASE_WINDOW_TITLE, filename].join(join_str));
}

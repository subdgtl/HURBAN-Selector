use std::cell::RefCell;
use std::f32;
use std::time::{Duration, Instant};

use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::convert::{cast_u8_color_to_f32, clamp_cast_i32_to_u32, clamp_cast_u32_to_i32};
use crate::interpreter::{ast, LogMessageLevel, ParamRefinement, Ty};
use crate::notifications::{NotificationLevel, Notifications};
use crate::project;
use crate::renderer::DrawMeshMode;
use crate::session::Session;

const FONT_OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/SpaceMono-Regular.ttf");
const FONT_OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/SpaceMono-Bold.ttf");

const WRAP_POS_TOOLTIP_TEXT_PIXELS: f32 = 400.0;
const WRAP_POS_CONSOLE_TEXT_PIXELS: f32 = 380.0;

const MARGIN: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Dark,
    Funky,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenshotOptions {
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
}

struct FontIds {
    regular: imgui::FontId,
    bold: imgui::FontId,
}

struct Colors {
    special_button_text: [f32; 4],
    special_button: [f32; 4],
    special_button_hovered: [f32; 4],
    special_button_active: [f32; 4],
    combo_box_selected_item: [f32; 4],
    combo_box_selected_item_hovered: [f32; 4],
    combo_box_selected_item_active: [f32; 4],
    log_message_info: [f32; 4],
    log_message_warn: [f32; 4],
    log_message_error: [f32; 4],
    header_error: [f32; 4],
    header_error_hovered: [f32; 4],
}

#[derive(Debug, Default)]
struct PipelineWindowState {
    autoscroll: bool,
}

#[derive(Debug, Default)]
struct NotificationsState {
    notifications_count: usize,
}

#[derive(Debug, Default)]
struct ConsoleState {
    message_count: usize,
}

pub struct MenuStatus {
    pub reset_viewport: bool,
    pub save_path: Option<String>,
    pub open_path: Option<String>,
    pub show_prevent_override_modal: bool,
}

pub enum SaveModalResult {
    Save,
    DontSave,
    Cancel,
    Nothing,
}

/// Thin wrapper around imgui and its winit platform. Its main responsibilty
/// is to create UI frames which draw the UI itself.
pub struct Ui {
    imgui_context: imgui::Context,
    imgui_winit_platform: WinitPlatform,
    font_ids: FontIds,
    colors: Colors,
    pipeline_window_state: RefCell<PipelineWindowState>,
    notifications_state: RefCell<NotificationsState>,
    console_state: RefCell<Vec<ConsoleState>>,

    /// A preallocated string buffer used for imgui strings in the
    /// UI. Every user of this buffer has the responsibility to clear
    /// it afterwards.
    global_imstring_buffer: RefCell<imgui::ImString>,
}

impl Ui {
    /// Initializes imgui with default settings for our application.
    pub fn new(window: &winit::window::Window, theme: Theme) -> Self {
        let mut imgui_context = imgui::Context::create();
        let mut style = imgui_context.style_mut();
        let mut colors = Colors {
            special_button_text: style[imgui::StyleColor::Text],
            special_button: [0.2, 0.7, 0.3, 1.0],
            special_button_hovered: [0.4, 0.8, 0.5, 1.0],
            special_button_active: [0.1, 0.5, 0.2, 1.0],
            combo_box_selected_item: style[imgui::StyleColor::Header],
            combo_box_selected_item_hovered: style[imgui::StyleColor::HeaderHovered],
            combo_box_selected_item_active: style[imgui::StyleColor::HeaderActive],
            log_message_info: [0.70, 0.70, 0.70, 1.0],
            log_message_warn: [0.80, 0.80, 0.05, 1.0],
            log_message_error: [1.0, 0.15, 0.05, 1.0],
            header_error: [0.85, 0.15, 0.05, 0.4],
            header_error_hovered: [1.00, 0.15, 0.05, 0.4],
        };

        style.window_padding = [4.0, 4.0];
        style.frame_padding = [4.0, 2.0];
        style.item_spacing = [2.0, 2.0];
        style.item_inner_spacing = [2.0, 2.0];
        style.indent_spacing = 8.0;

        style.scrollbar_size = 8.0;
        style.grab_min_size = 4.0;

        style.window_rounding = 3.0;
        style.frame_rounding = 3.0;
        style.scrollbar_rounding = 3.0;
        style.grab_rounding = 3.0;

        if theme == Theme::Funky {
            style.window_rounding = 0.0;
            style.frame_rounding = 0.0;
            style.scrollbar_rounding = 0.0;
            style.grab_rounding = 0.0;

            let light = cast_u8_color_to_f32([0xea, 0xe7, 0xe1, 0xff]);
            let light_transparent = cast_u8_color_to_f32([0xea, 0xe7, 0xe1, 0x40]);
            let orange = cast_u8_color_to_f32([0xf2, 0x80, 0x37, 0xff]);
            let orange_light = cast_u8_color_to_f32([0xf2, 0xac, 0x79, 0xff]);
            let orange_light_transparent = cast_u8_color_to_f32([0xf2, 0xac, 0x79, 0x40]);
            let orange_dark = cast_u8_color_to_f32([0xd0, 0x5d, 0x20, 0xff]);
            let orange_dark_transparent = cast_u8_color_to_f32([0xd0, 0x5d, 0x20, 0x40]);
            let green_light = [0.4, 0.8, 0.5, 1.0];
            let green_light_transparent = [0.4, 0.8, 0.5, 0.4];
            let green_dark = [0.1, 0.5, 0.2, 1.0];

            style[imgui::StyleColor::Text] = orange;
            style[imgui::StyleColor::TextDisabled] = orange_light;
            style[imgui::StyleColor::WindowBg] = light_transparent;
            style[imgui::StyleColor::PopupBg] = light;
            style[imgui::StyleColor::Border] = light_transparent;
            style[imgui::StyleColor::FrameBg] = light_transparent;
            style[imgui::StyleColor::FrameBgHovered] = light_transparent;
            style[imgui::StyleColor::FrameBgActive] = light_transparent;
            style[imgui::StyleColor::TitleBg] = light_transparent;
            style[imgui::StyleColor::TitleBgActive] = light_transparent;
            style[imgui::StyleColor::TitleBgCollapsed] = light_transparent;
            style[imgui::StyleColor::MenuBarBg] = light_transparent;
            style[imgui::StyleColor::ScrollbarBg] = light_transparent;
            style[imgui::StyleColor::ScrollbarGrab] = orange_dark;
            style[imgui::StyleColor::ScrollbarGrabHovered] = orange;
            style[imgui::StyleColor::ScrollbarGrabActive] = orange_light;
            style[imgui::StyleColor::CheckMark] = orange;
            style[imgui::StyleColor::SliderGrab] = orange;
            style[imgui::StyleColor::SliderGrabActive] = orange_light;
            style[imgui::StyleColor::Button] = light_transparent;
            style[imgui::StyleColor::ButtonHovered] = orange_light_transparent;
            style[imgui::StyleColor::ButtonActive] = orange_dark_transparent;
            style[imgui::StyleColor::Header] = light_transparent;
            style[imgui::StyleColor::HeaderHovered] = light_transparent;
            style[imgui::StyleColor::HeaderActive] = light_transparent;
            style[imgui::StyleColor::Separator] = orange_light;
            style[imgui::StyleColor::SeparatorHovered] = orange_light;
            style[imgui::StyleColor::SeparatorActive] = orange_light;
            style[imgui::StyleColor::ResizeGrip] = orange;
            style[imgui::StyleColor::ResizeGripHovered] = orange_light;
            style[imgui::StyleColor::ResizeGripActive] = orange_light;
            style[imgui::StyleColor::Tab] = light_transparent;
            style[imgui::StyleColor::TabHovered] = light_transparent;
            style[imgui::StyleColor::TabActive] = light_transparent;
            style[imgui::StyleColor::TabUnfocused] = light_transparent;
            style[imgui::StyleColor::TabUnfocusedActive] = light_transparent;
            style[imgui::StyleColor::PlotLines] = orange;
            style[imgui::StyleColor::TextSelectedBg] = orange_light_transparent;
            style[imgui::StyleColor::NavHighlight] = light_transparent;

            colors.special_button_text = orange;
            colors.special_button = green_light_transparent;
            colors.special_button_hovered = green_light;
            colors.special_button_active = green_dark;

            colors.combo_box_selected_item = light;
            colors.combo_box_selected_item_hovered = orange_light;
            colors.combo_box_selected_item_active = orange_dark;

            colors.log_message_warn = [0.90, 0.75, 0.05, 1.0];

            colors.header_error = [0.9, 0.0, 0.0, 0.2];
            colors.header_error_hovered = [1.0, 0.0, 0.0, 0.3];
        }

        imgui_context.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui_context);

        platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Default);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (15.0 * hidpi_factor) as f32;

        let regular_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: FONT_OPENSANS_REGULAR_BYTES,
                size_pixels: font_size,
                config: None,
            }]);
        let bold_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: FONT_OPENSANS_BOLD_BYTES,
                size_pixels: font_size,
                config: None,
            }]);

        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        Ui {
            imgui_context,
            imgui_winit_platform: platform,
            font_ids: FontIds {
                regular: regular_font_id,
                bold: bold_font_id,
            },
            colors,
            pipeline_window_state: RefCell::new(PipelineWindowState::default()),
            console_state: RefCell::new(Vec::new()),
            notifications_state: RefCell::new(NotificationsState::default()),
            global_imstring_buffer: RefCell::new(imgui::ImString::with_capacity(1024)),
        }
    }

    pub fn fonts(&mut self) -> imgui::FontAtlasRefMut {
        self.imgui_context.fonts()
    }

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        self.imgui_winit_platform
            .handle_event(self.imgui_context.io_mut(), window, &event);
    }

    pub fn prepare_frame(&mut self, window: &winit::window::Window) -> UiFrame {
        self.imgui_winit_platform
            .prepare_frame(self.imgui_context.io_mut(), window)
            .expect("Failed to start imgui frame");

        UiFrame {
            imgui_winit_platform: &self.imgui_winit_platform,
            imgui_ui: self.imgui_context.frame(),
            font_ids: &self.font_ids,
            colors: &self.colors,
            console_state: &self.console_state,
            pipeline_window_state: &self.pipeline_window_state,
            notifications_state: &self.notifications_state,
            global_imstring_buffer: &self.global_imstring_buffer,
        }
    }

    pub fn set_delta_time(&mut self, duration_last_frame_s: f32) {
        self.imgui_context.io_mut().delta_time = duration_last_frame_s;
    }
}

/// This structure is responsible for drawing and rendering of a single UI
/// frame.
pub struct UiFrame<'a> {
    imgui_winit_platform: &'a WinitPlatform,
    imgui_ui: imgui::Ui<'a>,
    font_ids: &'a FontIds,
    colors: &'a Colors,
    console_state: &'a RefCell<Vec<ConsoleState>>,
    pipeline_window_state: &'a RefCell<PipelineWindowState>,
    notifications_state: &'a RefCell<NotificationsState>,
    global_imstring_buffer: &'a RefCell<imgui::ImString>,
}

impl<'a> UiFrame<'a> {
    pub fn want_capture_keyboard(&self) -> bool {
        self.imgui_ui.io().want_capture_keyboard
    }

    pub fn want_capture_mouse(&self) -> bool {
        self.imgui_ui.io().want_capture_mouse
    }

    pub fn render(self, window: &winit::window::Window) -> &'a imgui::DrawData {
        self.imgui_winit_platform
            .prepare_render(&self.imgui_ui, window);
        self.imgui_ui.render()
    }

    pub fn draw_screenshot_window(
        &self,
        screenshot_modal_open: &mut bool,
        screenshot_options: &mut ScreenshotOptions,
        viewport_width: u32,
        viewport_height: u32,
    ) -> bool {
        let ui = &self.imgui_ui;

        let window_name = imgui::im_str!("Screenshot");
        if *screenshot_modal_open {
            ui.open_popup(window_name);
        }

        let mut take_screenshot_clicked = false;

        let viewport_width_f32 = viewport_width as f32;
        let viewport_height_f32 = viewport_height as f32;
        let mut viewport_scale = [
            screenshot_options.width as f32 / viewport_width_f32,
            screenshot_options.height as f32 / viewport_height_f32,
        ];

        let bold_font_token = ui.push_font(self.font_ids.bold);
        ui.popup_modal(window_name)
            .opened(screenshot_modal_open)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .always_auto_resize(true)
            .build(|| {
                let regular_font_token = ui.push_font(self.font_ids.regular);

                let mut dimensions = [
                    clamp_cast_u32_to_i32(screenshot_options.width),
                    clamp_cast_u32_to_i32(screenshot_options.height),
                ];

                if ui
                    .input_int2(imgui::im_str!("Dimensions"), &mut dimensions)
                    .build()
                {
                    screenshot_options.width = clamp_cast_i32_to_u32(dimensions[0]);
                    screenshot_options.height = clamp_cast_i32_to_u32(dimensions[1]);
                }

                if ui
                    .input_float2(imgui::im_str!("Scale"), &mut viewport_scale)
                    .build()
                {
                    screenshot_options.width = clamp_cast_i32_to_u32(
                        (viewport_width_f32 * viewport_scale[0]).round() as i32,
                    );
                    screenshot_options.height = clamp_cast_i32_to_u32(
                        (viewport_height_f32 * viewport_scale[1]).round() as i32,
                    );
                }

                ui.checkbox(
                    imgui::im_str!("Transparent"),
                    &mut screenshot_options.transparent,
                );

                ui.text(imgui::im_str!(
                    "Attempting to take screenshots may crash the program."
                ));
                ui.text(imgui::im_str!("Be sure to save your work."));

                if ui.button(imgui::im_str!("Take Screenshot"), [0.0, 0.0]) {
                    take_screenshot_clicked = true;
                }

                regular_font_token.pop(ui);
            });
        bold_font_token.pop(ui);

        if take_screenshot_clicked {
            *screenshot_modal_open = false;
        }

        take_screenshot_clicked
    }

    pub fn draw_notifications_window(&self, notifications: &Notifications) {
        let notifications_count = notifications.iter().count();
        if notifications_count == 0 {
            self.notifications_state.borrow_mut().notifications_count = 0;
            return;
        }

        let ui = &self.imgui_ui;

        const NOTIFICATIONS_WINDOW_WIDTH: f32 = 400.0;
        const NOTIFICATIONS_WINDOW_HEIGHT_MULT: f32 = 0.12;

        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let notifications_window_height =
            window_inner_height * NOTIFICATIONS_WINDOW_HEIGHT_MULT - MARGIN;
        let notifications_window_vertical_position =
            MARGIN * 2.0 + (1.0 - NOTIFICATIONS_WINDOW_HEIGHT_MULT) * window_inner_height;

        let color_token = ui.push_style_colors(&[
            (imgui::StyleColor::Border, [0.0, 0.0, 0.0, 0.1]),
            (imgui::StyleColor::WindowBg, [0.0, 0.0, 0.0, 0.1]),
        ]);

        imgui::Window::new(imgui::im_str!("Notifications"))
            .title_bar(false)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size(
                [NOTIFICATIONS_WINDOW_WIDTH, notifications_window_height],
                imgui::Condition::Always,
            )
            .position(
                [
                    window_inner_width + MARGIN - NOTIFICATIONS_WINDOW_WIDTH,
                    notifications_window_vertical_position,
                ],
                imgui::Condition::Always,
            )
            .build(ui, || {
                for notification in notifications.iter() {
                    let text_color_token = match notification.level {
                        NotificationLevel::Info => ui.push_style_color(
                            imgui::StyleColor::Text,
                            self.colors.log_message_info,
                        ),
                        NotificationLevel::Warn => ui.push_style_color(
                            imgui::StyleColor::Text,
                            self.colors.log_message_warn,
                        ),
                    };

                    ui.text_wrapped(&imgui::im_str!("{}", notification.text));

                    text_color_token.pop(ui);
                }

                let mut notifications_state = self.notifications_state.borrow_mut();
                if notifications_count != notifications_state.notifications_count {
                    notifications_state.notifications_count = notifications_count;
                    ui.set_scroll_here_y();
                }
            });

        color_token.pop(ui);
    }

    pub fn draw_menu_window(
        &self,
        screenshot_modal_open: &mut bool,
        draw_mode: &mut DrawMeshMode,
        project_status: &mut project::ProjectStatus,
    ) -> MenuStatus {
        let ui = &self.imgui_ui;
        let mut status = MenuStatus {
            reset_viewport: false,
            save_path: None,
            open_path: None,
            show_prevent_override_modal: false,
        };

        const UTILITIES_WINDOW_WIDTH: f32 = 150.0;
        const UTILITIES_WINDOW_HEIGHT: f32 = 232.0;
        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;

        let bold_font_token = ui.push_font(self.font_ids.bold);
        imgui::Window::new(imgui::im_str!("Menu"))
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size(
                [UTILITIES_WINDOW_WIDTH, UTILITIES_WINDOW_HEIGHT],
                imgui::Condition::Always,
            )
            .position(
                [window_inner_width + MARGIN - UTILITIES_WINDOW_WIDTH, MARGIN],
                imgui::Condition::Always,
            )
            .build(ui, || {
                let regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));

                ui.radio_button(imgui::im_str!("Shaded"), draw_mode, DrawMeshMode::Shaded);
                ui.radio_button(imgui::im_str!("Edges"), draw_mode, DrawMeshMode::Edges);
                ui.radio_button(
                    imgui::im_str!("Shaded with Edges"),
                    draw_mode,
                    DrawMeshMode::ShadedEdges,
                );
                ui.radio_button(
                    imgui::im_str!("X-RAY"),
                    draw_mode,
                    DrawMeshMode::ShadedEdgesXray,
                );

                status.reset_viewport =
                    ui.button(imgui::im_str!("Reset Viewport"), [-f32::MIN_POSITIVE, 0.0]);

                if ui.button(imgui::im_str!("Screenshot"), [-f32::MIN_POSITIVE, 0.0]) {
                    *screenshot_modal_open = true;
                }

                let ext_description =
                    &format!("HURBAN Selector project (.{})", project::PROJECT_EXTENSION);
                let ext_filter: &[&str] = &[&format!("*.{}", project::PROJECT_EXTENSION)];

                if ui.button(imgui::im_str!("Save project"), [-f32::MIN_POSITIVE, 0.0]) {
                    match project_status.path.clone() {
                        Some(project_path_str) => {
                            status.save_path = Some(project_path_str.to_string())
                        }
                        None => {
                            if let Some(path) = tinyfiledialogs::save_file_dialog_with_filter(
                                "Save project",
                                &format!("new_project.{}", project::PROJECT_EXTENSION),
                                ext_filter,
                                ext_description,
                            ) {
                                status.save_path = Some(path);
                            }
                        }
                    }
                }

                if ui.button(
                    imgui::im_str!("Save project as..."),
                    [-f32::MIN_POSITIVE, 0.0],
                ) {
                    if let Some(path) = tinyfiledialogs::save_file_dialog_with_filter(
                        "Save project",
                        &format!("new_project.{}", project::PROJECT_EXTENSION),
                        ext_filter,
                        ext_description,
                    ) {
                        status.save_path = Some(path);
                    }
                }

                if ui.button(imgui::im_str!("Open project"), [-f32::MIN_POSITIVE, 0.0])
                    || project_status.open_requested
                {
                    if project_status.changed_since_last_save
                        && project_status.prevent_overwrite_status.is_none()
                    {
                        status.show_prevent_override_modal = true;
                    } else if let Some(path) = tinyfiledialogs::open_file_dialog(
                        "Open project",
                        "",
                        Some((ext_filter, ext_description)),
                    ) {
                        status.open_path = Some(path);
                    }

                    project_status.prevent_overwrite_status = None;
                    project_status.open_requested = false;
                }

                regular_font_token.pop(ui);
            });
        bold_font_token.pop(ui);

        status
    }

    pub fn draw_error_modal(&self, project_error: &Option<project::ProjectError>) -> bool {
        let ui = &self.imgui_ui;
        let mut modal_closed = false;

        ui.open_popup(imgui::im_str!("Error"));
        ui.popup_modal(imgui::im_str!("Error"))
            .resizable(false)
            .build(|| {
                let error_message = project_error
                    .clone()
                    .expect("Failed to read project error.")
                    .to_string();

                ui.text(error_message);

                if ui.button(imgui::im_str!("OK"), [0.0, 0.0]) {
                    modal_closed = true;

                    ui.close_current_popup();
                }
            });

        modal_closed
    }

    pub fn draw_prevent_overwrite_modal(&self) -> SaveModalResult {
        let ui = &self.imgui_ui;
        let mut save_modal_result = SaveModalResult::Nothing;

        ui.open_popup(imgui::im_str!("Unsaved changes"));
        ui.popup_modal(imgui::im_str!("Unsaved changes"))
            .resizable(false)
            .build(|| {
                ui.text("Your changes will be lost if you don't save them.");

                if ui.button(imgui::im_str!("Don't save"), [0.0, 0.0]) {
                    save_modal_result = SaveModalResult::DontSave;

                    ui.close_current_popup();
                }

                if ui.button(imgui::im_str!("Cancel"), [0.0, 0.0]) {
                    save_modal_result = SaveModalResult::Cancel;

                    ui.close_current_popup();
                }

                if ui.button(imgui::im_str!("Save"), [0.0, 0.0]) {
                    save_modal_result = SaveModalResult::Save;

                    ui.close_current_popup();
                }
            });

        save_modal_result
    }

    pub fn draw_save_dialog(&self) -> Option<String> {
        let ext_description = &format!("HURBAN Selector project (.{})", project::PROJECT_EXTENSION);
        let ext_filter: &[&str] = &[&format!("*.{}", project::PROJECT_EXTENSION)];

        tinyfiledialogs::save_file_dialog_with_filter(
            "Save project",
            &format!("new_project.{}", project::PROJECT_EXTENSION),
            ext_filter,
            ext_description,
        )
    }

    // FIXME: @Refactoring Refactor this once we have full-featured
    // functionality. Until then, this is exploratory code and we
    // don't care.
    #[allow(clippy::cognitive_complexity)]
    pub fn draw_pipeline_window(&self, current_time: Instant, session: &mut Session) -> bool {
        let ui = &self.imgui_ui;
        self.console_state
            .borrow_mut()
            .resize_with(session.stmts().len(), Default::default);

        let function_table = session.function_table();

        const PIPELINE_WINDOW_WIDTH: f32 = 400.0;
        const PIPELINE_WINDOW_HEIGHT_MULT: f32 = 0.7;

        let window_logical_size = ui.io().display_size;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let pipeline_window_height = window_inner_height * PIPELINE_WINDOW_HEIGHT_MULT;

        let interpreter_busy = session.interpreter_busy();
        let mut change = None;

        let bold_font_token = ui.push_font(self.font_ids.bold);
        imgui::Window::new(imgui::im_str!("Sequence of operations"))
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size([PIPELINE_WINDOW_WIDTH, pipeline_window_height], imgui::Condition::Always)
            .position([MARGIN, MARGIN], imgui::Condition::Always)
            .build(ui, || {
                let regular_font_token = ui.push_font(self.font_ids.regular);
                for (stmt_index, stmt) in session.stmts().iter().enumerate() {
                    match stmt {
                        ast::Stmt::VarDecl(var_decl) => {
                            let call_expr = var_decl.init_expr();
                            let func_ident = call_expr.ident();
                            let func = &function_table[&func_ident];

                            let error = session.error_at_stmt(stmt_index);
                            let error_color_token = if error.is_some() {
                                Some(ui.push_style_colors(&[
                                    (imgui::StyleColor::Header, self.colors.header_error),
                                    (imgui::StyleColor::HeaderHovered, self.colors.header_error_hovered),
                                    (imgui::StyleColor::HeaderActive, self.colors.header_error_hovered),
                                ]))
                            } else {
                                None
                            };

                            let collapsing_header_open = ui
                                .collapsing_header(&imgui::im_str!(
                                    "#{} {} ##{}",
                                    stmt_index + 1,
                                    func.info().name,
                                    stmt_index
                                ))
                                .default_open(true)
                                .build();

                            if ui.is_item_hovered() {
                                if let Some(error) = error {
                                    ui.tooltip(|| {
                                        let wrap_token = ui
                                            .push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);

                                        let mut imstring_buffer = self.global_imstring_buffer
                                            .borrow_mut();

                                        // FIXME: @Optimization don't allocate intermediate
                                        // string and use `write!` once imgui-rs implements
                                        // `io::Write` for `ImString`.
                                        // https://github.com/Gekkio/imgui-rs/issues/290
                                        imstring_buffer.push_str(&error.to_string());

                                        ui.text_colored(
                                            [1.0, 0.0, 0.0, 1.0],
                                            &*imstring_buffer,
                                        );

                                        imstring_buffer.clear();

                                        wrap_token.pop(ui);
                                    });
                                } else if !func.info().description.is_empty() {
                                    ui.tooltip(|| {
                                        let wrap_token = ui
                                            .push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                                        ui.text(func.info().description);
                                        wrap_token.pop(ui);
                                    })
                                }
                            }

                            if let Some(color_token) = error_color_token {
                                color_token.pop(ui);
                            }

                            if collapsing_header_open {
                                ui.indent();

                                assert_eq!(
                                    call_expr.args().len(),
                                    func.param_info().len(),
                                    "Function call must be generated with correct number of arguments",
                                );

                                let operation_arg_style_tokens = if interpreter_busy {
                                    Some(push_disabled_style(ui))
                                } else {
                                    None
                                };

                                for (arg_index, (param_info, arg)) in func
                                    .param_info()
                                    .iter()
                                    .zip(call_expr.args().iter())
                                    .enumerate()
                                {
                                    let input_label = imgui::im_str!(
                                        "{}##{}-{}",
                                        &param_info.name,
                                        stmt_index,
                                        arg_index
                                    );

                                    match param_info.refinement {
                                        ParamRefinement::Boolean(_) => {
                                            let mut boolean_lit =
                                                arg.unwrap_literal().unwrap_boolean();

                                            if ui.checkbox(&input_label, &mut boolean_lit) {
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Boolean(
                                                        boolean_lit,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Int(param_refinement_int) => {
                                            let mut int_lit = arg.unwrap_literal().unwrap_int();

                                            if ui.input_int(&input_label, &mut int_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                int_lit = param_refinement_int.clamp(int_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Int(int_lit)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Uint(param_refinement_uint) => {
                                            let uint_lit = arg.unwrap_literal().unwrap_uint();
                                            let mut int_value = clamp_cast_u32_to_i32(uint_lit);

                                            if ui.input_int(&input_label, &mut int_value)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                let uint_value = clamp_cast_i32_to_u32(int_value);
                                                let uint_value = param_refinement_uint.clamp(uint_value);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Uint(uint_value)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float(param_refinement_float) => {
                                            let mut float_lit = arg.unwrap_literal().unwrap_float();

                                            if ui.input_float(&input_label, &mut float_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                float_lit = param_refinement_float.clamp(float_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float(float_lit)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float2(param_refinement_float2) => {
                                            let mut float2_lit =
                                                arg.unwrap_literal().unwrap_float2();

                                            if ui
                                                .input_float2(&input_label, &mut float2_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                float2_lit = param_refinement_float2.clamp(float2_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float2(
                                                        float2_lit,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float3(param_refinement_float3) => {
                                            let mut float3_lit =
                                                arg.unwrap_literal().unwrap_float3();

                                            if ui
                                                .input_float3(&input_label, &mut float3_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                float3_lit = param_refinement_float3.clamp(float3_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float3(
                                                        float3_lit,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::String(param_refinement_string) => {
                                            let mut imstring_buffer = self.global_imstring_buffer
                                                .borrow_mut();

                                            let string_lit = arg.unwrap_literal().unwrap_string();
                                            imstring_buffer.push_str(string_lit);

                                            if param_refinement_string.file_path {
                                                if file_input(
                                                    ui,
                                                    &input_label,
                                                    param_refinement_string.file_ext_filter,
                                                    &mut imstring_buffer,
                                                ) {
                                                    let string_value = format!("{}", imstring_buffer);
                                                    change = Some((
                                                        stmt_index,
                                                        arg_index,
                                                        ast::Expr::Lit(ast::LitExpr::String(string_value)),
                                                    ));
                                                }
                                            } else if ui
                                                .input_text(&input_label, &mut imstring_buffer)
                                                .read_only(interpreter_busy)
                                                .build() {
                                                    let string_value = format!("{}", imstring_buffer);
                                                    change = Some((
                                                        stmt_index,
                                                        arg_index,
                                                        ast::Expr::Lit(ast::LitExpr::String(string_value)),
                                                    ));
                                                }

                                            imstring_buffer.clear();
                                        }
                                        ParamRefinement::Mesh => {
                                            let changed_expr = self.draw_var_combo_box(
                                                session,
                                                stmt_index,
                                                arg,
                                                Ty::Mesh,
                                                &input_label,
                                            );

                                            if let Some(changed_expr) = changed_expr {
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    changed_expr,
                                                ));
                                            }
                                        }
                                        ParamRefinement::MeshArray => {
                                            let changed_expr = self.draw_var_combo_box(
                                                session,
                                                stmt_index,
                                                arg,
                                                Ty::MeshArray,
                                                &input_label,
                                            );

                                            if let Some(changed_expr) = changed_expr {
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    changed_expr,
                                                ));
                                            }
                                        }
                                    }

                                    if ui.is_item_hovered() && !param_info.description.is_empty() {
                                        ui.tooltip(|| {
                                            let wrap_token = ui
                                                .push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                                            ui.text(param_info.description);
                                            wrap_token.pop(ui);
                                        });
                                    }
                                }

                                let console_id = imgui::im_str!("##console{}", stmt_index);
                                if let Some(window_token) = imgui::ChildWindow::new(&console_id)
                                    .size([0.0, 80.0])
                                    .scrollable(true)
                                    .scroll_bar(true)
                                    .always_vertical_scrollbar(true)
                                    .begin(ui)
                                {
                                    let wrap_token = ui
                                        .push_text_wrap_pos(WRAP_POS_CONSOLE_TEXT_PIXELS);

                                    let log_messages = session.log_messages_at_stmt(stmt_index);
                                    for log_message in log_messages {
                                        ui.text_colored(match log_message.level {
                                            LogMessageLevel::Info => self.colors.log_message_info,
                                            LogMessageLevel::Warn => self.colors.log_message_warn,
                                            LogMessageLevel::Error => self.colors.log_message_error,
                                        }, &log_message.message);
                                    }

                                    let message_count = log_messages.len();
                                    let mut console_state = self.console_state.borrow_mut();
                                    if console_state[stmt_index].message_count < message_count {
                                        ui.set_scroll_here_y();
                                        console_state[stmt_index].message_count = message_count;
                                    }

                                    wrap_token.pop(ui);
                                    window_token.end(ui);
                                }

                                if let Some((color_token, style_token)) = operation_arg_style_tokens {
                                    color_token.pop(ui);
                                    style_token.pop(ui);
                                }

                                ui.unindent();
                            }
                        }
                    }
                }
                regular_font_token.pop(ui);

                let mut pipeline_window_state = self.pipeline_window_state.borrow_mut();
                if pipeline_window_state.autoscroll {
                    ui.set_scroll_here_y();
                    pipeline_window_state.autoscroll = false;
                }
            });
        bold_font_token.pop(ui);

        // FIXME: Debounce changes to parameters

        // Only submit the change if interpreter is not busy. Not all
        // imgui components can be made read-only, so we can not trust
        // it.
        if !interpreter_busy {
            if let Some((stmt_index, arg_index, expr)) = change.clone() {
                let stmt = &session.stmts()[stmt_index];
                match stmt {
                    ast::Stmt::VarDecl(var_decl) => {
                        let init_expr = var_decl.init_expr();
                        let new_var_decl = var_decl
                            .clone_with_init_expr(init_expr.clone_with_arg_at(arg_index, expr));

                        session.set_prog_stmt_at(
                            current_time,
                            stmt_index,
                            ast::Stmt::VarDecl(new_var_decl),
                        );
                    }
                }
            }
        }

        change.is_some()
    }

    pub fn draw_operations_window(
        &self,
        current_time: Instant,
        session: &mut Session,
        duration_autorun_delay: Duration,
    ) -> bool {
        let ui = &self.imgui_ui;
        let function_table = session.function_table();

        const OPERATIONS_WINDOW_WIDTH: f32 = 400.0;
        const OPERATIONS_WINDOW_HEIGHT_MULT: f32 = 0.3;

        let window_logical_size = ui.io().display_size;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let operations_window_height = window_inner_height * OPERATIONS_WINDOW_HEIGHT_MULT - MARGIN;
        let operations_window_vertical_position =
            MARGIN * 2.0 + (1.0 - OPERATIONS_WINDOW_HEIGHT_MULT) * window_inner_height;

        let running_enabled = !session.interpreter_busy() && session.autorun_delay().is_none();
        let popping_enabled = !session.interpreter_busy() && !session.stmts().is_empty();
        let pushing_enabled = !session.interpreter_busy();

        let mut function_clicked = None;
        let mut interpret_clicked = false;
        let mut pop_stmt_clicked = false;

        let mut autorun_enabled = session.autorun_delay().is_some();
        let mut autorun_clicked = false;

        let bold_font_token = ui.push_font(self.font_ids.bold);
        imgui::Window::new(imgui::im_str!("Operations"))
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size(
                [OPERATIONS_WINDOW_WIDTH, operations_window_height],
                imgui::Condition::Always,
            )
            .position(
                [MARGIN, operations_window_vertical_position],
                imgui::Condition::Always,
            )
            .build(ui, || {
                let regular_font_token = ui.push_font(self.font_ids.regular);
                ui.columns(2, imgui::im_str!("Controls columns"), false);

                let pipeline_button_color_token = ui.push_style_colors(&[
                    (imgui::StyleColor::Text, self.colors.special_button_text),
                    (imgui::StyleColor::Button, self.colors.special_button),
                    (
                        imgui::StyleColor::ButtonHovered,
                        self.colors.special_button_hovered,
                    ),
                    (
                        imgui::StyleColor::ButtonActive,
                        self.colors.special_button_active,
                    ),
                ]);
                let running_tokens = if running_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };

                let bold_font_token = ui.push_font(self.font_ids.bold);
                if ui.button(imgui::im_str!("Run"), [-f32::MIN_POSITIVE, 25.0]) && running_enabled {
                    interpret_clicked = true;
                }
                bold_font_token.pop(ui);
                if let Some((color_token, style_token)) = running_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }
                pipeline_button_color_token.pop(ui);

                ui.next_column();

                let popping_tokens = if popping_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                if ui.button(
                    imgui::im_str!("Remove last operation"),
                    [-f32::MIN_POSITIVE, 25.0],
                ) && popping_enabled
                {
                    pop_stmt_clicked = true;
                }
                if let Some((color_token, style_token)) = popping_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }

                ui.columns(1, imgui::im_str!("Autorun columns"), false);
                autorun_clicked =
                    ui.checkbox(imgui::im_str!("Run automatically"), &mut autorun_enabled);

                ui.separator();

                let pushing_tokens = if pushing_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                ui.columns(3, imgui::im_str!("Add operations columns"), false);
                for (func_ident, func) in function_table {
                    if ui.button(
                        &imgui::im_str!("{}", func.info().name),
                        [-f32::MIN_POSITIVE, 20.0],
                    ) && pushing_enabled
                    {
                        function_clicked = Some(func_ident);
                    }

                    if ui.is_item_hovered() && !func.info().description.is_empty() {
                        ui.tooltip(|| {
                            let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                            ui.text(func.info().description);
                            wrap_token.pop(ui);
                        });
                    }

                    ui.next_column();
                }
                if let Some((color_token, style_token)) = pushing_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }
                regular_font_token.pop(ui);
            });
        bold_font_token.pop(ui);

        let function_added = function_clicked.is_some();

        if let Some(func_ident) = function_clicked {
            let func = &function_table[&func_ident];
            let mut args = Vec::with_capacity(func.param_info().len());

            for param_info in func.param_info() {
                let expr = match param_info.refinement {
                    ParamRefinement::Boolean(boolean_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Boolean(boolean_refinement.default_value))
                    }
                    ParamRefinement::Int(int_param_refinement) => ast::Expr::Lit(
                        ast::LitExpr::Int(int_param_refinement.default_value.unwrap_or_default()),
                    ),
                    ParamRefinement::Uint(uint_param_refinement) => ast::Expr::Lit(
                        ast::LitExpr::Uint(uint_param_refinement.default_value.unwrap_or_default()),
                    ),
                    ParamRefinement::Float(float_param_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Float(
                            float_param_refinement.default_value.unwrap_or_default(),
                        ))
                    }
                    ParamRefinement::Float2(float2_param_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Float2([
                            float2_param_refinement.default_value_x.unwrap_or_default(),
                            float2_param_refinement.default_value_y.unwrap_or_default(),
                        ]))
                    }
                    ParamRefinement::Float3(float3_param_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Float3([
                            float3_param_refinement.default_value_x.unwrap_or_default(),
                            float3_param_refinement.default_value_y.unwrap_or_default(),
                            float3_param_refinement.default_value_z.unwrap_or_default(),
                        ]))
                    }
                    ParamRefinement::String(string_param_refinement) => {
                        let initial_value = String::from(string_param_refinement.default_value);
                        ast::Expr::Lit(ast::LitExpr::String(initial_value))
                    }
                    ParamRefinement::Mesh => {
                        let one_past_last_stmt = session.stmts().len();
                        let visible_vars_iter =
                            session.visible_vars_at_stmt(one_past_last_stmt, Ty::Mesh);

                        if visible_vars_iter.clone().count() == 0 {
                            ast::Expr::Lit(ast::LitExpr::Nil)
                        } else {
                            let last = visible_vars_iter
                                .last()
                                .expect("Need at least one variable to provide default value");

                            ast::Expr::Var(ast::VarExpr::new(last))
                        }
                    }
                    ParamRefinement::MeshArray => {
                        let one_past_last_stmt = session.stmts().len();
                        let visible_vars_iter =
                            session.visible_vars_at_stmt(one_past_last_stmt, Ty::MeshArray);

                        if visible_vars_iter.clone().count() == 0 {
                            ast::Expr::Lit(ast::LitExpr::Nil)
                        } else {
                            let last = visible_vars_iter
                                .last()
                                .expect("Need at least one variable to provide default value");

                            ast::Expr::Var(ast::VarExpr::new(last))
                        }
                    }
                };

                args.push(expr);
            }

            let init_expr = ast::CallExpr::new(*func_ident, args);
            let stmt = ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                session.next_free_var_ident(),
                init_expr,
            ));

            session.push_prog_stmt(current_time, stmt);
            self.pipeline_window_state.borrow_mut().autoscroll = true;
        }

        if interpret_clicked {
            session.interpret();
        }

        if pop_stmt_clicked {
            session.pop_prog_stmt(current_time);
        }

        if autorun_clicked {
            if autorun_enabled {
                session.set_autorun_delay(Some(duration_autorun_delay));
            } else {
                session.set_autorun_delay(None);
            }
        }

        function_added || pop_stmt_clicked
    }

    fn draw_var_combo_box(
        &self,
        session: &Session,
        stmt_index: usize,
        arg: &ast::Expr,
        ty: Ty,
        input_label: &imgui::ImStr,
    ) -> Option<ast::Expr> {
        let ui = &self.imgui_ui;

        let mut visible_vars_iter = session.visible_vars_at_stmt(stmt_index, ty);

        let mut selected_var_index = match arg {
            ast::Expr::Lit(ast::LitExpr::Nil) => None,
            ast::Expr::Var(var) => visible_vars_iter
                .clone()
                .position(|var_ident| var_ident == var.ident())
                .map(Some)
                .unwrap_or(None),
            _ => panic!("Arg can either be a variable or nil"),
        };

        // FIXME: Show used var idents differently from unused,
        // e.g. grayed-out

        // FIXME: find a way to make combo boxes read-only
        let mut combo = imgui::ComboBox::new(input_label);

        let mut combo_changed = false;
        let preview_value = selected_var_index
            .map(|index| {
                visible_vars_iter
                    .clone()
                    .nth(index)
                    .expect("Failed to find nth visible var to display preview value")
            })
            .map(|var_ident| {
                format_var_name(
                    session
                        .var_name_for_ident(var_ident)
                        .expect("Failed to find name for ident"),
                    var_ident,
                    ty == Ty::MeshArray,
                )
            })
            .unwrap_or_else(|| imgui::ImString::new("<Select one option>"));

        combo = combo.preview_value(&preview_value);

        let combo_box_color_token = ui.push_style_colors(&[
            (
                imgui::StyleColor::Header,
                self.colors.combo_box_selected_item,
            ),
            (
                imgui::StyleColor::HeaderHovered,
                self.colors.combo_box_selected_item_hovered,
            ),
            (
                imgui::StyleColor::HeaderActive,
                self.colors.combo_box_selected_item_active,
            ),
        ]);
        if let Some(combo_token) = combo.begin(ui) {
            for (index, var_ident) in visible_vars_iter.clone().enumerate() {
                let text = format_var_name(
                    session
                        .var_name_for_ident(var_ident)
                        .expect("Failed to find name for ident"),
                    var_ident,
                    ty == Ty::MeshArray,
                );
                let selected = if let Some(selected_var_index) = selected_var_index {
                    index == selected_var_index
                } else {
                    false
                };

                if imgui::Selectable::new(&text).selected(selected).build(ui) {
                    selected_var_index = Some(index);
                    combo_changed = true;
                }
            }

            if imgui::Selectable::new(imgui::im_str!(""))
                .selected(selected_var_index.is_none())
                .build(ui)
            {
                selected_var_index = None;
                combo_changed = true;
            }

            combo_token.end(ui);
        }
        combo_box_color_token.pop(ui);

        if combo_changed {
            if let Some(selected_var_index) = selected_var_index {
                let var_ident = visible_vars_iter
                    .nth(selected_var_index)
                    .expect("Failed to find nth visible var to create new var expr");
                Some(ast::Expr::Var(ast::VarExpr::new(var_ident)))
            } else {
                Some(ast::Expr::Lit(ast::LitExpr::Nil))
            }
        } else {
            None
        }
    }
}

fn format_var_name(
    name: &str,
    ident: ast::VarIdent,
    surround_with_brackets: bool,
) -> imgui::ImString {
    if surround_with_brackets {
        imgui::im_str!("[{}] #{}", name, ident.0 + 1)
    } else {
        imgui::im_str!("{} #{}", name, ident.0 + 1)
    }
}

fn push_disabled_style(ui: &imgui::Ui) -> (imgui::ColorStackToken, imgui::StyleStackToken) {
    let button_color = ui.style_color(imgui::StyleColor::Button);
    let text_color = ui.style_color(imgui::StyleColor::TextDisabled);

    let color_token = ui.push_style_colors(&[
        (imgui::StyleColor::Text, text_color),
        (imgui::StyleColor::Button, button_color),
        (imgui::StyleColor::ButtonHovered, button_color),
        (imgui::StyleColor::ButtonActive, button_color),
    ]);
    let style_token = ui.push_style_vars(&[imgui::StyleVar::Alpha(0.5)]);

    (color_token, style_token)
}

fn file_input(
    ui: &imgui::Ui,
    label: &imgui::ImStr,
    file_ext_filter: Option<(&[&str], &str)>,
    buffer: &mut imgui::ImString,
) -> bool {
    use std::env;
    use std::path::Path;

    let open_button_label = imgui::im_str!("Open##{}", label);
    let open_button_width = ui.calc_text_size(&open_button_label, true, 50.0)[0] + 8.0;
    let input_position = open_button_width + 2.0; // Padding

    let mut changed = false;

    let group_token = ui.begin_group();

    if ui.button(&open_button_label, [open_button_width, 0.0]) {
        if let Some(absolute_path_string) =
            tinyfiledialogs::open_file_dialog("Open", "", file_ext_filter)
        {
            buffer.clear();

            let current_dir = env::current_dir().expect("Couldn't get current dir");
            let absolute_path = Path::new(&absolute_path_string);

            match absolute_path.strip_prefix(&current_dir) {
                Ok(stripped_path) => {
                    buffer.push_str(&stripped_path.to_string_lossy());
                }
                Err(_) => {
                    buffer.push_str(&absolute_path.to_string_lossy());
                }
            }
        }

        changed = true;
    }

    ui.same_line(input_position);
    ui.set_next_item_width(ui.calc_item_width() - input_position);

    ui.input_text(&label, buffer).read_only(true).build();

    group_token.end(ui);

    changed
}

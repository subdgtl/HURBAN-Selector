use std::cell::RefCell;
use std::f32;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::convert::{cast_u8_color_to_f32, clamp_cast_i32_to_u32, clamp_cast_u32_to_i32};
use crate::imgui_winit_support::{HiDpiMode, WinitPlatform};
use crate::interpreter::{ast, LogMessageLevel, ParamRefinement, Ty};
use crate::notifications::{NotificationLevel, Notifications};
use crate::project;
use crate::session::Session;
use crate::{ScreenshotOptions, Theme, ViewportDrawMode};

const FONT_OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/SpaceMono-Regular.ttf");
const FONT_OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/SpaceMono-Bold.ttf");

const WRAP_POS_TOOLTIP_TEXT_PIXELS: f32 = 400.0;
const WRAP_POS_CONSOLE_TEXT_PIXELS: f32 = 380.0;

const MARGIN: f32 = 10.0;

const OPERATIONS_WINDOW_WIDTH: f32 = 400.0;
const OPERATIONS_WINDOW_HEIGHT_MULT: f32 = 0.33;

const PIPELINE_WINDOW_WIDTH: f32 = OPERATIONS_WINDOW_WIDTH;
const PIPELINE_WINDOW_HEIGHT_MULT: f32 = 1.0 - OPERATIONS_WINDOW_HEIGHT_MULT;
const PIPELINE_OPERATION_CONSOLE_HEIGHT: f32 = 40.0;

const MENU_WINDOW_WIDTH: f32 = 160.0;
const MENU_WINDOW_HEIGHT: f32 = 321.0;

const NOTIFICATIONS_WINDOW_WIDTH: f32 = 600.0;
const NOTIFICATIONS_WINDOW_HEIGHT_MULT: f32 = 0.1;

const SUBDIGITAL_LOGO_WINDOW_WIDTH: f32 = 100.0;

const ABOUT_WINDOW_WIDTH: f32 = 600.0;

const DRAG_SPEED: f32 = 0.01;

struct FontIds {
    regular: imgui::FontId,
    bold: imgui::FontId,
    big_bold: imgui::FontId,
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
    tooltip_text: [f32; 4],
    notification_window: [f32; 4],
    popup_window_background: [f32; 4],
    logo_window: [f32; 4],
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

pub enum OverwriteModalTrigger {
    NewProject,
    OpenProject,
}

#[derive(Default)]
pub struct MenuStatus {
    pub viewport_draw_used_values_changed: bool,
    pub reset_viewport: bool,
    pub export_obj: bool,
    pub new_project: bool,
    pub save_path: Option<PathBuf>,
    pub open_path: Option<PathBuf>,
    pub prevent_overwrite_modal: Option<OverwriteModalTrigger>,
}

pub enum SaveModalResult {
    Save,
    DontSave,
    Cancel,
    Nothing,
}

/// Thin wrapper around imgui and its winit platform. Its main responsibility
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
            log_message_info: [0.3, 0.3, 0.3, 1.0],
            log_message_warn: [0.80, 0.80, 0.05, 1.0],
            log_message_error: [1.0, 0.15, 0.05, 1.0],
            header_error: [0.85, 0.15, 0.05, 0.4],
            header_error_hovered: [1.00, 0.15, 0.05, 0.4],
            tooltip_text: [1.0, 1.0, 1.0, 1.0],
            notification_window: [0.0, 0.0, 0.0, 0.1],
            popup_window_background: [0.0, 0.0, 0.0, 0.4],
            logo_window: [0.0, 0.0, 0.0, 0.0],
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

        if theme == Theme::Light {
            style.window_rounding = 0.0;
            style.frame_rounding = 0.0;
            style.scrollbar_rounding = 0.0;
            style.grab_rounding = 0.0;

            let black = [0.0, 0.0, 0.0, 1.0];
            let white = [1.0, 1.0, 1.0, 1.0];
            let white_80_transparent = [1.0, 1.0, 1.0, 0.8];
            let light = cast_u8_color_to_f32([0xea, 0xe7, 0xe1, 0xff]);
            let light_transparent = cast_u8_color_to_f32([0xea, 0xe7, 0xe1, 0x40]);
            let orange = cast_u8_color_to_f32([0xf2, 0x80, 0x37, 0xff]);
            let orange_light = cast_u8_color_to_f32([0xf2, 0xac, 0x79, 0xff]);
            let orange_light_transparent = cast_u8_color_to_f32([0xf2, 0xac, 0x79, 0x40]);
            let orange_dark = cast_u8_color_to_f32([0xd0, 0x5d, 0x20, 0xff]);
            let orange_dark_transparent = cast_u8_color_to_f32([0xd0, 0x5d, 0x20, 0x40]);
            let green_light = [0.4, 0.8, 0.5, 1.0];
            let green_dark = [0.1, 0.5, 0.2, 1.0];
            let green_dark_transparent = [0.1, 0.5, 0.2, 0.4];
            let red = [1.0, 0.0, 0.0, 1.0];
            let red_transparent = [1.0, 0.0, 0.0, 0.4];
            let transparent = [0.0, 0.0, 0.0, 0.0];

            style[imgui::StyleColor::Text] = orange_dark;
            style[imgui::StyleColor::TextDisabled] = orange_light;
            style[imgui::StyleColor::WindowBg] = white_80_transparent;
            style[imgui::StyleColor::PopupBg] = orange;
            style[imgui::StyleColor::Border] = transparent;
            style[imgui::StyleColor::FrameBg] = light_transparent;
            style[imgui::StyleColor::FrameBgHovered] = orange_light_transparent;
            style[imgui::StyleColor::FrameBgActive] = orange_light_transparent;
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
            style[imgui::StyleColor::Separator] = orange;
            style[imgui::StyleColor::SeparatorHovered] = orange;
            style[imgui::StyleColor::SeparatorActive] = orange;
            style[imgui::StyleColor::ResizeGrip] = orange;
            style[imgui::StyleColor::ResizeGripHovered] = orange_light;
            style[imgui::StyleColor::ResizeGripActive] = orange_light;
            style[imgui::StyleColor::Tab] = light_transparent;
            style[imgui::StyleColor::TabHovered] = orange_light_transparent;
            style[imgui::StyleColor::TabActive] = light_transparent;
            style[imgui::StyleColor::TabUnfocused] = light_transparent;
            style[imgui::StyleColor::TabUnfocusedActive] = light_transparent;
            style[imgui::StyleColor::PlotLines] = orange;
            style[imgui::StyleColor::TextSelectedBg] = orange_light_transparent;
            style[imgui::StyleColor::NavHighlight] = light_transparent;

            colors.special_button_text = white;
            colors.special_button = green_light;
            colors.special_button_hovered = green_dark;
            colors.special_button_active = green_dark_transparent;

            colors.combo_box_selected_item = light;
            colors.combo_box_selected_item_hovered = orange_light;
            colors.combo_box_selected_item_active = orange_dark;

            colors.tooltip_text = white;

            colors.log_message_warn = black;
            colors.log_message_error = red;

            colors.header_error = red_transparent;
            colors.header_error_hovered = red;

            colors.notification_window = white_80_transparent;

            colors.popup_window_background = white_80_transparent;
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
        let big_bold_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: FONT_OPENSANS_BOLD_BYTES,
                size_pixels: font_size * 1.5,
                config: None,
            }]);

        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        Ui {
            imgui_context,
            imgui_winit_platform: platform,
            font_ids: FontIds {
                regular: regular_font_id,
                bold: bold_font_id,
                big_bold: big_bold_font_id,
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

    pub fn process_event<T>(
        &mut self,
        event: &winit::event::Event<T>,
        window: &winit::window::Window,
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

    pub fn want_capture_keyboard(&self) -> bool {
        self.imgui_context.io().want_capture_keyboard
    }

    pub fn want_capture_mouse(&self) -> bool {
        self.imgui_context.io().want_capture_mouse
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

        let window_color_token = ui.push_style_color(
            imgui::StyleColor::PopupBg,
            self.colors.popup_window_background,
        );

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
            .movable(true)
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
                    .input_int2(imgui::im_str!("Dimensions (px)"), &mut dimensions)
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
                    imgui::im_str!("Transparent Background"),
                    &mut screenshot_options.transparent,
                );

                if ui.button(imgui::im_str!("Take Screenshot"), [0.0, 0.0]) {
                    take_screenshot_clicked = true;
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(
                            self.colors.log_message_warn,
                            "WARNING: Attempting to take screenshots may crash the program.\n\
                             Be sure to save your work.",
                        );
                        wrap_token.pop(ui);
                    });
                }

                regular_font_token.pop(ui);
            });
        bold_font_token.pop(ui);
        window_color_token.pop(ui);

        if take_screenshot_clicked {
            *screenshot_modal_open = false;
        }

        take_screenshot_clicked
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_about_window(
        &self,
        about_modal_open: &mut bool,
        tex_scheme: imgui::TextureId,
        width_scheme: u32,
        height_scheme: u32,
        tex_logos: imgui::TextureId,
        width_logos: u32,
        height_logos: u32,
    ) {
        let ui = &self.imgui_ui;

        let window_color_token = ui.push_style_color(
            imgui::StyleColor::PopupBg,
            self.colors.popup_window_background,
        );

        let window_name = imgui::im_str!("About");
        if *about_modal_open {
            ui.open_popup(window_name);
        }

        let bold_font_token = ui.push_font(self.font_ids.bold);
        ui.popup_modal(window_name)
            .opened(about_modal_open)
            .movable(true)
            .resizable(false)
            .collapsible(false)
            .always_auto_resize(false)
            .build(|| {
                let window_width = ui.window_size()[0];

                let wrap_token = ui.push_text_wrap_pos(ABOUT_WINDOW_WIDTH);

                let big_bold_font_token = ui.push_font(self.font_ids.big_bold);
                ui.text(imgui::im_str!("H.U.R.B.A.N. selector 0.1 alpha"));
                big_bold_font_token.pop(ui);
                ui.text(imgui::im_str!("by Subdigital s.r.o. (https://sub.digital), 2020"));
                ui.new_line();
                ui.text(imgui::im_str!("CREDITS"));
                let mut regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text("Lead developer: Jan Toth <yanchi.toth@gmail.com>");
                ui.text("Geometry developer: Jan Pernecky <jan@sub.digital>");
                ui.text("Developer: Ondrej Slintak <ondrowan@gmail.com>");
                ui.text("Concept: Maros Schmidt, Slovak Design Center");
                ui.text("Production: Lucia Dubacova, Slovak Design Center");
                ui.new_line();
                regular_font_token.pop(ui);

                imgui::Image::new(
                    tex_scheme,
                    [window_width * 0.95, window_width * 0.95 / width_scheme as f32 * height_scheme as f32],
                ).build(ui);
                ui.new_line();

                ui.text(imgui::im_str!("ABOUT"));
                regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text_wrapped(imgui::im_str!(
                    "The H.U.R.B.A.N. selector is a part of the SDC's Inolab \
                    Department's plan to create a research platform for designers to \
                    test and verify new algorithms and create new forms.\n\
                    \n\
                    H.U.R.B.A.N. selector is an experimental software for \
                    hybridization of multiple 3D models with an aim to find new \
                    aesthetic forms. \
                    It serves as a gateway to full-fledged parametric design software. \
                    A user builds an operation pipeline where each stacked operation allows \
                    reconfiguration anytime influencing inputs and outputs of subsequent \
                    operations. The program extends the creative possibilities of designers and helps \
                    them create beyond the limits of their imagination given by memory/brain \
                    capacity as well as the ability to create different variations of form and \
                    compositions.\n\
                    \n\
                    The software is currently in very early development stages. It strives to be \
                    a tool for simple parametric modeling using various \
                    hybridization strategies for mesh and voxel models, allowing designers to \
                    smoothly interpolate between multiple source models."));
                ui.new_line();
                regular_font_token.pop(ui);

                ui.text(imgui::im_str!("PARTNERS"));
                regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text_wrapped(imgui::im_str!(
                    "The Software is produced within the INTERREG V-A Slovakia - \
                    Austria 2014 - 2020 'Design & Innovation' project"
                ));
                regular_font_token.pop(ui);
                ui.new_line();
                imgui::Image::new(
                    tex_logos,
                    [window_width * 0.95, window_width * 0.95 / width_logos as f32 * height_logos as f32],
                ).build(ui);
                ui.new_line();

                ui.text(imgui::im_str!("SOURCE CODE LICENSE"));
                regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text_wrapped(imgui::im_str!(
                    "The editor source code is provided under the GNU GENERAL PUBLIC \
                    LICENSE, Version 3. If the research or implementation yields \
                    interesting results, those will be extracted from the editor and \
                    published and licensed separately, most likely under a more permissive \
                    license such as MIT.\n\
                    \n\
                    The source code of H.U.R.B.A.N. selector written in Rust can be found on \
                    GitHub (https://github.com/subdgtl/HURBAN-selector)."));
                ui.new_line();
                regular_font_token.pop(ui);
                ui.new_line();
                ui.separator();
                ui.new_line();
                ui.text(imgui::im_str!("END-USER LICENSE AGREEMENT of H.U.R.B.A.N. selector"));
                ui.new_line();
                regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text_wrapped(imgui::im_str!("\
                    This End-User License Agreement ('EULA') is a legal agreement between you \
                    and Slovak Design Center\n\
                    \n\
                    This EULA agreement governs your acquisition and use of our H.U.R.B.A.N. \
                    selector software ('Software') directly from Slovak Design Center \
                    or indirectly through a Slovak Design Center \
                    authorized reseller or distributor (a 'Reseller').\n\
                    \n\
                    Please read this EULA agreement carefully before completing the installation \
                    process and using the H.U.R.B.A.N. selector software. It provides a license to \
                    use the H.U.R.B.A.N. selector software and contains warranty information and \
                    liability disclaimers.\n\
                    \n\
                    If you register for a free trial of the H.U.R.B.A.N. selector software, this EULA \
                    agreement will also govern that trial. By clicking 'accept' or installing and/or \
                    using the H.U.R.B.A.N. selector software, you are confirming your acceptance of \
                    the Software and agreeing to become bound by the terms of this EULA agreement.\n\
                    \n\
                    If you are entering into this EULA agreement on behalf of a company or other legal \
                    entity, you represent that you have the authority to bind such entity and its \
                    affiliates to these terms and conditions. If you do not have such authority or if \
                    you do not agree with the terms and conditions of this EULA agreement, do not \
                    install or use the Software, and you must not accept this EULA agreement.\n\
                    \n\
                    This EULA agreement shall apply only to the Software supplied by Slovak Design \
                    Center herewith regardless of whether other software is \
                    referred to or described herein. The terms also apply to any Slovak Design Center \
                    updates, supplements, Internet-based services, and support \
                    services for the Software, unless other terms accompany those items on delivery. \
                    If so, those terms apply. This EULA was created by EULA Template for H.U.R.B.A.N. \
                    selector."));
                ui.new_line();
                ui.text(imgui::im_str!("LICENSE AGREEMENT"));
                ui.new_line();
                ui.text_wrapped(imgui::im_str!("\
                    Slovak Design Center hereby grants you a personal, \
                    non-transferable, non-exclusive licence to use the H.U.R.B.A.N. selector software \
                    on your devices in accordance with the terms of this EULA agreement.\n\
                    \n\
                    You are permitted to load the H.U.R.B.A.N. selector software (for example a PC, \
                    laptop, mobile or tablet) under your control. You are responsible for ensuring \
                    your device meets the minimum requirements of the H.U.R.B.A.N. selector software.\n\
                    \n\
                    You are not permitted to:\n\
                    * Reproduce, copy, distribute, resell or otherwise use the Software for any \
                    commercial purpose\n\
                    * Allow any third party to use the Software on behalf of or for the benefit of \
                    any third party\n\
                    * Use the Software in any way which breaches any applicable local, national or \
                    international law\n\
                    * Use the Software for any purpose that Slovak Design Center \
                    considers is a breach of this EULA agreement"));
                ui.new_line();
                ui.text(imgui::im_str!("INTELLECTUAL PROPERTY AND OWNERSHIP"));
                ui.new_line();
                ui.text_wrapped(imgui::im_str!("\
                    Slovak Design Center and Subdigital s.r.o. shall at all times retain ownership of \
                    the Software as originally downloaded by you and all subsequent downloads of the \
                    Software by you. The Software (and the copyright, and other intellectual property \
                    rights of whatever nature in the Software, including any modifications made thereto) \
                    are and shall remain the property of Slovak Design Center and Subdigital s.r.o..\n\
                    \n\
                    Slovak Design Center reserves the right to grant licences to \
                    use the Software to third parties."));
                ui.new_line();
                ui.text(imgui::im_str!("TERMINATION"));
                ui.new_line();
                ui.text_wrapped(imgui::im_str!("\
                    This EULA agreement is effective from the date you first use the Software and shall \
                    continue until terminated. You may terminate it at any time upon written notice to \
                    Slovak Design Center.\n\
                    \n\
                    It will also terminate immediately if you fail to comply with any term of this EULA \
                    agreement. Upon such termination, the licenses granted by this EULA agreement will \
                    immediately terminate and you agree to stop all access and use of the Software. \
                    The provisions that by their nature continue and survive will survive any termination \
                    of this EULA agreement."));
                ui.new_line();
                ui.text(imgui::im_str!("GOVERNING LAW"));
                ui.new_line();
                ui.text_wrapped(imgui::im_str!("\
                    This EULA agreement, and any dispute arising out of or in connection with this EULA \
                    agreement, shall be governed by and construed in accordance with the laws of \
                    Slovak republic."));
                regular_font_token.pop(ui);
                wrap_token.pop(ui);
            });

        bold_font_token.pop(ui);
        window_color_token.pop(ui);
    }

    pub fn draw_notifications_window(&self, notifications: &Notifications) {
        let notifications_count = notifications.iter().count();
        if notifications_count == 0 {
            self.notifications_state.borrow_mut().notifications_count = 0;
            return;
        }

        let ui = &self.imgui_ui;

        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let notifications_window_height =
            window_inner_height * NOTIFICATIONS_WINDOW_HEIGHT_MULT - MARGIN;
        let notifications_window_vertical_position =
            MARGIN * 2.0 + (1.0 - NOTIFICATIONS_WINDOW_HEIGHT_MULT) * window_inner_height;

        let color_token =
            ui.push_style_color(imgui::StyleColor::WindowBg, self.colors.notification_window);

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
                        NotificationLevel::Error => ui.push_style_color(
                            imgui::StyleColor::Text,
                            self.colors.log_message_error,
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

    pub fn draw_subdigital_logo(
        &self,
        tex_subdigital_logo: imgui::TextureId,
        width_subdigital_logo: u32,
        height_subdigital_logo: u32,
    ) {
        let ui = &self.imgui_ui;

        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let subdigital_logo_window_height_mult =
            height_subdigital_logo as f32 / width_subdigital_logo as f32;

        let subdigital_logo_window_height =
            SUBDIGITAL_LOGO_WINDOW_WIDTH * subdigital_logo_window_height_mult;

        let subdigital_logo_window_horizontal_position =
            window_inner_width - SUBDIGITAL_LOGO_WINDOW_WIDTH;
        let subdigital_logo_window_vertical_position =
            window_inner_height - subdigital_logo_window_height * 0.8;

        let color_token = ui.push_style_colors(&[
            (imgui::StyleColor::WindowBg, self.colors.logo_window),
            (imgui::StyleColor::Border, self.colors.logo_window),
        ]);

        imgui::Window::new(imgui::im_str!("Logo"))
            .title_bar(false)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size(
                [
                    SUBDIGITAL_LOGO_WINDOW_WIDTH + MARGIN,
                    subdigital_logo_window_height,
                ],
                imgui::Condition::Always,
            )
            .position(
                [
                    subdigital_logo_window_horizontal_position,
                    subdigital_logo_window_vertical_position,
                ],
                imgui::Condition::Always,
            )
            .build(ui, || {
                imgui::Image::new(
                    tex_subdigital_logo,
                    [SUBDIGITAL_LOGO_WINDOW_WIDTH, subdigital_logo_window_height],
                )
                .build(ui);
            });

        color_token.pop(ui);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_menu_window(
        &self,
        current_time: Instant,
        screenshot_modal_open: &mut bool,
        about_modal_open: &mut bool,
        viewport_draw_mode: &mut ViewportDrawMode,
        viewport_draw_used_values: &mut bool,
        project_status: &mut project::ProjectStatus,
        session: &mut Session,
        notifications: &mut Notifications,
    ) -> MenuStatus {
        let ui = &self.imgui_ui;
        let mut status = MenuStatus::default();

        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;

        let bold_font_token = ui.push_font(self.font_ids.bold);
        #[allow(clippy::cognitive_complexity)]
        imgui::Window::new(imgui::im_str!("Menu"))
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size(
                [MENU_WINDOW_WIDTH, MENU_WINDOW_HEIGHT],
                imgui::Condition::Always,
            )
            .position(
                [window_inner_width + MARGIN - MENU_WINDOW_WIDTH, MARGIN],
                imgui::Condition::Always,
            )
            .build(ui, || {
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        let regular_font_token = ui.push_font(self.font_ids.regular);
                        ui.text_colored(self.colors.tooltip_text, "MAIN MENU\n\
                        \n\
                        Viewport information and settings.\n\
                        Screenshot and file management.");
                        regular_font_token.pop(ui);
                        wrap_token.pop(ui);
                    });
                }
                let regular_font_token = ui.push_font(self.font_ids.regular);
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "FRAMES PER SECOND\n\
                        \n\
                        Shows the rendering performance of the current model on the current computer. \
                        The desired value for standard computers is 60 FPS \
                        some monitors may limit this value to 30 FPS \
                        and some gaming machines may support higher rates.\n\
                        \n\
                        If the FPS suddenly drops, it is na indication that the current model \
                        is already too heavy (contains too many vertices and faces) \
                        and for sake of performance and safety, it should be reduced.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.radio_button(
                    imgui::im_str!("Shaded"),
                    viewport_draw_mode,
                    ViewportDrawMode::Shaded,
                ) {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        "Viewport mode changed to Shaded.",
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SHADED VIEWPORT MODE\n\
                        \n\
                        The geometry will be shaded with solid color and no edges will be highlighted.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.radio_button(
                    imgui::im_str!("Wireframes"),
                    viewport_draw_mode,
                    ViewportDrawMode::Wireframe,
                ) {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        "Viewport mode changed to Wireframes.",
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "WIREFRAME VIEWPORT MODE\n\
                        \n\
                        The geometry will be rendered as a wireframe model, the surface will be \
                        fully transparent and only edges will be highlighted.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.radio_button(
                    imgui::im_str!("Shaded with Edges"),
                    viewport_draw_mode,
                    ViewportDrawMode::ShadedWireframe,
                ) {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        "Viewport mode changed to Shaded with Edges (Wireframes).",
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SHADED WITH EDGES VIEWPORT MODE\n\
                        \n\
                        The geometry will be shaded with solid color and visible edges will be highlighted.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.radio_button(
                    imgui::im_str!("X-RAY"),
                    viewport_draw_mode,
                    ViewportDrawMode::ShadedWireframeXray,
                ) {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        "Viewport mode changed to X-Ray: Shaded with internal Edges (Wireframes).",
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SHADED WITH X-RAY EDGES VIEWPORT MODE\n\
                        \n\
                        The geometry will be shaded with solid color and all edges, \
                        including the ones hidden behind the solid color of the surfaces, \
                        will be highlighted.");
                        wrap_token.pop(ui);
                    });
                }

                status.viewport_draw_used_values_changed = ui.checkbox(
                    imgui::im_str!("Draw used geometry"),
                    viewport_draw_used_values,
                );
                if status.viewport_draw_used_values_changed {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        if *viewport_draw_used_values {
                            "Viewport now draws used geometry."
                        } else {
                            "Viewport now doesn't draw used geometry."
                        }
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text,"DRAW USED GEOMETRY IN VIEWPORT\n\
                        \n\
                        When enabled, used geometry will be drawn with a transparent material.\n\
                        \n\
                        Used geometry is geometry that has been already referenced as a parameter \
                        in an operation.");
                        wrap_token.pop(ui);
                    });
                }

                status.reset_viewport =
                    ui.button(imgui::im_str!("Reset viewport"), [-f32::MIN_POSITIVE, 0.0]);
                if status.reset_viewport {
                    notifications.push(
                        current_time,
                        NotificationLevel::Info,
                        "Viewport camera reset to fit all visible geometry.",
                    );
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "RESET VIEWPORT CAMERA\n\
                        \n\
                        Set the viewport camera to look at all visible geometry in the scene.");
                        wrap_token.pop(ui);
                    });
                }

                ui.separator();

                if ui.button(imgui::im_str!("New"), [-f32::MIN_POSITIVE, 0.0])
                    || project_status.new_requested
                {
                    if project_status.changed_since_last_save
                        && project_status.prevent_overwrite_status.is_none()
                    {
                        status.prevent_overwrite_modal = Some(OverwriteModalTrigger::NewProject);
                    } else {
                        status.new_project = true;
                    }

                    project_status.prevent_overwrite_status = None;
                    project_status.new_requested = false;
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "RESET THE PIPELINE\n\
                        \n\
                        Closes the current project and starts a new one. \n\
                        \n\
                        The new project is not saved by default and has to be saved manually.");
                        wrap_token.pop(ui);
                    });
                }

                                if ui.button(imgui::im_str!("Open"), [-f32::MIN_POSITIVE, 0.0])
                    || project_status.open_requested
                {
                    // FIXME: @Refactoring Factor out this use of
                    // tinyfiledialogs from this module
                    if project_status.changed_since_last_save
                        && project_status.prevent_overwrite_status.is_none()
                    {
                        status.prevent_overwrite_modal = Some(OverwriteModalTrigger::OpenProject);
                    } else if let Some(path) = tinyfiledialogs::open_file_dialog(
                        "Open",
                        "",
                        Some((project::EXTENSION_FILTER, project::EXTENSION_DESCRIPTION)),
                    ) {
                        status.open_path = Some(PathBuf::from(path));
                    }

                    project_status.prevent_overwrite_status = None;
                    project_status.open_requested = false;
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "OPEN PROJECT FROM A .hurban FILE\n\
                        \n\
                        Opens the sequence of operations saved in a .hurban file.\n\
                        \n\
                        The .hurban project file contains only the operation pipeline. It does not contain any \
                        actual geometry, but rather just the sequence of operations that generates the geometry \
                        and references to the external files to import. \
                        It is advised to keep the files to import next to the .hurban project file \
                        and distribute them together.");
                        wrap_token.pop(ui);
                    });
                }

                ui.separator();

                if ui.button(imgui::im_str!("Save"), [-f32::MIN_POSITIVE, 0.0]) {
                    match &project_status.path {
                        Some(project_path) => {
                            status.save_path = Some(project_path.clone())
                        }
                        None => {
                            // FIXME: @Refactoring Factor out this use of
                            // tinyfiledialogs from this module
                            if let Some(path) = tinyfiledialogs::save_file_dialog_with_filter(
                                "Save",
                                project::DEFAULT_NEW_FILENAME,
                                project::EXTENSION_FILTER,
                                project::EXTENSION_DESCRIPTION,
                            ) {
                                status.save_path = Some(PathBuf::from(path));
                            }
                        }
                    }
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SAVE PROJECT INTO THE CURRENT A .hurban FILE\n\
                        \n\
                        Saves the current project into a .hurban file. \
                        When used for the first time, opens a system dialog to specify save file location.\n\
                        \n\
                        The .hurban project file contains only the operation pipeline. It does not contain any \
                        actual geometry, but rather just the sequence of operations that generates the geometry \
                        and references to the external files to import. \
                        It is advised to keep the files to import next to the .hurban project file \
                        and distribute them together.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.button(imgui::im_str!("Save as..."), [-f32::MIN_POSITIVE, 0.0]) {
                    // FIXME: @Refactoring Factor out this use of
                    // tinyfiledialogs from this module
                    if let Some(path) = tinyfiledialogs::save_file_dialog_with_filter(
                        "Save",
                        project::DEFAULT_NEW_FILENAME,
                        project::EXTENSION_FILTER,
                        project::EXTENSION_DESCRIPTION,
                    ) {
                        status.save_path = Some(PathBuf::from(path));
                    }
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SAVE PROJECT INTO A NEW .hurban FILE\n\
                        \n\
                        Saves the current project into a .hurban file. \
                        Opens a system dialog to specify save file location.\n\
                        \n\
                        The .hurban project file contains only the operation pipeline. It does not contain any \
                        actual geometry, but rather just the sequence of operations that generates the geometry \
                        and references to the external files to import. \
                        It is advised to keep the files to import next to the .hurban project file \
                        and distribute them together.");
                        wrap_token.pop(ui);
                    });
                }

                if ui.button(imgui::im_str!("Save screenshot..."), [-f32::MIN_POSITIVE, 0.0]) {
                    *screenshot_modal_open = true;
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "SAVE SCREENSHOT\n\
                        \n\
                        Opens a system dialog for saving the current viewport into a PNG file.");
                        wrap_token.pop(ui);
                    });
                }

                let export_obj_disabled_unsynced = !session.synced();
                let export_obj_disabled_empty = session.stmts().is_empty();
                let export_obj_disabled = export_obj_disabled_unsynced || export_obj_disabled_empty;
                let export_obj_button_tokens = if export_obj_disabled  {
                    Some(push_disabled_style(ui))
                } else {
                    None
                };
                let export_obj = ui.button(
                    imgui::im_str!("Export OBJ..."),
                    [-f32::MIN_POSITIVE, 0.0],
                );
                if let Some((color_token, style_token)) = export_obj_button_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "EXPORT OBJ\n\
                        \n\
                        Opens a system dialog for exporting all unused geometry into an OBJ file.");
                        if export_obj_disabled_unsynced {
                            ui.text_colored(
                                self.colors.log_message_warn,
                                "WARNING: All operations must be executed before exporting.",
                            );
                        }
                        if export_obj_disabled_empty {
                            ui.text_colored(
                                self.colors.log_message_warn,
                                "WARNING: Can not export empty scene.\n\
                                 Try adding operations to the pipeline and executing first.",
                            );
                        }
                        wrap_token.pop(ui);
                    });
                }

                status.export_obj = !export_obj_disabled && export_obj;

                ui.separator();

                if ui.button(imgui::im_str!("About"), [-f32::MIN_POSITIVE, 0.0]) {
                    *about_modal_open = true;
                }
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "ABOUT H.U.R.B.A.N. selector\n\
                        \n\
                        Program description and credits.");
                        wrap_token.pop(ui);
                    });
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
        let window_color_token = ui.push_style_color(
            imgui::StyleColor::PopupBg,
            self.colors.popup_window_background,
        );
        ui.open_popup(imgui::im_str!("Unsaved changes"));
        ui.popup_modal(imgui::im_str!("Unsaved changes"))
            .resizable(false)
            .build(|| {
                ui.text("To preserve unsaved changes in the pipeline please save the project.");

                let width_unit = ui.window_size()[0] / 11.0;

                if ui.button(imgui::im_str!("Save"), [width_unit * 3.0, 0.0]) {
                    save_modal_result = SaveModalResult::Save;

                    ui.close_current_popup();
                }

                ui.same_line(width_unit * 4.0);

                if ui.button(imgui::im_str!("Discard changes"), [width_unit * 3.0, 0.0]) {
                    save_modal_result = SaveModalResult::DontSave;

                    ui.close_current_popup();
                }

                ui.same_line(width_unit * 8.0);

                if ui.button(imgui::im_str!("Cancel"), [width_unit * 3.0, 0.0]) {
                    save_modal_result = SaveModalResult::Cancel;

                    ui.close_current_popup();
                }
            });

        window_color_token.pop(ui);

        save_modal_result
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

        let window_logical_size = ui.io().display_size;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let pipeline_window_height = window_inner_height * PIPELINE_WINDOW_HEIGHT_MULT;

        let interpreter_busy = session.interpreter_busy();
        let mut change = None;

        let bold_font_token = ui.push_font(self.font_ids.bold);
        imgui::Window::new(imgui::im_str!("Operation pipeline"))
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .size([PIPELINE_WINDOW_WIDTH, pipeline_window_height], imgui::Condition::Always)
            .position([MARGIN, MARGIN], imgui::Condition::Always)
            .build(ui, || {
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        let regular_font_token = ui.push_font(self.font_ids.regular);
                        ui.text_colored(self.colors.tooltip_text,
                            "OPERATION PIPELINE\n\
                            \n\
                            An ordered sequence of operations that generate the viewport geometry. \
                            When the pipeline is run, the operations are being executed one after \
                            another from top down. Each operation can be customized by setting the \
                            input parameter values or specifying the input data (mesh geometry, \
                            mesh group, path to file).\n\
                            \n\
                            Each operation in the pipeline generates data: either a mesh geometry or \
                            a mesh group which can be later used in a subsequent operation. Only unused \
                            (freshly generated) geometry (mesh or group) is rendered in the viewport \
                            by default, \
                            however even the geometry, which has been already used, can be reused in \
                            subsequent operations. Operations can take as an input only that geometry, \
                            which has been generated in the preceding operations in the pipeline.\n\
                            \n\
                            It is possible to change any input parameters of any operation at any time, \
                            not only after the operation has been added to the pipeline. \
                            This is useful when some parameters need to be adjusted only after the results \
                            of subsequent operations can be visually evaluated. This approach is a gateway \
                            to full-fledged parametric modeling paradigm.\n\
                            \n\
                            The .hurban project files can be build as a tool for achieving certain manipulations \
                            with mesh geometry and because they don't contain the imported mesh models, \
                            one sequence of operations (one project file) can be reused for various input mesh \
                            models. Hence, H.U.R.B.A.N. selector is not only a geometry transformation tool, \
                            but also a tool-building platform and the project files ara not only geometries but \
                            also tools.",
                        );
                        regular_font_token.pop(ui);
                        wrap_token.pop(ui);
                    });
                }
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
                                    (imgui::StyleColor::Text, self.colors.tooltip_text),
                                ]))
                            } else {
                                None
                            };

                            let collapsing_header_open = imgui::CollapsingHeader::new(&imgui::im_str!(
                                    "#{} {} ##{}",
                                    stmt_index + 1,
                                    func.info().name,
                                    stmt_index
                                ))
                                .default_open(true)
                                .build(ui);

                            if ui.is_item_hovered() {
                                if let Some(error) = error {
                                    let color_token = ui.push_style_color(
                                        imgui::StyleColor::PopupBg,
                                        self.colors.header_error_hovered,
                                    );

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
                                            self.colors.tooltip_text,
                                            &*imstring_buffer,
                                        );

                                        imstring_buffer.clear();

                                        wrap_token.pop(ui);
                                    });
                                    color_token.pop(ui);
                                } else if !func.info().description.is_empty() {
                                    ui.tooltip(|| {
                                        let wrap_token = ui
                                            .push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                                            ui.text_colored(self.colors.tooltip_text, func.info().description);
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

                                            let mut drag_int = imgui::Drag::<i32>::new(&input_label)
                                                .speed(DRAG_SPEED);

                                            match (
                                                param_refinement_int.min_value,
                                                param_refinement_int.max_value,
                                            ) {
                                                (Some(min_value), Some(max_value)) => {
                                                    drag_int = drag_int.range(min_value..=max_value);
                                                }
                                                (Some(min_value), None) => {
                                                    drag_int = drag_int.range(min_value..);
                                                }
                                                (None, Some(max_value)) => {
                                                    drag_int = drag_int.range(..=max_value);
                                                }
                                                (None, None) => (),
                                            }

                                            if drag_int.build(ui, &mut int_lit) {
                                                int_lit = param_refinement_int.clamp(int_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Int(int_lit)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Uint(param_refinement_uint) => {
                                            let mut uint_lit = arg.unwrap_literal().unwrap_uint();

                                            let mut drag_uint = imgui::Drag::<u32>::new(&input_label)
                                                .speed(DRAG_SPEED);

                                            match (
                                                param_refinement_uint.min_value,
                                                param_refinement_uint.max_value,
                                            ) {
                                                (Some(min_value), Some(max_value)) => {
                                                    drag_uint = drag_uint.range(min_value..=max_value);
                                                }
                                                (Some(min_value), None) => {
                                                    drag_uint = drag_uint.range(min_value..);
                                                }
                                                (None, Some(max_value)) => {
                                                    drag_uint = drag_uint.range(..=max_value);
                                                }
                                                (None, None) => (),
                                            }

                                            if drag_uint.build(ui, &mut uint_lit) {
                                                let uint_value = param_refinement_uint.clamp(uint_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Uint(uint_value)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float(param_refinement_float) => {
                                            let mut float_lit = arg.unwrap_literal().unwrap_float();

                                            let mut drag_float = imgui::Drag::<f32>::new(&input_label)
                                                .speed(DRAG_SPEED);

                                            match (
                                                param_refinement_float.min_value,
                                                param_refinement_float.max_value,
                                            ) {
                                                (Some(min_value), Some(max_value)) => {
                                                    drag_float = drag_float.range(min_value..=max_value);
                                                }
                                                (Some(min_value), None) => {
                                                    drag_float = drag_float.range(min_value..);
                                                }
                                                (None, Some(max_value)) => {
                                                    drag_float = drag_float.range(..=max_value);
                                                }
                                                (None, None) => (),
                                            }

                                            if drag_float.build(ui, &mut float_lit)
                                            {
                                                let float_value = param_refinement_float.clamp(float_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float(float_value)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float2(param_refinement_float2) => {
                                            let mut float2_lit =
                                                arg.unwrap_literal().unwrap_float2();

                                            let mut drag_float2 = imgui::Drag::<f32>::new(&input_label)
                                                .speed(DRAG_SPEED);

                                            match (
                                                param_refinement_float2.min_value,
                                                param_refinement_float2.max_value,
                                            ) {
                                                (Some(min_value), Some(max_value)) => {
                                                    drag_float2 = drag_float2.range(min_value..=max_value);
                                                }
                                                (Some(min_value), None) => {
                                                    drag_float2 = drag_float2.range(min_value..);
                                                }
                                                (None, Some(max_value)) => {
                                                    drag_float2 = drag_float2.range(..=max_value);
                                                }
                                                (None, None) => (),
                                            }

                                            if drag_float2.build_array(ui, &mut float2_lit)
                                            {
                                                let float2_value = param_refinement_float2.clamp(float2_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float2(
                                                        float2_value,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float3(param_refinement_float3) => {
                                            let mut float3_lit =
                                                arg.unwrap_literal().unwrap_float3();

                                            let mut drag_float3 = imgui::Drag::<f32>::new(&input_label)
                                                .speed(DRAG_SPEED);

                                            match (
                                                param_refinement_float3.min_value,
                                                param_refinement_float3.max_value,
                                            ) {
                                                (Some(min_value), Some(max_value)) => {
                                                    drag_float3 = drag_float3.range(min_value..=max_value);
                                                }
                                                (Some(min_value), None) => {
                                                    drag_float3 = drag_float3.range(min_value..);
                                                }
                                                (None, Some(max_value)) => {
                                                    drag_float3 = drag_float3.range(..=max_value);
                                                }
                                                (None, None) => (),
                                            }

                                            if drag_float3.build_array(ui, &mut float3_lit)
                                            {
                                                let float3_value = param_refinement_float3.clamp(float3_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float3(
                                                        float3_value,
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
                                                ui.text_colored(self.colors.tooltip_text, param_info.description);
                                            wrap_token.pop(ui);
                                        });
                                    }
                                }

                                let console_id = imgui::im_str!("##console{}", stmt_index);
                                if let Some(window_token) = imgui::ChildWindow::new(&console_id)
                                    .size([0.0, PIPELINE_OPERATION_CONSOLE_HEIGHT])
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

        let changed = change.is_some();

        // FIXME: Debounce changes to parameters

        // Only submit the change if interpreter is not busy. Not all
        // imgui components can be made read-only, so we can not trust
        // it.
        if !interpreter_busy {
            if let Some((stmt_index, arg_index, expr)) = change {
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

        changed
    }

    pub fn draw_operations_window(
        &self,
        current_time: Instant,
        session: &mut Session,
        notifications: &mut Notifications,
        duration_autorun_delay: Duration,
    ) -> bool {
        let ui = &self.imgui_ui;
        let function_table = session.function_table();

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
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        let regular_font_token = ui.push_font(self.font_ids.regular);
                        ui.text_colored(self.colors.tooltip_text, "AVAILABLE OPERATIONS\n\
                        \n\
                        A list of available operations to be stacked into the sequence of operations \
                        in the Operation pipeline.");
                        regular_font_token.pop(ui);
                        wrap_token.pop(ui);
                    });
                }

                let regular_font_token = ui.push_font(self.font_ids.regular);
                ui.columns(2, imgui::im_str!("Controls columns"), false);

                let pipeline_button_color_token = ui.push_style_colors(&[
                    (imgui::StyleColor::Button, self.colors.special_button),
                    (
                        imgui::StyleColor::ButtonHovered,
                        self.colors.special_button_hovered,
                    ),
                    (
                        imgui::StyleColor::ButtonActive,
                        self.colors.special_button_active,
                    ),
                    (imgui::StyleColor::Text, self.colors.special_button_text),
                    (imgui::StyleColor::TextDisabled, self.colors.special_button_text),
                ]);
                let running_tokens = if running_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };

                let bold_font_token = ui.push_font(self.font_ids.bold);
                if ui.button(
                    imgui::im_str!("Run (Enter)"),
                    [-f32::MIN_POSITIVE, 25.0],
                ) && running_enabled
                {
                    interpret_clicked = true;
                }
                bold_font_token.pop(ui);
                if let Some((color_token, style_token)) = running_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }
                pipeline_button_color_token.pop(ui);

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "RUN RECOMPUTATION OF THE OPERATION PIPELINE\n\
                        \n\
                        Executes the list of operations stacked in the Operation pipeline one \
                        after another from top down. The Operation pipeline editing is disabled \
                        during the computation. If any operation fails due to invalid input parameters, \
                        the computation stops and the error will be reported in the console log of the \
                        respective operation.");
                        ui.text_colored(self.colors.log_message_warn,"\n\
                        WARNING: The execution cannot be stopped. If it takes long time or crashes, \
                        the unsaved progress of the .hurban project file will be lost!");
                        wrap_token.pop(ui);
                    });
                }

                ui.next_column();

                let popping_tokens = if popping_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                if ui.button(
                    imgui::im_str!("Remove last operation (Del)"),
                    [-f32::MIN_POSITIVE, 25.0],
                ) && popping_enabled
                {
                    pop_stmt_clicked = true;
                    notifications.push(
                        current_time,
                        NotificationLevel::Warn,
                        "Removed last operation from the Operation pipeline.",
                    );
                }
                if let Some((color_token, style_token)) = popping_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }

                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                        ui.text_colored(self.colors.tooltip_text, "REMOVE LAST OPERATION FROM THE OPERATION PIPELINE\n\
                        \n\
                        Only the last operation in the sequence of operations stacked into \
                        the Operation pipeline can be removed.");
                        let text_color_token = ui.push_style_color(
                            imgui::StyleColor::Text,
                            self.colors.log_message_warn,
                        );
                        ui.text("\n\
                        The removal cannot be undone!");
                        text_color_token.pop(ui);
                        wrap_token.pop(ui);
                    });
                }

                ui.columns(1, imgui::im_str!("Autorun columns"), false);
                autorun_clicked =
                    ui.checkbox(imgui::im_str!("Run automatically"), &mut autorun_enabled);

                    if ui.is_item_hovered() {
                        ui.tooltip(|| {
                            let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                            ui.text_colored(self.colors.tooltip_text, "RUN RECOMPUTATION OF THE OPERATION PIPELINE AUTOMATICALLY\n\
                            \n\
                            Executes the list of operations stacked in the Sequence of \
                            operations one after another from top down automatically whenever a \
                            parameter or an operation changes in the Sequence of operations.");
                            ui.text_colored(self.colors.log_message_warn, "\n\
                            WARNING: The execution may take long or even hang the computer! If \
                            not sure how heavy is the geometry, turn the automatic recomputation off. \
                            The execution cannot be stopped. If it takes long time or crashes, \
                            the unsaved progress of the .hurban project file will be lost!");
                            wrap_token.pop(ui);
                        });
                    }

                ui.separator();

                let pushing_tokens = if pushing_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                ui.columns(3, imgui::im_str!("Add operations columns"), false);

                let mut previous_group_number = 0;
                let mut column_counter = 0;
                for (func_ident, func) in function_table {
                    let current_group_number = func_ident.0 / 1000_u64;
                    if current_group_number != previous_group_number {
                        while column_counter % 3 != 0 {
                            ui.next_column();
                            column_counter += 1;
                        }
                        ui.separator();
                    }
                    if ui.button(
                        &imgui::im_str!("{}", func.info().name),
                        [-f32::MIN_POSITIVE, 20.0],
                    ) && pushing_enabled
                    {
                        function_clicked = Some(func_ident);
                        notifications.push(
                            current_time,
                            NotificationLevel::Info,
                            format!("Added new operation to the Operation pipeline: {}.", func.info().name),
                        );
                    }

                    if ui.is_item_hovered() && !func.info().description.is_empty() {
                        ui.tooltip(|| {
                            let wrap_token = ui.push_text_wrap_pos(WRAP_POS_TOOLTIP_TEXT_PIXELS);
                            ui.text_colored(self.colors.tooltip_text, func.info().description);
                            wrap_token.pop(ui);
                        });
                    }

                    ui.next_column();
                    column_counter += 1;

                    previous_group_number = current_group_number;
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
                session
                    .next_free_var_ident()
                    .expect("Failed to find free variable identifier"),
                init_expr,
            ));

            session.push_prog_stmt(current_time, stmt);
            self.pipeline_window_state.borrow_mut().autoscroll = true;
        }

        if interpret_clicked {
            notifications.push(
                current_time,
                NotificationLevel::Info,
                "Execution of the Operation pipeline has started...",
            );
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
                let (var_decl_stmt_index, var_name) = session
                    .var_decl_stmt_index_and_var_name_for_ident(var_ident)
                    .expect("Failed to find name for ident");

                format_var_name(var_decl_stmt_index, var_name, ty == Ty::MeshArray)
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
            (
                imgui::StyleColor::PopupBg,
                self.colors.popup_window_background,
            ),
        ]);
        if let Some(combo_token) = combo.begin(ui) {
            for (index, var_ident) in visible_vars_iter.clone().enumerate() {
                let (var_decl_stmt_index, var_name) = session
                    .var_decl_stmt_index_and_var_name_for_ident(var_ident)
                    .expect("Failed to find name for ident");

                let text = format_var_name(var_decl_stmt_index, var_name, ty == Ty::MeshArray);
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
    var_decl_stmt_index: usize,
    var_name: &str,
    surround_with_brackets: bool,
) -> imgui::ImString {
    if surround_with_brackets {
        imgui::im_str!("[{}] #{}", var_name, var_decl_stmt_index + 1)
    } else {
        imgui::im_str!("{} #{}", var_name, var_decl_stmt_index + 1)
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
                    // FIXME: @Correctness Path stripping never works unless the
                    // models are located in the current directory.
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

use std::collections::HashMap;
use std::convert::TryFrom;

use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::interpreter::ast::LitExpr;
use crate::operation_manager::{Op, OpParamUiRepr, OpStatus, OpUiParam, OperationManager};
use crate::renderer::DrawGeometryMode;

const OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Regular.ttf");
const OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Bold.ttf");
const OPENSANS_LIGHT_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Light.ttf");

pub struct FontIds {
    _regular: imgui::FontId,
    _light: imgui::FontId,
    _bold: imgui::FontId,
}

/// Thin wrapper around imgui and its winit platform. Its main responsibilty
/// is to create UI frames which draw the UI itself.
pub struct Ui {
    imgui_context: imgui::Context,
    imgui_winit_platform: WinitPlatform,
    font_ids: FontIds,
}

impl Ui {
    /// Initializes imgui with default settings for our application.
    pub fn new(window: &winit::window::Window) -> Self {
        let mut imgui_context = imgui::Context::create();
        let mut style = imgui_context.style_mut();
        style.window_padding = [10.0, 10.0];

        imgui_context.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui_context);

        platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Default);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (20.0 * hidpi_factor) as f32;

        let regular_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_REGULAR_BYTES,
                size_pixels: font_size,
                config: None,
            }]);
        let bold_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_BOLD_BYTES,
                size_pixels: font_size,
                config: None,
            }]);
        let light_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_LIGHT_BYTES,
                size_pixels: font_size,
                config: None,
            }]);

        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        Ui {
            imgui_context,
            imgui_winit_platform: platform,
            font_ids: FontIds {
                _regular: regular_font_id,
                _bold: bold_font_id,
                _light: light_font_id,
            },
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

        UiFrame::new(
            &mut self.imgui_context,
            &self.imgui_winit_platform,
            &self.font_ids,
        )
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
    _font_ids: &'a FontIds,
}

impl<'a> UiFrame<'a> {
    pub fn new(
        imgui_context: &'a mut imgui::Context,
        imgui_winit_platform: &'a WinitPlatform,
        font_ids: &'a FontIds,
    ) -> Self {
        UiFrame {
            imgui_winit_platform,
            imgui_ui: imgui_context.frame(),
            _font_ids: font_ids,
        }
    }

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

    pub fn draw_renderer_settings_window(&self, draw_mode: &mut DrawGeometryMode) {
        let ui = &self.imgui_ui;
        let window_width = 250.0;
        let window_position = [
            self.imgui_ui.io().display_size[0] - window_width - 50.0,
            50.0,
        ];

        imgui::Window::new(imgui::im_str!("Renderer Settings"))
            .position(window_position, imgui::Condition::Always)
            .size([window_width, 200.0], imgui::Condition::Once)
            .build(ui, || {
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
                ui.radio_button(
                    imgui::im_str!("Shaded"),
                    draw_mode,
                    DrawGeometryMode::Shaded,
                );
                ui.radio_button(imgui::im_str!("Edges"), draw_mode, DrawGeometryMode::Edges);
                ui.radio_button(
                    imgui::im_str!("Shaded with Edges"),
                    draw_mode,
                    DrawGeometryMode::ShadedEdges,
                );
                ui.radio_button(
                    imgui::im_str!("Shaded with Edges (X-RAY)"),
                    draw_mode,
                    DrawGeometryMode::ShadedEdgesXray,
                );
            });
    }

    /// Draws child window for a single operation.
    pub fn draw_operation_window(&self, name: &str, params: &mut [OpUiParam], status: OpStatus) {
        let ui = &self.imgui_ui;

        let border_color = match status {
            OpStatus::Ready => int_to_float_color(255, 255, 255, 255),
            OpStatus::Running => int_to_float_color(0, 0, 255, 255),
            OpStatus::Finished => int_to_float_color(0, 255, 0, 255),
            OpStatus::Error => int_to_float_color(255, 0, 0, 255),
        };

        let border_color_token = ui.push_style_color(imgui::StyleColor::Border, border_color);

        imgui::ChildWindow::new(&imgui::im_str!("{}", name))
            .size([ui.window_content_region_width(), 150.0])
            .border(true)
            .always_auto_resize(true)
            .build(ui, || {
                for (id_counter, param) in params.iter_mut().enumerate() {
                    let display_name = imgui::im_str!("{}", &param.name);
                    let label = imgui::im_str!("");

                    ui.columns(2, imgui::im_str!(""), true);
                    ui.set_column_width(
                        ui.current_column_index(),
                        ui.window_content_region_width() / 3.0,
                    );
                    ui.text(&display_name);
                    ui.next_column();

                    // Since by default tokens are generated from lables, but
                    // ours are gonna be empty, we need to provide them manually.
                    let token = ui.push_id(id_counter as i32);

                    match param.value {
                        LitExpr::Int(ref mut value) => match param.repr {
                            OpParamUiRepr::IntInput => {
                                ui.input_int(&label, value).build();
                            }
                            OpParamUiRepr::IntSlider => {
                                imgui::Slider::new(&label, 1..=10).build(ui, value);
                            }
                            _ => {}
                        },

                        LitExpr::Uint(ref mut value) => match param.repr {
                            OpParamUiRepr::IntInput => {
                                // Imgui's `input_int` requires i32, but our Uint
                                // values are u32, so we need to cast them before
                                // and after.
                                let mut i32_value: i32 = *value as i32;
                                ui.input_int(&label, &mut i32_value).build();
                                *value = u32::try_from(i32_value)
                                    .expect("Failed to convert signed int to unsigned int");
                            }
                            OpParamUiRepr::IntSlider => {
                                imgui::Slider::new(&label, 1..=10).build(ui, value);
                            }
                            OpParamUiRepr::GeometryDropdown(ref choices) => {
                                // Imgui's combo box requires usize, but our Uint
                                // values are u32, so we need to cast them before
                                // and after.
                                let mut usize_value: usize = *value as usize;
                                imgui::ComboBox::new(&label).build_simple(
                                    ui,
                                    &mut usize_value,
                                    choices,
                                    &|item| std::borrow::Cow::from(imgui::im_str!("{}", item.1)),
                                );
                                *value = u32::try_from(usize_value)
                                    .expect("Failed to convert signed int to unsigned int");
                            }
                            _ => {}
                        },

                        LitExpr::Float(ref mut value) => match param.repr {
                            OpParamUiRepr::FloatInput => {
                                ui.input_float(&label, value).build();
                            }
                            OpParamUiRepr::FloatSlider => {
                                imgui::Slider::new(&label, 1.0..=10.0).build(ui, value);
                            }
                            _ => {}
                        },

                        LitExpr::Boolean(ref mut value) => match param.repr {
                            OpParamUiRepr::Checkbox => {
                                ui.checkbox(&label, value);
                            }
                            OpParamUiRepr::Radio => {
                                // There are two different elements, so they
                                // need extra tokens.
                                let token = ui.push_id("true");
                                ui.radio_button(&label, value, true);
                                token.pop(&ui);

                                ui.same_line(0.0);

                                let token = ui.push_id("false");
                                ui.radio_button(&label, value, false);
                                token.pop(&ui);
                            }
                            _ => {}
                        },
                        _ => {}
                    }

                    token.pop(&ui);

                    ui.next_column();
                }
            });

        border_color_token.pop(&ui);
    }

    /// Draws window for all selected operations.
    pub fn draw_operations_window<'b>(
        &self,
        available_operations: &'b HashMap<String, Op>,
        operation_ui: &mut OperationManager,
    ) -> (bool, bool) {
        let ui = &self.imgui_ui;
        let mut run_button_clicked = false;
        let mut remove_button_clicked = false;
        let last_operation_successful = operation_ui.last_operation_successful();
        let window_height = self.imgui_ui.io().display_size[1] - 100.0;

        imgui::Window::new(&imgui::im_str!("Operations"))
            .movable(false)
            .position([50.0, 50.0], imgui::Condition::Always)
            .size([500.0, window_height], imgui::Condition::FirstUseEver)
            .build(ui, || {
                if last_operation_successful {
                    for (name, ui_op) in available_operations {
                        if ui.button(&imgui::im_str!("{}", name), [90.0, 30.0]) {
                            operation_ui.add_operation(ui_op.clone());
                        }
                    }
                }

                for (i, selected_op) in &mut operation_ui.selected_ops_iter_mut().enumerate() {
                    self.draw_operation_window(
                        &format!("{} #{}", &selected_op.op.name, i),
                        &mut selected_op.op.params,
                        selected_op.status,
                    );
                }

                if operation_ui.runnable() {
                    if ui.button(imgui::im_str!("Remove last operation"), [256.0, 32.0]) {
                        remove_button_clicked = true;
                    }

                    if ui.button(imgui::im_str!("Run"), [64.0, 32.0]) {
                        run_button_clicked = true;
                    }
                }
            });

        (run_button_clicked, remove_button_clicked)
    }
}

fn int_to_float_color(r: u8, g: u8, b: u8, a: u8) -> [f32; 4] {
    [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        f32::from(a) / 255.0,
    ]
}

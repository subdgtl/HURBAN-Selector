use std::time::Instant;

use wgpu::winit;

use crate::imgui_input::{HiDpiMode, WinitPlatform};

const OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Regular.ttf");
const OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Bold.ttf");
const OPENSANS_LIGHT_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Light.ttf");

pub struct FontIds {
    _regular: imgui::FontId,
    light: imgui::FontId,
    bold: imgui::FontId,
}

/// Thin wrapper around imgui and its winit platform. Its main responsibilty
/// is to create UI frames which draw the UI itself.
pub struct Ui<'a> {
    window: &'a winit::Window,
    imgui_context: imgui::Context,
    imgui_winit_platform: WinitPlatform,
    font_ids: FontIds,
}

impl<'a> Ui<'a> {
    /// Initializes imgui with default settings for our application.
    pub fn new(window: &'a winit::Window) -> Self {
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
            window,
            imgui_context,
            imgui_winit_platform: platform,
            font_ids: FontIds {
                _regular: regular_font_id,
                bold: bold_font_id,
                light: light_font_id,
            },
        }
    }

    pub fn fonts(&mut self) -> imgui::FontAtlasRefMut {
        self.imgui_context.fonts()
    }

    pub fn handle_event(&mut self, event: &winit::Event, time: Instant) {
        self.imgui_winit_platform
            .handle_event(self.imgui_context.io_mut(), &self.window, &event, time);
    }

    pub fn prepare_frame(&mut self) -> UiFrame {
        self.imgui_winit_platform
            .prepare_frame(self.imgui_context.io_mut(), &self.window)
            .expect("Failed to start imgui frame");

        UiFrame::new(
            &self.window,
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
    window: &'a winit::Window,
    imgui_winit_platform: &'a WinitPlatform,
    imgui_ui: imgui::Ui<'a>,
    font_ids: &'a FontIds,
}

impl<'a> UiFrame<'a> {
    pub fn new(
        window: &'a winit::Window,
        imgui_context: &'a mut imgui::Context,
        imgui_winit_platform: &'a WinitPlatform,
        font_ids: &'a FontIds,
    ) -> Self {
        UiFrame {
            window,
            imgui_winit_platform,
            imgui_ui: imgui_context.frame(),
            font_ids,
        }
    }

    pub fn want_capture_keyboard(&self) -> bool {
        self.imgui_ui.io().want_capture_keyboard
    }

    pub fn want_capture_mouse(&self) -> bool {
        self.imgui_ui.io().want_capture_mouse
    }

    pub fn render(self) -> &'a imgui::DrawData {
        self.imgui_winit_platform
            .prepare_render(&self.imgui_ui, &self.window);
        self.imgui_ui.render()
    }

    pub fn draw_fps_window(&self) {
        let ui = &self.imgui_ui;

        ui.window(imgui::im_str!("FPS"))
            .position([450.0, 50.0], imgui::Condition::Always)
            .build(|| {
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
            });
    }

    /// Draws window with list of model filenames. If any of them is clicked, the
    /// filename is returned for further processing.
    pub fn draw_model_window(
        &self,
        filenames: &[String],
        selected_filename: &str,
        loading_progress: f32,
    ) -> Option<String> {
        let ui = &self.imgui_ui;
        let _window_styles = ui.push_style_vars(&[
            imgui::StyleVar::WindowRounding(0.0),
            imgui::StyleVar::ScrollbarRounding(0.0),
            imgui::StyleVar::FramePadding([20.0, 15.0]),
        ]);
        let _button_style = ui.push_style_var(imgui::StyleVar::ButtonTextAlign([-1.0, 0.5]));
        let mut clicked_button = None;

        let _regular_font_token = ui.push_font(self.font_ids.bold);

        let _default_colors = self.imgui_ui.push_style_colors(&[
            (
                imgui::StyleColor::WindowBg,
                int_to_float_color(87, 90, 28, 128),
            ),
            (
                imgui::StyleColor::TitleBg,
                int_to_float_color(25, 75, 113, 255),
            ),
            (
                imgui::StyleColor::Button,
                int_to_float_color(99, 129, 79, 255),
            ),
        ]);

        let window_width = 350.0;
        let window_height = self.imgui_ui.io().display_size[1] - 100.0;

        ui.window(imgui::im_str!("H.U.R.B.A.N. Selector"))
            .position([50.0, 50.0], imgui::Condition::Always)
            .size([window_width, window_height], imgui::Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                let inner_width = window_width - 20.0;
                let _light_font_token = ui.push_font(self.font_ids.light);

                ui.child_frame(imgui::im_str!("progress bar"), [inner_width, 40.0])
                    .build(|| {
                        let _progress_bar_text_color = ui.push_style_color(
                            imgui::StyleColor::Text,
                            int_to_float_color(0, 0, 0, 255),
                        );

                        ui.progress_bar(loading_progress)
                            .size([inner_width, 30.0])
                            .overlay_text(imgui::im_str!(""))
                            .build();
                    });

                ui.child_frame(
                    imgui::im_str!("model list"),
                    [inner_width, window_height - 115.0],
                )
                .build(|| {
                    for filename in filenames {
                        let mut _selected_button_colors: imgui::ColorStackToken;

                        if selected_filename == filename {
                            _selected_button_colors = self.imgui_ui.push_style_colors(&[
                                (
                                    imgui::StyleColor::Button,
                                    int_to_float_color(232, 210, 20, 255),
                                ),
                                (imgui::StyleColor::Text, int_to_float_color(0, 0, 0, 255)),
                            ]);
                        }

                        if ui.button(&imgui::im_str!("{}", filename), [inner_width, 60.0]) {
                            clicked_button = Some(filename.clone());
                        }
                    }
                })
            });

        clicked_button
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

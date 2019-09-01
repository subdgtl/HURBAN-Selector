use imgui_winit_support::{HiDpiMode, WinitPlatform};
use wgpu::winit;

const OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Regular.ttf");
const OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Bold.ttf");
const OPENSANS_LIGHT_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Light.ttf");

/// Thin wrapper around imgui and its winit platform. Its main responsibilty
/// is to create UI frames which draw the UI itself.
pub struct Ui<'a> {
    window: &'a winit::Window,
    imgui_context: imgui::Context,
    imgui_winit_platform: WinitPlatform,
    regular_font_id: imgui::FontId,
    bold_font_id: imgui::FontId,
    light_font_id: imgui::FontId,
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
        let font_size = (13.0 * hidpi_factor) as f32;

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
            regular_font_id,
            bold_font_id,
            light_font_id,
        }
    }

    pub fn handle_event(&mut self, event: &winit::Event) {
        self.imgui_winit_platform
            .handle_event(self.imgui_context.io_mut(), &self.window, &event);
    }

    pub fn prepare_frame(&mut self) -> UiFrame {
        self.imgui_winit_platform
            .prepare_frame(self.imgui_context.io_mut(), &self.window)
            .expect("Failed to start imgui frame");

        UiFrame::new(
            &self.window,
            &mut self.imgui_context,
            &self.imgui_winit_platform,
            &self.regular_font_id,
            &self.bold_font_id,
            &self.light_font_id,
        )
    }

    pub fn imgui_context(&mut self) -> &mut imgui::Context {
        &mut self.imgui_context
    }
}

/// This structure is responsible for drawing and rendering of a single UI
/// frame.
pub struct UiFrame<'a> {
    window: &'a winit::Window,
    imgui_winit_platform: &'a WinitPlatform,
    imgui_ui: imgui::Ui<'a>,
    regular_font_id: &'a imgui::FontId,
    bold_font_id: &'a imgui::FontId,
    light_font_id: &'a imgui::FontId,
}

impl<'a> UiFrame<'a> {
    pub fn new(
        window: &'a winit::Window,
        imgui_context: &'a mut imgui::Context,
        imgui_winit_platform: &'a WinitPlatform,
        regular_font_id: &'a imgui::FontId,
        bold_font_id: &'a imgui::FontId,
        light_font_id: &'a imgui::FontId,
    ) -> Self {
        UiFrame {
            window,
            imgui_winit_platform,
            imgui_ui: imgui_context.frame(),
            regular_font_id,
            bold_font_id,
            light_font_id,
        }
    }

    pub fn io(&self) -> &imgui::Io {
        self.imgui_ui.io()
    }

    pub fn render(self) -> &'a imgui::DrawData {
        self.imgui_winit_platform
            .prepare_render(&self.imgui_ui, &self.window);
        self.imgui_ui.render()
    }

    pub fn draw_fps_window(&self) {
        let ui = &self.imgui_ui;

        ui.window(imgui::im_str!("FPS")).build(|| {
            ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
        });
    }

    /// Draws window with list of model filenames. If any of them is clicked, the
    /// filename is returned for further processing.
    pub fn draw_model_window(&self, filenames: &[String], loading_progress: f32) -> Option<String> {
        let ui = &self.imgui_ui;
        let _button_style = ui.push_style_var(imgui::StyleVar::ButtonTextAlign([-1.0, 0.0]));
        let mut clicked_button = None;

        ui.window(imgui::im_str!("Models"))
            .position([50.0, 200.0], imgui::Condition::Always)
            .movable(false)
            .resizable(false)
            .build(|| {
                ui.progress_bar(loading_progress).build();

                for filename in filenames {
                    if ui.button(&imgui::im_str!("{}", filename), [180.0, 20.0]) {
                        clicked_button = Some(filename.clone());
                    }
                }
            });

        clicked_button
    }
}

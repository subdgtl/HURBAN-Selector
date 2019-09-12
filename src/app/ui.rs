use imgui_winit_support::{HiDpiMode, WinitPlatform};

const OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../../resources/OpenSans-Regular.ttf");
const OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../../resources/OpenSans-Bold.ttf");
const OPENSANS_LIGHT_BYTES: &[u8] = include_bytes!("../../resources/OpenSans-Light.ttf");

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
    _font_ids: FontIds,
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
            _font_ids: FontIds {
                _regular: regular_font_id,
                _light: light_font_id,
                _bold: bold_font_id,
            },
        }
    }

    pub fn fonts(&mut self) -> imgui::FontAtlasRefMut {
        self.imgui_context.fonts()
    }

    pub fn want_capture_keyboard(&self) -> bool {
        self.imgui_context.io().want_capture_keyboard
    }

    pub fn want_capture_mouse(&self) -> bool {
        self.imgui_context.io().want_capture_mouse
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
            &self._font_ids,
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
        _font_ids: &'a FontIds,
    ) -> Self {
        UiFrame {
            imgui_winit_platform,
            imgui_ui: imgui_context.frame(),
            _font_ids,
        }
    }

    pub fn render(self, window: &winit::window::Window) -> &'a imgui::DrawData {
        self.imgui_winit_platform
            .prepare_render(&self.imgui_ui, window);
        self.imgui_ui.render()
    }

    pub fn draw_fps_window(&self) {
        let ui = &self.imgui_ui;

        imgui::Window::new(imgui::im_str!("FPS"))
            .position([450.0, 50.0], imgui::Condition::Always)
            .build(ui, || {
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
            });
    }
}

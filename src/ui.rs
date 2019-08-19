use std::time::Instant;

use imgui_winit_support::{HiDpiMode, WinitPlatform};
use wgpu::winit;

/// Convenience wrapper around imgui and its winit support.
pub struct Ui<'w> {
    platform: WinitPlatform,
    window: &'w wgpu::winit::Window,
    context: imgui::Context,
}

impl<'w> Ui<'w> {
    /// Creates UI with hardcoded default settings.
    pub fn new(window: &'w winit::Window) -> Self {
        let mut imgui_context = imgui::Context::create();
        let mut style = imgui_context.style_mut();
        style.window_padding = [10.0, 10.0];

        imgui_context.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui_context);

        platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Default);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;

        imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    size_pixels: font_size,
                    ..imgui::FontConfig::default()
                }),
            }]);

        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        Ui {
            platform,
            window,
            context: imgui_context,
        }
    }

    pub fn context(&mut self) -> &mut imgui::Context {
        &mut self.context
    }

    pub fn update_delta_time(&mut self, delta: Instant) -> Instant {
        self.context.io_mut().update_delta_time(delta)
    }

    /// Processes winit event and returns whether UI capture keyboard or mouse
    /// events.
    pub fn process_event(&mut self, event: &winit::Event) -> (bool, bool) {
        self.platform
            .handle_event(self.context.io_mut(), self.window, &event);

        (
            self.context.io().want_capture_keyboard,
            self.context.io().want_capture_mouse,
        )
    }

    /// Prepares imgui frame, creates windows defined inside of `callback`,
    /// prepares rendering and finally returns imgui's `DrawData` that can be
    /// passed into renderer.
    pub fn create<F>(&mut self, callback: F) -> &imgui::DrawData
    where
        F: FnOnce(&imgui::Ui),
    {
        self.platform
            .prepare_frame(self.context.io_mut(), self.window)
            .expect("Failed to start imgui frame");

        let ui = self.context.frame();

        callback(&ui);

        self.platform.prepare_render(&ui, self.window);

        ui.render()
    }
}

pub fn draw_fps_window(ui: &imgui::Ui) {
    ui.window(imgui::im_str!("FPS")).build(|| {
        ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
    });
}

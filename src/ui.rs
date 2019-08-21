use imgui_winit_support::{HiDpiMode, WinitPlatform};
use wgpu::winit;

pub fn init(window: &winit::Window) -> (imgui::Context, WinitPlatform) {
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

    (imgui_context, platform)
}

pub fn draw_fps_window(ui: &imgui::Ui) {
    ui.window(imgui::im_str!("FPS")).build(|| {
        ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));
    });
}

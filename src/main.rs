#![windows_subsystem = "windows"]

use std::env;

use hurban_selector as hs;

fn main() {
    let theme = env::var("HS_THEME")
        .ok()
        .map(|theme| match theme.as_str() {
            "dark" => hs::Theme::Dark,
            "funky" => hs::Theme::Funky,
            unsupported_theme => {
                panic!("Unsupported theme value requested: {}", unsupported_theme,)
            }
        })
        .unwrap_or(hs::Theme::Dark);

    let fullscreen = env::var("HS_FULLSCREEN")
        .ok()
        .map(|fullscreen| match fullscreen.as_str() {
            "0" => false,
            "1" => true,
            unsupported_fullscreen => panic!(
                "Unsupported fullscreen value requested: {}",
                unsupported_fullscreen,
            ),
        })
        .unwrap_or(false);

    let msaa = env::var("HS_MSAA")
        .ok()
        .map(|msaa| match msaa.as_str() {
            "1" => hs::Msaa::Disabled,
            "4" => hs::Msaa::X4,
            "8" => hs::Msaa::X8,
            "16" => hs::Msaa::X16,
            unsupported_msaa => panic!("Unsupported MSAA value requested: {}", unsupported_msaa),
        })
        .unwrap_or(hs::Msaa::Disabled);

    let present_mode = env::var("HS_VSYNC")
        .ok()
        .map(|vsync| match vsync.as_str() {
            "0" => hs::PresentMode::NoVsync,
            "1" => hs::PresentMode::Vsync,
            unsupported_vsync => panic!(
                "Unsupported vsync behavior requested: {}",
                unsupported_vsync,
            ),
        })
        .unwrap_or(hs::PresentMode::Vsync);

    let gpu_backend = env::var("HS_GPU_BACKEND")
        .ok()
        .map(|backend| match backend.as_str() {
            "vulkan" => hs::GpuBackend::Vulkan,
            "d3d12" => hs::GpuBackend::D3d12,
            "metal" => hs::GpuBackend::Metal,
            _ => panic!("Unknown gpu backend requested"),
        });

    let app_log_level = env::var("HS_APP_LOG_LEVEL")
        .ok()
        .map(|app_log_level| match app_log_level.as_str() {
            "error" => hs::LogLevel::Error,
            "warning" => hs::LogLevel::Warning,
            "info" => hs::LogLevel::Info,
            "debug" => hs::LogLevel::Debug,
            _ => panic!("Unknown library log level requested"),
        });

    let lib_log_level = env::var("HS_LIB_LOG_LEVEL")
        .ok()
        .map(|lib_log_level| match lib_log_level.as_str() {
            "error" => hs::LogLevel::Error,
            "warning" => hs::LogLevel::Warning,
            "info" => hs::LogLevel::Info,
            "debug" => hs::LogLevel::Debug,
            _ => panic!("Unknown library log level requested"),
        });

    hs::init_and_run(hs::Options {
        theme,
        fullscreen,
        msaa,
        present_mode,
        gpu_backend,
        app_log_level,
        lib_log_level,
    });
}

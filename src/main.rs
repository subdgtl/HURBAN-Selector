#![windows_subsystem = "windows"]

use std::env;

use hurban_selector as hs;

fn main() {
    env_logger::init();

    let msaa = env::var("HS_MSAA")
        .ok()
        .map(|msaa| match msaa.as_str() {
            "0" => hs::Msaa::Disabled,
            "4" => hs::Msaa::X4,
            "8" => hs::Msaa::X8,
            "16" => hs::Msaa::X16,
            unsupported_msaa => panic!("Unsupported MSAA value requested: {}", unsupported_msaa),
        })
        .unwrap_or(hs::Msaa::Disabled);

    let present_mode = env::var("HS_VSYNC")
        .ok()
        .map(|vsync| match vsync.as_str() {
            "1" => hs::PresentMode::Vsync,
            "0" => hs::PresentMode::NoVsync,
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

    hs::init_and_run(hs::Options {
        msaa,
        present_mode,
        gpu_backend,
    });
}

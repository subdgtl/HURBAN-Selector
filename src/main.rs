use std::time::Instant;

#[cfg(debug_assertions)]
use env_logger;
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
use wgpu;
use wgpu::winit;
use wgpu::winit::dpi::PhysicalSize;

use crate::viewport_renderer::ViewportRenderer;

mod primitives;
mod viewport_renderer;

const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

fn main() {
    #[cfg(debug_assertions)]
    env_logger::init();

    let mut event_loop = winit::EventsLoop::new();
    let window = winit::Window::new(&event_loop).expect("Failed to create window.");
    let window_size = window
        .get_inner_size()
        .expect("Failed to get window inner size")
        .to_physical(window.get_hidpi_factor());

    let wgpu_instance = wgpu::Instance::new();
    let surface = wgpu_instance.create_surface(&window);
    let adapter = wgpu_instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::HighPerformance,
    });

    let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });
    let mut swap_chain = create_swap_chain(&device, &surface, window_size);

    // FIXME: This will later come from the camera system
    let camera_position_initial = &Point3::new(1.5f32, -5.0, 3.0);

    let mut viewport_renderer = ViewportRenderer::new(
        &mut device,
        SWAP_CHAIN_FORMAT,
        window_size,
        Matrix4::look_at_rh(
            &camera_position_initial,
            &Point3::new(0f32, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
        ),
    );

    // FIXME: This is just temporary code so that we can see something
    // in the scene and know the renderer works.
    let (cube1_v, cube1_i) = primitives::cube([0.0, 0.0, 0.0], 0.5);
    let (cube2_v, cube2_i) = primitives::cube([5.0, 5.0, 0.0], 0.7);
    let (cube3_v, cube3_i) = primitives::cube([5.0, 0.0, 0.0], 1.0);
    let (cube4_v, cube4_i) = primitives::cube([0.0, 0.0, 5.0], 1.5);
    let plane1_v = primitives::plane([0.0, 0.0, 20.0], 10.0);
    let plane2_v = primitives::plane([0.0, 0.0, -20.0], 10.0);

    let cube1 = viewport_renderer
        .add_geometry_indexed(&device, &cube1_v, &cube1_i)
        .unwrap();
    let cube2 = viewport_renderer
        .add_geometry_indexed(&device, &cube2_v, &cube2_i)
        .unwrap();
    let cube3 = viewport_renderer
        .add_geometry_indexed(&device, &cube3_v, &cube3_i)
        .unwrap();
    let cube4 = viewport_renderer
        .add_geometry_indexed(&device, &cube4_v, &cube4_i)
        .unwrap();
    let plane1 = viewport_renderer.add_geometry(&device, &plane1_v).unwrap();
    let plane2 = viewport_renderer.add_geometry(&device, &plane2_v).unwrap();

    let time_start = Instant::now();
    let mut time = time_start;

    let mut running = true;
    while running {
        let (_duration_last_frame, duration_running) = {
            let now = Instant::now();
            let duration_last_frame = now.duration_since(time);
            let duration_running = now.duration_since(time_start);
            time = now;

            (duration_last_frame, duration_running)
        };

        event_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                match event {
                    winit::WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(code),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        if let winit::VirtualKeyCode::Q = code {
                            running = false;
                        }
                    }
                    winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::Resized(logical_size) => {
                        let physical_size = logical_size.to_physical(window.get_hidpi_factor());
                        log::debug!(
                            "Window resized to new size: logical [{},{}], physical [{},{}]",
                            logical_size.width,
                            logical_size.height,
                            physical_size.width,
                            physical_size.height,
                        );

                        swap_chain = create_swap_chain(&device, &surface, physical_size);
                        viewport_renderer.set_screen_size(&mut device, physical_size);
                    }
                    _ => (),
                }
            }
        });

        // FIXME: This will eventually come from the camera system
        let camera_rotation =
            Rotation3::new(Vector3::z() * duration_running.as_millis() as f32 / 1000.0);
        let camera_position = camera_rotation * camera_position_initial;
        let view_matrix = Matrix4::look_at_rh(
            &camera_position,
            &Point3::new(0f32, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
        );

        viewport_renderer.set_view_matrix(&mut device, view_matrix);

        let frame = swap_chain.get_next_texture();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        viewport_renderer.draw_geometry(
            &mut encoder,
            &frame.view,
            &[cube1, cube2, cube3, cube4, plane1, plane2],
        );

        device.get_queue().submit(&[encoder.finish()]);
    }

    viewport_renderer.remove_geometry(cube1);
    viewport_renderer.remove_geometry(cube2);
    viewport_renderer.remove_geometry(cube3);
    viewport_renderer.remove_geometry(cube4);
    viewport_renderer.remove_geometry(plane1);
    viewport_renderer.remove_geometry(plane2);
}

fn create_swap_chain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    window_size: PhysicalSize,
) -> wgpu::SwapChain {
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: SWAP_CHAIN_FORMAT,
            width: window_size.width.round() as u32,
            height: window_size.height.round() as u32,
            present_mode: wgpu::PresentMode::Vsync,
        },
    )
}

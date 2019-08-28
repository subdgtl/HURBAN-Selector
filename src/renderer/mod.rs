use nalgebra::base::Matrix4;
use wgpu::winit;

pub use self::scene_renderer::{SceneRendererGeometry, SceneRendererGeometryId};

use self::imgui_renderer::{ImguiRenderer, ImguiRendererClearFlags, ImguiRendererOptions};
use self::scene_renderer::{
    SceneRenderer, SceneRendererAddGeometryError, SceneRendererClearFlags, SceneRendererOptions,
};

#[macro_use]
mod common;

mod imgui_renderer;
mod scene_renderer;

const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::D32Float;

#[derive(Debug, Clone, PartialEq)]
pub struct RendererOptions {
    pub msaa: Msaa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Msaa {
    Disabled,
    X4,
    X8,
    X16,
}

impl Msaa {
    pub fn enabled(self) -> bool {
        match self {
            Msaa::Disabled => false,
            _ => true,
        }
    }

    pub fn sample_count(self) -> u32 {
        match self {
            Msaa::Disabled => 1,
            Msaa::X4 => 4,
            Msaa::X8 => 8,
            Msaa::X16 => 16,
        }
    }
}

pub struct Renderer {
    device: wgpu::Device,
    surface: wgpu::Surface,
    swap_chain: wgpu::SwapChain,
    msaa_texture_view: Option<wgpu::TextureView>,
    depth_texture_view: wgpu::TextureView,
    scene_renderer: SceneRenderer,
    imgui_renderer: ImguiRenderer,
    options: RendererOptions,
}

impl Renderer {
    pub fn new(
        instance: &wgpu::Instance,
        window: &winit::Window,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
        imgui_font_atlas: imgui::FontAtlasRefMut,
        options: RendererOptions,
    ) -> Self {
        let surface = instance.create_surface(window);
        let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
            power_preference: wgpu::PowerPreference::HighPerformance,
        });
        let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        });

        let window_size = window
            .get_inner_size()
            .expect("Failed to get window inner size")
            .to_physical(window.get_hidpi_factor());
        let (width, height) = (window_size.width as u32, window_size.height as u32);

        let swap_chain = create_swap_chain(&device, &surface, width, height);
        let msaa_texture = if options.msaa.enabled() {
            Some(create_msaa_texture(
                &device,
                width,
                height,
                options.msaa.sample_count(),
            ))
        } else {
            None
        };
        let depth_texture =
            create_depth_texture(&device, width, height, options.msaa.sample_count());

        let scene_renderer = SceneRenderer::new(
            &mut device,
            projection_matrix,
            view_matrix,
            SceneRendererOptions {
                sample_count: options.msaa.sample_count(),
                output_color_attachment_format: SWAP_CHAIN_FORMAT,
                output_depth_attachment_format: DEPTH_FORMAT,
            },
        );

        let imgui_renderer = ImguiRenderer::new(
            imgui_font_atlas,
            &mut device,
            ImguiRendererOptions {
                sample_count: options.msaa.sample_count(),
                output_color_attachment_format: SWAP_CHAIN_FORMAT,
            },
        )
        .expect("Failed to create imgui renderer");

        Self {
            device,
            surface,
            swap_chain,
            msaa_texture_view: msaa_texture.map(|texture| texture.create_default_view()),
            depth_texture_view: depth_texture.create_default_view(),
            scene_renderer,
            imgui_renderer,
            options,
        }
    }

    /// Update camera matrices (projection matrix and view matrix).
    pub fn set_camera_matrices(
        &mut self,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) {
        self.scene_renderer
            .set_camera_matrices(&mut self.device, projection_matrix, view_matrix);
    }

    /// Update window size. Recreate swap chain and all render target
    /// textures.
    pub fn set_window_size(&mut self, window_size: winit::dpi::PhysicalSize) {
        let (width, height) = (
            window_size.width.round() as u32,
            window_size.height.round() as u32,
        );

        self.swap_chain = create_swap_chain(&self.device, &self.surface, width, height);

        if self.options.msaa.enabled() {
            let msaa_texture = create_msaa_texture(
                &self.device,
                width,
                height,
                self.options.msaa.sample_count(),
            );

            self.msaa_texture_view = Some(msaa_texture.create_default_view());
        }

        let depth_texture = create_depth_texture(
            &self.device,
            width,
            height,
            self.options.msaa.sample_count(),
        );
        self.depth_texture_view = depth_texture.create_default_view();
    }

    pub fn add_scene_geometry(
        &mut self,
        geometry: &SceneRendererGeometry,
    ) -> Result<SceneRendererGeometryId, SceneRendererAddGeometryError> {
        self.scene_renderer.add_geometry(&self.device, geometry)
    }

    pub fn remove_scene_geometry(&mut self, id: SceneRendererGeometryId) {
        self.scene_renderer.remove_geometry(id);
    }

    pub fn add_ui_texture_rgba8_unorm(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> imgui::TextureId {
        self.imgui_renderer
            .add_texture_rgba8_unorm(&mut self.device, width, height, data)
    }

    pub fn remove_ui_texture(&mut self, id: imgui::TextureId) {
        self.imgui_renderer.remove_texture(id);
    }

    pub fn begin_render(&mut self) -> Render {
        let frame = self.swap_chain.get_next_texture();
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        Render {
            color_needs_clearing: true,
            depth_needs_clearing: true,
            device: &mut self.device,
            frame,
            encoder: Some(encoder),
            msaa_attachment: self.msaa_texture_view.as_ref(),
            depth_attachment: &self.depth_texture_view,
            scene_renderer: &self.scene_renderer,
            imgui_renderer: &self.imgui_renderer,
        }
    }
}

pub struct Render<'a> {
    color_needs_clearing: bool,
    depth_needs_clearing: bool,
    device: &'a mut wgpu::Device,
    frame: wgpu::SwapChainOutput<'a>,
    encoder: Option<wgpu::CommandEncoder>,
    msaa_attachment: Option<&'a wgpu::TextureView>,
    depth_attachment: &'a wgpu::TextureView,
    scene_renderer: &'a SceneRenderer,
    imgui_renderer: &'a ImguiRenderer,
}

impl Render<'_> {
    pub fn draw_geometry(&mut self, ids: &[SceneRendererGeometryId]) {
        let mut clear_flags = SceneRendererClearFlags::empty();
        if self.color_needs_clearing {
            clear_flags.insert(SceneRendererClearFlags::COLOR);
        }
        if self.depth_needs_clearing {
            clear_flags.insert(SceneRendererClearFlags::DEPTH);
        }

        self.scene_renderer.draw_geometry(
            clear_flags,
            self.encoder
                .as_mut()
                .expect("Need encoder to record drawing"),
            &self.frame.view,
            self.msaa_attachment,
            &self.depth_attachment,
            ids,
        );

        self.color_needs_clearing = false;
        self.depth_needs_clearing = false;
    }

    pub fn draw_ui(&mut self, draw_data: &imgui::DrawData) {
        let mut clear_flags = ImguiRendererClearFlags::empty();
        if self.color_needs_clearing {
            clear_flags.insert(ImguiRendererClearFlags::COLOR);
        }

        self.imgui_renderer
            .draw_ui(
                clear_flags,
                self.device,
                self.encoder
                    .as_mut()
                    .expect("Need encoder to record drawing"),
                &self.frame.view,
                self.msaa_attachment,
                draw_data,
            )
            .expect("Imgui drawing failed");

        self.color_needs_clearing = false;
    }

    pub fn finish(mut self) {
        let encoder = self.encoder.take().expect("Can't finish rendering twice");
        self.device.get_queue().submit(&[encoder.finish()]);
    }
}

impl Drop for Render<'_> {
    fn drop(&mut self) {
        assert!(
            self.encoder.is_none(),
            "Rendering must be finished by the time it goes out of scope"
        );
    }
}

fn create_swap_chain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    width: u32,
    height: u32,
) -> wgpu::SwapChain {
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: SWAP_CHAIN_FORMAT,
            width,
            height,
            present_mode: wgpu::PresentMode::Vsync,
        },
    )
}

fn create_msaa_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::Texture {
    assert!(
        sample_count > 1,
        "Msaa texture shouldn't be created if not multisampling"
    );

    device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: SWAP_CHAIN_FORMAT,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    })
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    })
}

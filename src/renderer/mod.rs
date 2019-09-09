pub use self::scene_renderer::{SceneRendererGeometry, SceneRendererGeometryId};

use std::fmt;

use nalgebra::base::Matrix4;

use self::imgui_renderer::{ImguiRenderer, ImguiRendererOptions};
use self::scene_renderer::{
    SceneRenderer, SceneRendererAddGeometryError, SceneRendererClearFlags, SceneRendererOptions,
};

#[macro_use]
mod common;

mod imgui_renderer;
mod scene_renderer;

const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug, Clone, PartialEq)]
pub struct RendererOptions {
    pub msaa: Msaa,
    pub present_mode: PresentMode,

    /// If present, try to force a gpu backend for the renderer to use
    pub backend: Option<Backend>,
}

/// Multi-sampling setting. Can be either disabled (1 sample per
/// pixel), or 4/8/16 samples per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Whether the renderer should wait for the flip when submitting the
/// render pass to the backbuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    NoVsync,
    Vsync,
}

/// The rendering backend used by `wgpu-rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    D3d12,
    Metal,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Backend::Vulkan => write!(f, "Vulkan"),
            Backend::D3d12 => write!(f, "D3D12"),
            Backend::Metal => write!(f, "Metal"),
        }
    }
}

/// High level renderer abstraction over wgpu-rs.
///
/// Handles GPU resources (swap chain, msaa buffer, depth buffer) and
/// their resizing as well as geometry and textures stored for
/// drawing.
///
/// Drawing happens within a single wgpu command encoder, which is
/// passed to the underlying scene and UI renderers to fill it with
/// draw commands. Use `renderer.begin_render_pass()` to start
/// recording draw commands and `render_pass.submit()` to execute
/// them.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
        window: &winit::window::Window,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
        imgui_font_atlas: imgui::FontAtlasRefMut,
        options: RendererOptions,
    ) -> Self {
        let backends = match options.backend {
            // FIXME: Vulkan on macOS does not actually work, the cargo feature does not enable it
            Some(Backend::Vulkan) => wgpu::BackendBit::VULKAN,
            Some(Backend::D3d12) => wgpu::BackendBit::DX12,
            Some(Backend::Metal) => wgpu::BackendBit::METAL,
            None => wgpu::BackendBit::PRIMARY,
        };

        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            backends,
        })
        .expect("Failed to acquire GPU adapter");

        let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        });

        let window_size = window.inner_size().to_physical(window.hidpi_factor());
        let (width, height) = (window_size.width as u32, window_size.height as u32);

        let swap_chain = create_swap_chain(&device, &surface, width, height, options.present_mode);
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
            &device,
            &mut queue,
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
            &device,
            &mut queue,
            ImguiRendererOptions {
                sample_count: options.msaa.sample_count(),
                output_color_attachment_format: SWAP_CHAIN_FORMAT,
            },
        )
        .expect("Failed to create imgui renderer");

        Self {
            device,
            queue,
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
        self.scene_renderer.set_camera_matrices(
            &self.device,
            &mut self.queue,
            projection_matrix,
            view_matrix,
        );
    }

    /// Update window size. Recreate swap chain and all render target
    /// textures.
    pub fn set_window_size(&mut self, window_size: winit::dpi::PhysicalSize) {
        let (width, height) = (
            window_size.width.round() as u32,
            window_size.height.round() as u32,
        );

        self.swap_chain = create_swap_chain(
            &self.device,
            &self.surface,
            width,
            height,
            self.options.present_mode,
        );

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

    /// Upload geometry to the GPU to be used in scene rendering. It
    /// will be available for drawing in subsequent render passes.
    pub fn add_scene_geometry(
        &mut self,
        geometry: &SceneRendererGeometry,
    ) -> Result<SceneRendererGeometryId, SceneRendererAddGeometryError> {
        self.scene_renderer.add_geometry(&self.device, geometry)
    }

    /// Remove geometry from the GPU.
    pub fn remove_scene_geometry(&mut self, id: SceneRendererGeometryId) {
        self.scene_renderer.remove_geometry(id);
    }

    /// Upload an RGBA8 texture to the GPU to be used in UI
    /// rendering. It will be available for drawing in the subsequent
    /// render passes.
    pub fn add_ui_texture_rgba8_unorm(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> imgui::TextureId {
        self.imgui_renderer.add_texture_rgba8_unorm(
            &self.device,
            &mut self.queue,
            width,
            height,
            data,
        )
    }

    /// Remove texture from the GPU.
    pub fn remove_ui_texture(&mut self, id: imgui::TextureId) {
        self.imgui_renderer.remove_texture(id);
    }

    /// Start recording draw commands.
    pub fn begin_render_pass(&mut self) -> RenderPass {
        let frame = self.swap_chain.get_next_texture();
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        RenderPass {
            color_needs_clearing: true,
            depth_needs_clearing: true,
            device: &self.device,
            queue: &mut self.queue,
            frame,
            encoder: Some(encoder),
            msaa_attachment: self.msaa_texture_view.as_ref(),
            depth_attachment: &self.depth_texture_view,
            scene_renderer: &self.scene_renderer,
            imgui_renderer: &self.imgui_renderer,
        }
    }
}

/// An ongoing recording of draw commands. Will be submitted on
/// `render_pass.submit()`. Must be submitted before it is dropped.
pub struct RenderPass<'a> {
    color_needs_clearing: bool,
    depth_needs_clearing: bool,
    device: &'a wgpu::Device,
    queue: &'a mut wgpu::Queue,
    frame: wgpu::SwapChainOutput<'a>,
    encoder: Option<wgpu::CommandEncoder>,
    msaa_attachment: Option<&'a wgpu::TextureView>,
    depth_attachment: &'a wgpu::TextureView,
    scene_renderer: &'a SceneRenderer,
    imgui_renderer: &'a ImguiRenderer,
}

impl RenderPass<'_> {
    /// Record a geometry drawing operation to the command
    /// buffer. Geometries with provided ids must be present in the
    /// renderer.
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

    /// Record a UI drawing operation to the command buffer. Textures
    /// referenced by the draw data must be present in the renderer.
    pub fn draw_ui(&mut self, draw_data: &imgui::DrawData) {
        self.imgui_renderer
            .draw_ui(
                self.color_needs_clearing,
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

    /// Submit the built command buffer for drawing.
    pub fn submit(mut self) {
        let encoder = self.encoder.take().expect("Can't finish rendering twice");
        self.queue.submit(&[encoder.finish()]);
    }
}

impl Drop for RenderPass<'_> {
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
    present_mode: PresentMode,
) -> wgpu::SwapChain {
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: SWAP_CHAIN_FORMAT,
            width,
            height,
            present_mode: match present_mode {
                PresentMode::NoVsync => wgpu::PresentMode::NoVsync,
                PresentMode::Vsync => wgpu::PresentMode::Vsync,
            },
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

pub use self::scene_renderer::{
    AddMeshError, DirectionalLight, DrawMeshMode, GpuMesh, GpuMeshHandle,
};

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io;
use std::mem;
use std::thread;

use bitflags::bitflags;
use nalgebra::Matrix4;

use crate::convert::{cast_u32, cast_u64};

use self::common::wgpu_size_of;
use self::imgui_renderer::{ImguiRenderer, Options as ImguiRendererOptions};
use self::scene_renderer::{Options as SceneRendererOptions, SceneRenderer};

#[macro_use]
mod common;

mod imgui_renderer;
mod scene_renderer;

static SHADER_BLIT_VERT: &[u8] = include_shader!("blit.vert.spv");
static SHADER_BLIT_FRAG: &[u8] = include_shader!("blit.frag.spv");

const COLOR_DEBUG_PURPLE: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};

const TEXTURE_FORMAT_SWAP_CHAIN: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const TEXTURE_FORMAT_COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const TEXTURE_FORMAT_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Which multi-sampling setting to use.
    pub msaa: Msaa,
    /// Whether to run with VSync or not.
    pub vsync: bool,
    /// Whether to select an explicit gpu backend for the renderer to use.
    pub backend: Option<GpuBackend>,
    /// Whether to select an explicit power preference profile for the renderer
    /// to use when choosing a GPU.
    pub power_preference: Option<GpuPowerPreference>,
    /// The color with which to render surfaces in `DrawMEshMode::Flat`.
    pub flat_shading_color: [f64; 4],
}

/// Multi-sampling setting. Can be either disabled (1 sample per
/// pixel), or 4/8/16 samples per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Msaa {
    Disabled,
    X2,
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
            Msaa::X2 => 2,
            Msaa::X4 => 4,
            Msaa::X8 => 8,
            Msaa::X16 => 16,
        }
    }
}

impl fmt::Display for Msaa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Msaa::Disabled => write!(f, "Disabled"),
            Msaa::X2 => write!(f, "2x"),
            Msaa::X4 => write!(f, "4x"),
            Msaa::X8 => write!(f, "8x"),
            Msaa::X16 => write!(f, "16x"),
        }
    }
}

/// The rendering backend used by `wgpu-rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Vulkan,
    D3d12,
    Metal,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuBackend::Vulkan => write!(f, "Vulkan"),
            GpuBackend::D3d12 => write!(f, "D3D12"),
            GpuBackend::Metal => write!(f, "Metal"),
        }
    }
}

/// The power preference for selecting a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPowerPreference {
    LowPower,
    HighPerformance,
}

impl fmt::Display for GpuPowerPreference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuPowerPreference::LowPower => write!(f, "Low Power"),
            GpuPowerPreference::HighPerformance => write!(f, "High Performance"),
        }
    }
}

/// Opaque handle to collection of textures for rendering stored in
/// renderer. Does not implement `Clone` on purpose. The handle is
/// acquired by creating the render target and has to be relinquished
/// to destroy it.
#[derive(Debug, PartialEq, Eq)]
pub struct OffscreenRenderTargetHandle(u64);

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
    width: u32,
    height: u32,
    swap_chain: wgpu::SwapChain,
    msaa_texture_view: Option<wgpu::TextureView>,
    color_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    offscreen_render_targets: HashMap<u64, OffscreenRenderTarget>,
    offscreen_render_targets_next_handle: u64,
    blit_pass_bind_group_color: wgpu::BindGroup,
    #[cfg(not(feature = "dist"))]
    blit_pass_bind_group_depth: wgpu::BindGroup,
    blit_texture_bind_group: wgpu::BindGroup,
    blit_render_pipeline: wgpu::RenderPipeline,
    scene_renderer: SceneRenderer,
    imgui_renderer: ImguiRenderer,
    options: Options,
}

impl Renderer {
    pub fn new<H: raw_window_handle::HasRawWindowHandle>(
        window: &H,
        width: u32,
        height: u32,
        imgui_font_atlas: imgui::FontAtlasRefMut,
        options: Options,
    ) -> Self {
        let backends = match options.backend {
            Some(GpuBackend::Vulkan) => wgpu::BackendBit::VULKAN,
            Some(GpuBackend::D3d12) => wgpu::BackendBit::DX12,
            Some(GpuBackend::Metal) => wgpu::BackendBit::METAL,
            None => wgpu::BackendBit::PRIMARY,
        };

        if let Some(backend) = options.backend {
            log::info!("Selected {} GPU backend", backend);
        } else {
            log::info!("No GPU backend selected, will run on default backend");
        }

        let power_preference = match options.power_preference {
            Some(GpuPowerPreference::HighPerformance) => wgpu::PowerPreference::HighPerformance,
            Some(GpuPowerPreference::LowPower) => wgpu::PowerPreference::LowPower,
            None => wgpu::PowerPreference::Default,
        };

        if let Some(power_preference) = options.power_preference {
            log::info!("Selected {} GPU power preference", power_preference);
        } else {
            log::info!("No GPU power preference selected, will select a default GPU");
        }

        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            power_preference,
            backends,
        })
        .expect("Failed to acquire GPU adapter");
        log::debug!("GPU adapter info: {:?}", adapter.get_info());

        let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits {
                // FIXME: @Optimization Use less bind groups if possible.
                max_bind_groups: 6,
            },
        });

        let swap_chain = create_swap_chain(&device, &surface, width, height, options.vsync);
        log::info!("Selected multisampling level: {}", options.msaa);
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

        let color_texture = create_color_texture(&device, width, height);
        let color_texture_view = color_texture.create_default_view();
        let depth_texture =
            create_depth_texture(&device, width, height, options.msaa.sample_count());

        let scene_renderer = SceneRenderer::new(
            &device,
            &mut queue,
            SceneRendererOptions {
                sample_count: options.msaa.sample_count(),
                output_color_attachment_format: TEXTURE_FORMAT_COLOR,
                output_depth_attachment_format: TEXTURE_FORMAT_DEPTH,
                flat_shading_color: options.flat_shading_color,
            },
        );

        let imgui_renderer = ImguiRenderer::new(
            imgui_font_atlas,
            &device,
            &mut queue,
            ImguiRendererOptions {
                output_color_attachment_format: TEXTURE_FORMAT_SWAP_CHAIN,
            },
        )
        .expect("Failed to create imgui renderer");

        let blit_pass_buffer_size = wgpu_size_of::<BlitPassUniforms>();
        let blit_pass_buffer_color = device
            .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM)
            .fill_from_slice(&[BlitPassUniforms {
                blit_sampler: BlitSampler::COLOR,
            }]);

        #[cfg(not(feature = "dist"))]
        let blit_pass_buffer_depth = device
            .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM)
            .fill_from_slice(&[BlitPassUniforms {
                blit_sampler: BlitSampler::DEPTH,
            }]);

        let blit_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                }],
            });

        let blit_pass_bind_group_color = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blit_pass_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &blit_pass_buffer_color,
                    range: 0..blit_pass_buffer_size,
                },
            }],
        });

        #[cfg(not(feature = "dist"))]
        let blit_pass_bind_group_depth = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &blit_pass_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &blit_pass_buffer_depth,
                    range: 0..blit_pass_buffer_size,
                },
            }],
        });

        let blit_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: scene_renderer.sampled_texture_bind_group_layout(),
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_texture_view),
            }],
        });

        let blit_vs_words = wgpu::read_spirv(io::Cursor::new(SHADER_BLIT_VERT))
            .expect("Couldn't read pre-built SPIR-V");
        let blit_fs_words = wgpu::read_spirv(io::Cursor::new(SHADER_BLIT_FRAG))
            .expect("Couldn't read pre-built SPIR-V");
        let blit_vs_module = device.create_shader_module(&blit_vs_words);
        let blit_fs_module = device.create_shader_module(&blit_fs_words);

        let blit_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    &blit_pass_bind_group_layout,
                    scene_renderer.sampler_bind_group_layout(),
                    scene_renderer.sampled_texture_bind_group_layout(),
                ],
            });

        let blit_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &blit_render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &blit_vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &blit_fs_module,
                entry_point: "main",
            }),
            rasterization_state: None,
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: TEXTURE_FORMAT_SWAP_CHAIN,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            device,
            queue,
            surface,
            width,
            height,
            swap_chain,
            msaa_texture_view: msaa_texture.map(|texture| texture.create_default_view()),
            color_texture_view,
            depth_texture_view: depth_texture.create_default_view(),
            offscreen_render_targets: HashMap::new(),
            offscreen_render_targets_next_handle: 0,
            blit_pass_bind_group_color,
            #[cfg(not(feature = "dist"))]
            blit_pass_bind_group_depth,
            blit_texture_bind_group,
            blit_render_pipeline,
            scene_renderer,
            imgui_renderer,
            options,
        }
    }

    /// Update window size. Recreate swap chain, the primary render
    /// target textures, and the bind group responsible for reading
    /// the primary color texture.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        if (width, height) != (self.width, self.height) {
            log::debug!(
                "Resizing renderer screen textures to dimensions: {}x{}",
                width,
                height,
            );

            self.width = width;
            self.height = height;

            self.swap_chain = create_swap_chain(
                &self.device,
                &self.surface,
                width,
                height,
                self.options.vsync,
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

            let color_texture = create_color_texture(&self.device, width, height);
            self.color_texture_view = color_texture.create_default_view();

            let depth_texture = create_depth_texture(
                &self.device,
                width,
                height,
                self.options.msaa.sample_count(),
            );
            self.depth_texture_view = depth_texture.create_default_view();

            // Also need to re-create the bind group for reading the
            // newly created color texture.
            self.blit_texture_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.scene_renderer.sampled_texture_bind_group_layout(),
                    bindings: &[wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.color_texture_view),
                    }],
                });
        }
    }

    // FIXME: This should return result for when the device is out of
    // memory, but wgpu-rs doesn't let us know that yet.

    pub fn add_offscreen_render_target(
        &mut self,
        width: u32,
        height: u32,
    ) -> OffscreenRenderTargetHandle {
        let handle = OffscreenRenderTargetHandle(self.offscreen_render_targets_next_handle);

        // FIXME: Add option to configure different MSAA for offscreen
        // render targets. This will require us to create multiple
        // pipelines, and therefore have a pipeline cache.
        let msaa = &self.options.msaa;

        log::debug!(
            "Adding offscreen render target {} with dimensions {}x{} and multisampling: {}",
            handle.0,
            width,
            height,
            msaa,
        );

        let msaa_texture = if msaa.enabled() {
            Some(create_msaa_texture(
                &self.device,
                width,
                height,
                msaa.sample_count(),
            ))
        } else {
            None
        };

        let color_texture = create_color_texture(&self.device, width, height);
        let color_texture_view = color_texture.create_default_view();
        let depth_texture = create_depth_texture(&self.device, width, height, msaa.sample_count());

        let read_buffer_size = 4 * cast_u64(width) * cast_u64(height);
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            size: read_buffer_size,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
        });

        let offscreen_render_target = OffscreenRenderTarget {
            width,
            height,
            msaa_texture_view: msaa_texture.map(|texture| texture.create_default_view()),
            color_texture,
            color_texture_view,
            depth_texture_view: depth_texture.create_default_view(),
            read_buffer_size,
            read_buffer,
        };

        self.offscreen_render_targets
            .insert(handle.0, offscreen_render_target);
        self.offscreen_render_targets_next_handle += 1;

        handle
    }

    pub fn remove_offscreen_render_target(&mut self, handle: OffscreenRenderTargetHandle) {
        log::debug!("Removing offscreen render target {}", handle.0);
        self.offscreen_render_targets.remove(&handle.0);
    }

    pub fn offscreen_render_target_data<F: FnOnce(u32, u32, &[u8]) + 'static>(
        &mut self,
        handle: &OffscreenRenderTargetHandle,
        callback: F,
    ) {
        let offscreen_render_target = &self.offscreen_render_targets[&handle.0];
        let width = offscreen_render_target.width;
        let height = offscreen_render_target.height;

        // If the render target was rendered to, we already filled
        // this buffer.
        offscreen_render_target.read_buffer.map_read_async(
            0,
            offscreen_render_target.read_buffer_size,
            move |result: wgpu::BufferMapAsyncResult<&[u8]>| {
                let data = result.unwrap().data;
                callback(width, height, data);
            },
        );

        self.device.poll(true);
    }

    /// Uploads mesh to the GPU to be used in scene rendering.
    ///
    /// The mesh will be available for drawing in subsequent render
    /// passes.
    pub fn add_scene_mesh(&mut self, mesh: &GpuMesh) -> Result<GpuMeshHandle, AddMeshError> {
        self.scene_renderer.add_mesh(&self.device, mesh)
    }

    /// Removes mesh from the GPU.
    pub fn remove_scene_mesh(&mut self, handle: GpuMeshHandle) {
        self.scene_renderer.remove_mesh(handle);
    }

    /// Uploads an RGBA8 texture to the GPU to be used in UI
    /// rendering.
    ///
    /// It will be available for drawing in the subsequent render
    /// passes.
    #[allow(dead_code)]
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

    /// Removes texture from the GPU.
    #[allow(dead_code)]
    pub fn remove_ui_texture(&mut self, id: imgui::TextureId) {
        self.imgui_renderer.remove_texture(id);
    }

    /// Starts recording draw commands in an append only log.
    ///
    /// Underlying render passes will use given `clear_color`.
    ///
    /// Everything recorded in a command buffer holds until it is overriden by
    /// later recordings, e.g. `CommandBuffer::set_light` can be called multiple
    /// times, each time setting the light value for all subsequent operations.
    ///
    /// Render target resources, such as the offscreen texture or the shadow map
    /// are cleared exactly once per command buffer.
    pub fn begin_command_buffer(&mut self, clear_color: [f64; 4]) -> CommandBuffer {
        let frame = self.swap_chain.get_next_texture();
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        CommandBuffer {
            backbuffer_needs_clearing: true,
            primary_render_target_needs_clearing: true,
            clear_color,
            device: &self.device,
            queue: &mut self.queue,
            encoder: Some(encoder),
            frame,
            msaa_texture_view: &self.msaa_texture_view,
            color_texture_view: &self.color_texture_view,
            depth_texture_view: &self.depth_texture_view,
            offscreen_render_targets: &self.offscreen_render_targets,
            offscreen_render_targets_cleared: HashSet::new(),
            blit_pass_bind_group_color: &self.blit_pass_bind_group_color,
            #[cfg(not(feature = "dist"))]
            blit_pass_bind_group_depth: &self.blit_pass_bind_group_depth,
            blit_texture_bind_group: &self.blit_texture_bind_group,
            blit_render_pipeline: &self.blit_render_pipeline,
            scene_renderer: &self.scene_renderer,
            imgui_renderer: &self.imgui_renderer,
        }
    }
}

/// An ongoing recording of draw commands. Submit with
/// `CommandBuffer::submit()`. Must be submitted before it is dropped.
pub struct CommandBuffer<'a> {
    backbuffer_needs_clearing: bool,
    primary_render_target_needs_clearing: bool,
    clear_color: [f64; 4],
    device: &'a wgpu::Device,
    queue: &'a mut wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
    frame: wgpu::SwapChainOutput<'a>,
    msaa_texture_view: &'a Option<wgpu::TextureView>,
    color_texture_view: &'a wgpu::TextureView,
    depth_texture_view: &'a wgpu::TextureView,
    offscreen_render_targets: &'a HashMap<u64, OffscreenRenderTarget>,
    offscreen_render_targets_cleared: HashSet<u64>,
    blit_pass_bind_group_color: &'a wgpu::BindGroup,
    #[cfg(not(feature = "dist"))]
    blit_pass_bind_group_depth: &'a wgpu::BindGroup,
    blit_texture_bind_group: &'a wgpu::BindGroup,
    blit_render_pipeline: &'a wgpu::RenderPipeline,
    scene_renderer: &'a SceneRenderer,
    imgui_renderer: &'a ImguiRenderer,
}

impl CommandBuffer<'_> {
    /// Update properties of the shadow casting light.
    pub fn set_light(&mut self, light: &DirectionalLight) {
        self.scene_renderer.set_light(
            &self.device,
            self.encoder.as_mut().expect("Need encoder to update data"),
            light,
        );
    }

    /// Update camera matrices (projection and view).
    pub fn set_camera_matrices(
        &mut self,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) {
        self.scene_renderer.set_camera_matrices(
            &self.device,
            self.encoder.as_mut().expect("Need encoder to upload data"),
            projection_matrix,
            view_matrix,
        );
    }

    /// Record a mesh drawing operation targeting the primary render target to
    /// the command buffer.
    ///
    /// # Warning
    ///
    /// Drawing consists of multiple render passes, some of which compute
    /// metadata (e.g. a shadow map) that is later used in the actual
    /// drawing. Calling this multiple times adds to this metadata every
    /// time. Each subsequent call will have available the information computed
    /// in previous calls, but not vice versa.
    ///
    /// For example, An object rendered in a first call will cast shadows on
    /// objects rendered in subsequent calls (in addition to casting shadows on
    /// objects rendered within the same call), but the shadows won't be present
    /// on objects rendered in prior calls.
    pub fn draw_meshes_to_primary_render_target<'a, H>(
        &mut self,
        mesh_handles: H,
        mode: DrawMeshMode,
        cast_shadows: bool,
    ) where
        H: Iterator<Item = &'a GpuMeshHandle> + Clone,
    {
        self.scene_renderer.draw_meshes(
            mode,
            cast_shadows,
            self.primary_render_target_needs_clearing,
            self.clear_color,
            self.encoder
                .as_mut()
                .expect("Need encoder to record drawing"),
            self.msaa_texture_view.as_ref(),
            &self.color_texture_view,
            &self.depth_texture_view,
            mesh_handles,
        );

        self.primary_render_target_needs_clearing = false;
    }

    /// Record a mesh drawing operation targeting a previously created
    /// offcreen render target to the command buffer.
    ///
    /// # Warning
    ///
    /// Drawing consists of multiple render passes, some of which compute
    /// metadata (e.g. a shadow map) that is later used in the actual
    /// drawing. Calling this multiple times adds to this metadata every
    /// time. Each subsequent call will have available the information computed
    /// in previous calls, but not vice versa.
    ///
    /// For example, An object rendered in a first call will cast shadows on
    /// objects rendered in subsequent calls (in addition to casting shadows on
    /// objects rendered within the same call), but the shadows won't be present
    /// on objects rendered in prior calls.
    pub fn draw_meshes_to_offscreen_render_target<'a, H>(
        &mut self,
        render_target_handle: &OffscreenRenderTargetHandle,
        mesh_handles: H,
        mode: DrawMeshMode,
        cast_shadows: bool,
    ) where
        H: Iterator<Item = &'a GpuMeshHandle> + Clone,
    {
        let offscreen_render_target = &self.offscreen_render_targets[&render_target_handle.0];
        let offscreen_render_target_needs_clearing = self
            .offscreen_render_targets_cleared
            .insert(render_target_handle.0);

        let encoder = self
            .encoder
            .as_mut()
            .expect("Need encoder to record drawing");

        self.scene_renderer.draw_meshes(
            mode,
            cast_shadows,
            offscreen_render_target_needs_clearing,
            self.clear_color,
            encoder,
            offscreen_render_target.msaa_texture_view.as_ref(),
            &offscreen_render_target.color_texture_view,
            &offscreen_render_target.depth_texture_view,
            mesh_handles,
        );

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &offscreen_render_target.color_texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &offscreen_render_target.read_buffer,
                offset: 0,
                row_pitch: cast_u32(mem::size_of::<[u8; 4]>()) * offscreen_render_target.width,
                image_height: offscreen_render_target.height,
            },
            wgpu::Extent3d {
                width: offscreen_render_target.width,
                height: offscreen_render_target.height,
                depth: 1,
            },
        );
    }

    /// Record a UI drawing operation targeting the backbuffer to the
    /// command buffer. Textures referenced by the draw data must be
    /// present in the renderer.
    pub fn draw_ui_to_backbuffer(&mut self, draw_data: &imgui::DrawData) {
        self.imgui_renderer
            .draw_ui(
                self.backbuffer_needs_clearing,
                self.clear_color,
                self.device,
                self.encoder
                    .as_mut()
                    .expect("Need encoder to record drawing"),
                &self.frame.view,
                draw_data,
            )
            .expect("Imgui drawing failed");

        self.backbuffer_needs_clearing = false;
    }

    /// Record a copy operation from the primary render target to the
    /// backbuffer.
    pub fn blit_primary_render_target_to_backbuffer(&mut self) {
        let encoder = self
            .encoder
            .as_mut()
            .expect("Need encoder to record drawing");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &self.frame.view,
                resolve_target: None,
                load_op: if self.backbuffer_needs_clearing {
                    wgpu::LoadOp::Clear
                } else {
                    wgpu::LoadOp::Load
                },
                store_op: wgpu::StoreOp::Store,
                // If we see this color, something has gone wrong :)
                clear_color: COLOR_DEBUG_PURPLE,
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(self.blit_render_pipeline);
        rpass.set_bind_group(0, &self.blit_pass_bind_group_color, &[]);
        rpass.set_bind_group(1, self.scene_renderer.sampler_bind_group(), &[]);
        rpass.set_bind_group(2, self.blit_texture_bind_group, &[]);
        rpass.draw(0..3, 0..1);

        self.backbuffer_needs_clearing = false;
    }

    /// Record a copy operation from the shadow map to the backbuffer. Use for
    /// debugging.
    #[cfg(not(feature = "dist"))]
    pub fn blit_shadow_map_to_backbuffer(&mut self) {
        let encoder = self
            .encoder
            .as_mut()
            .expect("Need encoder to record drawing");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &self.frame.view,
                resolve_target: None,
                load_op: if self.backbuffer_needs_clearing {
                    wgpu::LoadOp::Clear
                } else {
                    wgpu::LoadOp::Load
                },
                store_op: wgpu::StoreOp::Store,
                // If we see this color, something has gone wrong :)
                clear_color: COLOR_DEBUG_PURPLE,
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(self.blit_render_pipeline);
        rpass.set_bind_group(0, &self.blit_pass_bind_group_depth, &[]);
        rpass.set_bind_group(1, self.scene_renderer.sampler_bind_group(), &[]);
        rpass.set_bind_group(2, self.scene_renderer.shadow_map_texture_bind_group(), &[]);
        rpass.draw(0..3, 0..1);

        self.backbuffer_needs_clearing = false;
    }

    /// Submit the built command buffer for drawing.
    pub fn submit(mut self) {
        let encoder = self.encoder.take().expect("Can't finish rendering twice");
        self.queue.submit(&[encoder.finish()]);
    }
}

impl Drop for CommandBuffer<'_> {
    fn drop(&mut self) {
        if !thread::panicking() {
            assert!(
                self.encoder.is_none(),
                "Rendering must be finished by the time the command buffer goes out of scope",
            );
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct BlitPassUniforms {
    blit_sampler: BlitSampler,
}

bitflags! {
    struct BlitSampler: u32 {
        const COLOR = 0x01;
        const DEPTH = 0x02;
    }
}

struct OffscreenRenderTarget {
    width: u32,
    height: u32,
    msaa_texture_view: Option<wgpu::TextureView>,
    color_texture: wgpu::Texture,
    color_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    read_buffer_size: u64,
    read_buffer: wgpu::Buffer,
}

fn create_swap_chain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    width: u32,
    height: u32,
    vsync: bool,
) -> wgpu::SwapChain {
    log::debug!(
        "Creating swapchain with dimensions {}x{} and vsync: {}",
        width,
        height,
        vsync,
    );

    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: TEXTURE_FORMAT_SWAP_CHAIN,
            width,
            height,
            present_mode: if vsync {
                wgpu::PresentMode::Vsync
            } else {
                wgpu::PresentMode::NoVsync
            },
        },
    )
}

fn create_color_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    log::debug!(
        "Creating color texture with format {:?} and dimensions: {}x{}",
        TEXTURE_FORMAT_COLOR,
        width,
        height,
    );

    device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT_COLOR,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
            | wgpu::TextureUsage::SAMPLED
            | wgpu::TextureUsage::COPY_SRC,
    })
}

fn create_msaa_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::Texture {
    log::debug!(
        "Creating msaa texture with format {:?}, sample count {} and dimensions: {}x{}",
        TEXTURE_FORMAT_COLOR,
        sample_count,
        width,
        height,
    );

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
        format: TEXTURE_FORMAT_COLOR,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    })
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::Texture {
    log::debug!(
        "Creating depth texture with format {:?}, sample count {} and dimensions: {}x{}",
        TEXTURE_FORMAT_DEPTH,
        sample_count,
        width,
        height,
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
        format: TEXTURE_FORMAT_DEPTH,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    })
}

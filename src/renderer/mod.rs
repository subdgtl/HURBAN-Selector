pub use self::scene_renderer::{AddMeshError, DirectionalLight, GpuMesh, GpuMeshHandle, Material};

use std::collections::HashMap;
use std::fmt;
use std::iter;
use std::mem;
use std::ops::Deref;
use std::slice;
use std::thread;

use nalgebra::Matrix4;

use crate::convert::{cast_u32, cast_u64};

use self::imgui_renderer::{ImguiRenderer, Options as ImguiRendererOptions};
use self::scene_renderer::{Options as SceneRendererOptions, SceneRenderer};

#[macro_use]
mod common;

mod imgui_renderer;
mod scene_renderer;

#[cfg(target_os = "windows")]
static DEFAULT_BACKEND_LIST: &[GpuBackend] = &[GpuBackend::Vulkan, GpuBackend::D3d12];
#[cfg(target_os = "macos")]
static DEFAULT_BACKEND_LIST: &[GpuBackend] = &[GpuBackend::Metal];
#[cfg(target_os = "linux")]
static DEFAULT_BACKEND_LIST: &[GpuBackend] = &[GpuBackend::Vulkan];

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
    /// GPU backend to use for rendering.
    ///
    /// If not chosen, the renderer will pick a default based on the current
    /// platform.
    pub backend: Option<GpuBackend>,
    /// Power preference for selecting a GPU.
    pub power_preference: GpuPowerPreference,
    /// Level of multi-sampling based anti-aliasing to use in rendering.
    pub msaa: Msaa,
    /// The color with which to render surfaces in `Material::FlatWithShadows`.
    pub flat_material_color: [f64; 4],
    /// The transparency value of transparent matcap materials.
    pub transparent_matcap_shaded_material_alpha: f64,
}

/// Level of multi-sampling based anti-aliasing to use in rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Clap)]
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

/// GPU backend to use for rendering.
///
/// If not chosen, the renderer will pick a default based on the current
/// platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Clap)]
pub enum GpuBackend {
    Vulkan,
    D3d12,
    Metal,
}

impl Into<wgpu::BackendBit> for GpuBackend {
    fn into(self) -> wgpu::BackendBit {
        match self {
            GpuBackend::Vulkan => wgpu::BackendBit::VULKAN,
            GpuBackend::D3d12 => wgpu::BackendBit::DX12,
            GpuBackend::Metal => wgpu::BackendBit::METAL,
        }
    }
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

/// Power preference for selecting a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Clap)]
pub enum GpuPowerPreference {
    /// Let the implementation choose a GPU device based on battery state and
    /// power availability.
    Default,
    /// Try to use a GPU device with least power consumption.
    LowPower,
    /// Try to use a GPU device with most performance.
    HighPerformance,
}

impl Into<wgpu::PowerPreference> for GpuPowerPreference {
    fn into(self) -> wgpu::PowerPreference {
        match self {
            GpuPowerPreference::Default => wgpu::PowerPreference::Default,
            GpuPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            GpuPowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        }
    }
}

impl fmt::Display for GpuPowerPreference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuPowerPreference::Default => write!(f, "Default"),
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

pub struct OffscreenRenderTargetReadMapping<'a> {
    width: u32,
    height: u32,
    read_buffer_slice: wgpu::BufferSlice<'a>,
}

impl<'a> OffscreenRenderTargetReadMapping<'a> {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn data(&self) -> impl Deref<Target = [u8]> + 'a {
        self.read_buffer_slice.get_mapped_range()
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
    screen_render_target: RenderTarget,
    offscreen_read_buffer: Option<(u64, wgpu::Buffer)>,
    offscreen_render_targets: HashMap<u64, RenderTarget>,
    offscreen_render_targets_next_handle: u64,
    blit_pass_bind_group_color: wgpu::BindGroup,
    #[cfg(not(feature = "dist"))]
    blit_pass_bind_group_depth: wgpu::BindGroup,
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
        let gpu_backend_list = options
            .backend
            .as_ref()
            .map(|backend| slice::from_ref(backend))
            .unwrap_or(DEFAULT_BACKEND_LIST);

        let gpu_power_preference = options.power_preference;
        log::info!("GPU will use power preference: {}", gpu_power_preference);

        let mut gpu_backend_iter = gpu_backend_list.iter().copied();
        let (surface, adapter) = loop {
            if let Some(gpu_backend) = gpu_backend_iter.next() {
                log::info!("Trying to acquire GPU adapter for backend: {}", gpu_backend);

                let instance = wgpu::Instance::new(gpu_backend.into());
                let surface = unsafe { instance.create_surface(window) };
                let adapter_result = futures::executor::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: gpu_power_preference.into(),
                        compatible_surface: Some(&surface),
                    },
                ));

                match adapter_result {
                    Some(adapter) => {
                        log::info!("Found suitable GPU adapter for backend: {}", gpu_backend);
                        break (surface, adapter);
                    }
                    None => {
                        log::warn!("Failed to acquire GPU adapter for backend: {}", gpu_backend);
                    }
                }
            } else {
                panic!("Failed to find suitable GPU backend");
            }
        };

        log::info!("GPU adapter info: {:?}", adapter.get_info());

        let (device, mut queue) = futures::executor::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits {
                    // FIXME: @Optimization Use less bind groups if possible.
                    max_bind_groups: 6,
                    ..wgpu::Limits::default()
                },
                shader_validation: true,
            },
            None,
        ))
        .expect("Failed to request GPU device");

        let swap_chain = create_swap_chain(&device, &surface, width, height);

        log::info!("GPU will use multisampling level: {}", options.msaa);
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
        let color_texture_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture =
            create_depth_texture(&device, width, height, options.msaa.sample_count());

        let scene_renderer = SceneRenderer::new(
            &device,
            &mut queue,
            SceneRendererOptions {
                sample_count: options.msaa.sample_count(),
                output_color_attachment_format: TEXTURE_FORMAT_COLOR,
                output_depth_attachment_format: TEXTURE_FORMAT_DEPTH,
                flat_material_color: options.flat_material_color,
                transparent_matcap_shaded_material_alpha: options
                    .transparent_matcap_shaded_material_alpha,
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

        let blit_pass_buffer_color = common::create_buffer(
            &device,
            wgpu::BufferUsage::UNIFORM,
            &[BlitPassUniforms {
                blit_sampler: BlitSampler::Color,
            }],
        );

        #[cfg(not(feature = "dist"))]
        let blit_pass_buffer_depth = common::create_buffer(
            &device,
            wgpu::BufferUsage::UNIFORM,
            &[BlitPassUniforms {
                blit_sampler: BlitSampler::Depth,
            }],
        );

        let blit_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        // FIXME: @Optimization Provide this for runtime speedup
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let blit_pass_bind_group_color = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &blit_pass_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(blit_pass_buffer_color.slice(..)),
            }],
        });

        #[cfg(not(feature = "dist"))]
        let blit_pass_bind_group_depth = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &blit_pass_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(blit_pass_buffer_depth.slice(..)),
            }],
        });

        let color_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: scene_renderer.sampled_texture_bind_group_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_texture_view),
            }],
        });

        let blit_vs_source = wgpu::util::make_spirv(SHADER_BLIT_VERT);
        let blit_fs_source = wgpu::util::make_spirv(SHADER_BLIT_FRAG);
        let blit_vs_module = device.create_shader_module(blit_vs_source);
        let blit_fs_module = device.create_shader_module(blit_fs_source);

        let blit_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &blit_pass_bind_group_layout,
                    scene_renderer.sampler_bind_group_layout(),
                    scene_renderer.sampled_texture_bind_group_layout(),
                ],
                push_constant_ranges: &[],
            });

        let blit_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&blit_render_pipeline_layout),
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
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            device,
            queue,
            surface,
            swap_chain,
            screen_render_target: RenderTarget {
                width,
                height,
                msaa_texture_view: msaa_texture
                    .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default())),
                color_texture,
                color_texture_view,
                color_texture_bind_group,
                depth_texture_view: depth_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            },
            offscreen_read_buffer: None,
            offscreen_render_targets: HashMap::new(),
            offscreen_render_targets_next_handle: 0,
            blit_pass_bind_group_color,
            #[cfg(not(feature = "dist"))]
            blit_pass_bind_group_depth,
            blit_render_pipeline,
            scene_renderer,
            imgui_renderer,
            options,
        }
    }

    /// Update window size. Recreate swap chain, the screen render target
    /// textures, and the bind group responsible for reading the color texture.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        if (width, height)
            != (
                self.screen_render_target.width,
                self.screen_render_target.height,
            )
        {
            log::debug!(
                "Resizing renderer screen textures to dimensions: {}x{}",
                width,
                height,
            );

            self.screen_render_target.width = width;
            self.screen_render_target.height = height;

            self.swap_chain = create_swap_chain(&self.device, &self.surface, width, height);

            if self.options.msaa.enabled() {
                let msaa_texture = create_msaa_texture(
                    &self.device,
                    width,
                    height,
                    self.options.msaa.sample_count(),
                );
                self.screen_render_target.msaa_texture_view =
                    Some(msaa_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            }

            let color_texture = create_color_texture(&self.device, width, height);
            self.screen_render_target.color_texture_view =
                color_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_texture = create_depth_texture(
                &self.device,
                width,
                height,
                self.options.msaa.sample_count(),
            );
            self.screen_render_target.depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Also need to re-create the bind group for reading the
            // newly created color texture.
            self.screen_render_target.color_texture_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.scene_renderer.sampled_texture_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.screen_render_target.color_texture_view,
                        ),
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
        let color_texture_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture = create_depth_texture(&self.device, width, height, msaa.sample_count());

        let color_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.scene_renderer.sampled_texture_bind_group_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_texture_view),
            }],
        });

        let offscreen_render_target = RenderTarget {
            width,
            height,
            msaa_texture_view: msaa_texture
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default())),
            color_texture,
            color_texture_view,
            color_texture_bind_group,
            depth_texture_view: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
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

    pub fn offscreen_render_target_data<'a>(
        &'a mut self,
        handle: &OffscreenRenderTargetHandle,
    ) -> OffscreenRenderTargetReadMapping<'a> {
        let offscreen_render_target = &self.offscreen_render_targets[&handle.0];
        let width = offscreen_render_target.width;
        let height = offscreen_render_target.height;

        let read_buffer_required_size = 4 * cast_u64(width) * cast_u64(height);
        if let Some((read_buffer_current_size, _)) = &self.offscreen_read_buffer {
            if read_buffer_required_size > *read_buffer_current_size {
                let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: read_buffer_required_size,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                    mapped_at_creation: false,
                });

                self.offscreen_read_buffer = Some((read_buffer_required_size, read_buffer));
            }
        } else {
            let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: read_buffer_required_size,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                mapped_at_creation: false,
            });

            self.offscreen_read_buffer = Some((read_buffer_required_size, read_buffer));
        }

        let read_buffer = &self.offscreen_read_buffer.as_ref().unwrap().1;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &offscreen_render_target.color_texture,

                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &read_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    // TODO(yanchith): Verify that this satisfies
                    // wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
                    bytes_per_row: cast_u32(mem::size_of::<[u8; 4]>()) * width,
                    rows_per_image: height,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
        );

        self.queue.submit(iter::once(encoder.finish()));

        // Read from the buffer we just copied to. Immediately after we request
        // the mapping we wait on the device to provide it and block.

        // FIXME: @Optimization Make this async - let's not wait, let's poll
        let read_buffer_slice = read_buffer.slice(..);
        let future = read_buffer_slice.map_async(wgpu::MapMode::Read);

        self.device.poll(wgpu::Maintain::Wait);
        futures::executor::block_on(future).expect("Failed to map buffer");

        // TODO(yanchith): We need to unmap the buffer! Not sure if we can do it
        // without combining ref counting with Drop impl on
        // OffscreenRenderTargetReadMapping.
        OffscreenRenderTargetReadMapping {
            width,
            height,
            read_buffer_slice,
        }
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
    /// Underlying render passes will use given `clear_color`. If
    /// `offscreen_render_target_handle` is given, it will be used instead of
    /// the screen render target. If swap chain texture is requested, it can be
    /// rendered into.
    ///
    /// Everything recorded in a command buffer holds until it is overriden by
    /// later recordings, e.g. `CommandBuffer::set_light` can be called multiple
    /// times, each time setting the light value for all subsequent operations.
    ///
    /// Render target resources, such as the offscreen texture or the shadow map
    /// are cleared exactly once per command buffer.
    pub fn begin_command_buffer(
        &mut self,
        clear_color: [f64; 4],
        offscreen_render_target_handle: Option<&OffscreenRenderTargetHandle>,
        request_swap_chain_texture: bool,
    ) -> CommandBuffer {
        let render_target = if let Some(handle) = offscreen_render_target_handle {
            &self.offscreen_render_targets[&handle.0]
        } else {
            &self.screen_render_target
        };

        let frame = if request_swap_chain_texture {
            match self.swap_chain.get_current_frame() {
                Ok(frame) => Some(frame),
                Err(err) => match err {
                    wgpu::SwapChainError::Timeout | wgpu::SwapChainError::Outdated => {
                        log::warn!("GPU swapchain error: {}", err);
                        None
                    }
                    wgpu::SwapChainError::Lost | wgpu::SwapChainError::OutOfMemory => {
                        // FIXME: @Correctness Try recovering at least for
                        // wgpu::SwapChainError::Lost
                        log::error!("Serious GPU swapchain error: {}", err);
                        panic!("Encountered GPU swapchain error: {}", err);
                    }
                },
            }
        } else {
            None
        };

        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        CommandBuffer {
            swap_chain_needs_clearing: true,
            render_target_needs_clearing: true,
            clear_color,
            device: &self.device,
            queue: &mut self.queue,
            encoder: Some(encoder),
            frame,
            render_target,
            blit_pass_bind_group_color: &self.blit_pass_bind_group_color,
            #[cfg(not(feature = "dist"))]
            blit_pass_bind_group_depth: &self.blit_pass_bind_group_depth,
            blit_render_pipeline: &self.blit_render_pipeline,
            scene_renderer: &mut self.scene_renderer,
            imgui_renderer: &mut self.imgui_renderer,
        }
    }
}

/// An ongoing recording of draw commands. Submit with
/// `CommandBuffer::submit()`. Must be submitted before it is dropped.
pub struct CommandBuffer<'a> {
    swap_chain_needs_clearing: bool,
    render_target_needs_clearing: bool,
    clear_color: [f64; 4],
    device: &'a wgpu::Device,
    queue: &'a mut wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
    frame: Option<wgpu::SwapChainFrame>,
    render_target: &'a RenderTarget,
    blit_pass_bind_group_color: &'a wgpu::BindGroup,
    #[cfg(not(feature = "dist"))]
    blit_pass_bind_group_depth: &'a wgpu::BindGroup,
    blit_render_pipeline: &'a wgpu::RenderPipeline,
    scene_renderer: &'a mut SceneRenderer,
    imgui_renderer: &'a mut ImguiRenderer,
}

impl CommandBuffer<'_> {
    /// Update properties of the shadow casting light.
    pub fn set_light(&mut self, light: &DirectionalLight) {
        self.scene_renderer.set_light(self.queue, light);
    }

    /// Update camera matrices (projection and view).
    pub fn set_camera_matrices(
        &mut self,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) {
        self.scene_renderer
            .set_camera_matrices(self.queue, projection_matrix, view_matrix);
    }

    /// Record a mesh drawing operation targeting the render target to the
    /// command buffer.
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
    pub fn draw_meshes_to_render_target<'a, P>(&mut self, mesh_props: P)
    where
        P: Iterator<Item = (&'a GpuMeshHandle, Material, bool)> + Clone,
    {
        self.scene_renderer.draw_meshes(
            self.render_target_needs_clearing,
            self.clear_color,
            self.encoder
                .as_mut()
                .expect("Need encoder to record drawing"),
            self.render_target.msaa_texture_view.as_ref(),
            &self.render_target.color_texture_view,
            &self.render_target.depth_texture_view,
            mesh_props,
        );

        self.render_target_needs_clearing = false;
    }

    /// Record a UI drawing operation targeting the swap chain to the
    /// command buffer. Textures referenced by the draw data must be
    /// present in the renderer.
    pub fn draw_ui_to_swap_chain(&mut self, draw_data: &imgui::DrawData) {
        if let Some(frame) = &self.frame {
            self.imgui_renderer
                .draw_ui(
                    self.swap_chain_needs_clearing,
                    self.clear_color,
                    self.device,
                    self.queue,
                    self.encoder
                        .as_mut()
                        .expect("Need encoder to record drawing"),
                    &frame.output.view,
                    draw_data,
                )
                .expect("Imgui drawing failed");

            self.swap_chain_needs_clearing = false;
        } else {
            log::warn!("Can not draw to absent swap chain texture");
        }
    }

    /// Record a copy operation from the render target to the swap chain.
    pub fn blit_render_target_to_swap_chain(&mut self) {
        if let Some(frame) = &self.frame {
            let encoder = self
                .encoder
                .as_mut()
                .expect("Need encoder to record drawing");

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if self.swap_chain_needs_clearing {
                            wgpu::LoadOp::Clear(COLOR_DEBUG_PURPLE)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(self.blit_render_pipeline);
            rpass.set_bind_group(0, &self.blit_pass_bind_group_color, &[]);
            rpass.set_bind_group(1, self.scene_renderer.sampler_bind_group(), &[]);
            rpass.set_bind_group(2, &self.render_target.color_texture_bind_group, &[]);
            rpass.draw(0..3, 0..1);

            self.swap_chain_needs_clearing = false;
        } else {
            log::warn!("Can not draw to absent swap chain texture");
        }
    }

    /// Record a copy operation from the shadow map to the swap chain. Use for
    /// debugging.
    #[cfg(not(feature = "dist"))]
    pub fn blit_shadow_map_to_swap_chain(&mut self) {
        if let Some(frame) = &self.frame {
            let encoder = self
                .encoder
                .as_mut()
                .expect("Need encoder to record drawing");

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if self.swap_chain_needs_clearing {
                            wgpu::LoadOp::Clear(COLOR_DEBUG_PURPLE)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(self.blit_render_pipeline);
            rpass.set_bind_group(0, &self.blit_pass_bind_group_depth, &[]);
            rpass.set_bind_group(1, self.scene_renderer.sampler_bind_group(), &[]);
            rpass.set_bind_group(2, self.scene_renderer.shadow_map_texture_bind_group(), &[]);
            rpass.draw(0..3, 0..1);

            self.swap_chain_needs_clearing = false;
        } else {
            log::warn!("Can not draw to absent swap chain texture");
        }
    }

    /// Submit the built command buffer for drawing.
    pub fn submit(mut self) {
        let encoder = self.encoder.take().expect("Can't finish rendering twice");
        self.queue.submit(iter::once(encoder.finish()));
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
#[derive(Debug, Clone, Copy, PartialEq, zerocopy::AsBytes)]
struct BlitPassUniforms {
    blit_sampler: BlitSampler,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, zerocopy::AsBytes)]
enum BlitSampler {
    Color = 0,
    #[cfg(not(feature = "dist"))]
    Depth = 1,
}

struct RenderTarget {
    width: u32,
    height: u32,
    msaa_texture_view: Option<wgpu::TextureView>,
    color_texture: wgpu::Texture,
    color_texture_view: wgpu::TextureView,
    color_texture_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
}

fn create_swap_chain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    width: u32,
    height: u32,
) -> wgpu::SwapChain {
    log::debug!("Creating swapchain with dimensions {}x{}", width, height);
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: TEXTURE_FORMAT_SWAP_CHAIN,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
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
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
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
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
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
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT_DEPTH,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    })
}

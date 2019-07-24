use std::collections::HashMap;
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::mem;

use log;
use nalgebra::base::Matrix4;
use wgpu;
use wgpu::winit::dpi::PhysicalSize;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::D32Float;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
}

// FIXME(yanchith): @Optimization Determine u16/u32 dynamically per
// geometry to save memory
pub type Index = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeometryId(u64);

#[derive(Debug)]
pub enum ViewportRendererAddGeometryError {
    TooManyVertices(u32, usize),
    TooManyIndices(u32, usize),
}

impl fmt::Display for ViewportRendererAddGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ViewportRendererAddGeometryError::*;
        match self {
            TooManyVertices(max, given) => write!(
                f,
                "Geometry contains too many vertices {}. (max allowed is {})",
                given, max,
            ),
            TooManyIndices(max, given) => write!(
                f,
                "Geometry contains too many indices: {}. (max allowed is {})",
                given, max,
            ),
        }
    }
}

impl error::Error for ViewportRendererAddGeometryError {}

pub struct ViewportRenderer {
    geometries: HashMap<u64, GeometryDescriptor>,
    geometries_next_id: u64,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: wgpu::TextureView,
    uniform_matrix_buffer: wgpu::Buffer,
    uniform_matrix_bind_group: wgpu::BindGroup,
    view_matrix: Matrix4<f32>,
    aspect_ratio: f32,
}

impl ViewportRenderer {
    pub fn new(
        device: &mut wgpu::Device,
        screen_size: PhysicalSize,
        output_format: wgpu::TextureFormat,
        view_matrix: Matrix4<f32>,
    ) -> Self {
        let vs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/shaded.vert.spv"));
        let fs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/shaded.frag.spv"));
        let vs_module = device.create_shader_module(vs_bytes);
        let fs_module = device.create_shader_module(fs_bytes);

        let uniform_matrix_buffer_size = mem::size_of::<Matrix4<f32>>() as u64;
        let uniform_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: uniform_matrix_buffer_size,
            // TRANSFER_DST is here because uploading data to this
            // buffer is not done via MAP_WRITE, but rather creating
            // another mapped buffer, and issuing a transfer command.
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
        });

        let uniform_matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer,
                }],
            });
        let uniform_matrix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_matrix_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_matrix_buffer,
                    range: 0..uniform_matrix_buffer_size,
                },
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_matrix_bind_group_layout],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::PipelineStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::PipelineStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            },
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: output_format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    format: wgpu::VertexFormat::Float3,
                    shader_location: 0,
                }],
            }],
            sample_count: 1,
        });

        let PhysicalSize { width, height } = screen_size;
        let aspect_ratio = width as f32 / height as f32;
        let (tex_width, tex_height) = (width as u32, height as u32);

        let matrix = Self::create_matrix(aspect_ratio, &view_matrix);
        Self::upload_uniform_matrix_buffer(device, &uniform_matrix_buffer, &matrix);
        let depth_texture = Self::create_depth_texture(device, tex_width, tex_height);

        Self {
            geometries: HashMap::new(),
            geometries_next_id: 0,
            render_pipeline,
            depth_texture: depth_texture.create_default_view(),
            uniform_matrix_buffer,
            uniform_matrix_bind_group,
            view_matrix,
            aspect_ratio,
        }
    }

    pub fn set_view_matrix(&mut self, device: &mut wgpu::Device, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;

        let matrix = Self::create_matrix(self.aspect_ratio, &view_matrix);
        Self::upload_uniform_matrix_buffer(device, &self.uniform_matrix_buffer, &matrix);
    }

    pub fn set_screen_size(&mut self, device: &mut wgpu::Device, screen_size: PhysicalSize) {
        let PhysicalSize { width, height } = screen_size;
        let (tex_width, tex_height) = (width as u32, height as u32);
        self.aspect_ratio = width as f32 / height as f32;

        let matrix = Self::create_matrix(self.aspect_ratio, &self.view_matrix);
        Self::upload_uniform_matrix_buffer(device, &self.uniform_matrix_buffer, &matrix);

        let depth_texture = Self::create_depth_texture(device, tex_width, tex_height);
        self.depth_texture = depth_texture.create_default_view();
    }

    pub fn add_geometry(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
    ) -> Result<GeometryId, ViewportRendererAddGeometryError> {
        let id = GeometryId(self.geometries_next_id);
        let vertex_count = u32::try_from(vertices.len()).map_err(|_| {
            ViewportRendererAddGeometryError::TooManyVertices(u32::max_value(), vertices.len())
        })?;

        log::debug!("Adding geometry {} with {} vertices", id.0, vertex_count);

        let vertex_buffer = device
            .create_buffer_mapped(vertices.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(vertices);

        self.geometries.insert(
            id.0,
            GeometryDescriptor {
                vertices: (vertex_buffer, vertex_count),
                indices: None,
            },
        );

        self.geometries_next_id += 1;
        Ok(id)
    }

    pub fn add_geometry_indexed(
        &mut self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[Index],
    ) -> Result<GeometryId, ViewportRendererAddGeometryError> {
        let id = GeometryId(self.geometries_next_id);
        let vertex_count = u32::try_from(vertices.len()).map_err(|_| {
            ViewportRendererAddGeometryError::TooManyVertices(u32::max_value(), vertices.len())
        })?;
        let index_count = u32::try_from(indices.len()).map_err(|_| {
            ViewportRendererAddGeometryError::TooManyIndices(u32::max_value(), indices.len())
        })?;

        log::debug!(
            "Adding indexed geometry {} with {} vertices and {} indices",
            id.0,
            vertex_count,
            index_count,
        );

        let vertex_buffer = device
            .create_buffer_mapped(vertices.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(vertices);

        let index_buffer = device
            .create_buffer_mapped(indices.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(indices);

        self.geometries.insert(
            id.0,
            GeometryDescriptor {
                vertices: (vertex_buffer, vertex_count),
                indices: Some((index_buffer, index_count)),
            },
        );

        self.geometries_next_id += 1;
        Ok(id)
    }

    pub fn remove_geometry(&mut self, id: GeometryId) {
        log::debug!("Removing geometry with {}", id.0);
        // Dropping the geometry descriptor here unstreams the buffers from device memory
        self.geometries.remove(&id.0);
    }

    pub fn draw_geometry(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_attachment: &wgpu::TextureView,
        ids: &[GeometryId],
    ) {
        // FIXME(yanchith): Try a more high level renderer API that
        // would wrap both the viewport and gui renderers so that we
        // don't have to pass &Device and &mut Encoder
        // everywhere. This main renderer could delegeta to the
        // various subrenderers as it sees fit.
        /*

        let mut renderer = ...;

        loop {
            let mut render_pass = renderer.begin_render_pass();

            render_pass.draw_grid(...);
            render_pass.draw_geometry(...);
            render_pass.draw_geometry_wireframe(...);
            render_pass.draw_geometry_highlight(...);
            render_pass_draw_ui(...);

            render_pass.finish()?; // Consumes. Drop panicks unless this is explicitely called.
        }

        */

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: target_attachment,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: &self.depth_texture,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
        });

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.uniform_matrix_bind_group, &[]);

        for id in ids {
            if let Some(geometry) = &self.geometries.get(&id.0) {
                let (vertex_buffer, vertex_count) = &geometry.vertices;
                rpass.set_vertex_buffers(&[(vertex_buffer, 0)]);
                if let Some((index_buffer, index_count)) = &geometry.indices {
                    rpass.set_index_buffer(&index_buffer, 0);
                    rpass.draw_indexed(0..*index_count, 0, 0..1);
                } else {
                    rpass.draw(0..*vertex_count, 0..1);
                }
            } else {
                log::warn!("Geometry with id {} does not exist in this renderer.", id.0);
            }
        }
    }

    fn create_matrix(aspect_ratio: f32, view_matrix: &Matrix4<f32>) -> Matrix4<f32> {
        const FOVY: f32 = std::f32::consts::FRAC_PI_3;
        const ZNEAR: f32 = 0.01;
        const ZFAR: f32 = 1000.0;

        // Vulkan (and therefore wgpu) has different NDC and
        // clip-space semantics than OpenGL: Vulkan is right-handed, Y
        // grows downwards. The easiest way to keep everything working
        // as before and use all the libraries that assume OpenGL is
        // to apply a correction to the projection matrix which
        // normally changes the right-handed OpenGL world-space to
        // left-handed OpenGL clip-space (apart from actually
        // performing the projection).
        // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
        #[rustfmt::skip]
        let wgpu_correction_matrix = Matrix4::new(
            1.0,  0.0,  0.0,  0.0,
            0.0, -1.0,  0.0,  0.0,
            0.0,  0.0,  0.5,  0.0,
            0.0,  0.0,  0.5,  1.0,
        );
        let projection_matrix = Matrix4::new_perspective(aspect_ratio, FOVY, ZNEAR, ZFAR);

        wgpu_correction_matrix * projection_matrix * view_matrix
    }

    fn upload_uniform_matrix_buffer(
        device: &mut wgpu::Device,
        uniform_buffer: &wgpu::Buffer,
        matrix: &Matrix4<f32>,
    ) {
        let uniform_count = 16;
        let uniform_buffer_size = mem::size_of::<Matrix4<f32>>() as u64;

        let tmp_buffer = device
            .create_buffer_mapped(uniform_count, wgpu::BufferUsage::TRANSFER_SRC)
            .fill_from_slice(matrix.as_slice());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(&tmp_buffer, 0, uniform_buffer, 0, uniform_buffer_size);
        device.get_queue().submit(&[encoder.finish()]);
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
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
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        })
    }
}

struct GeometryDescriptor {
    vertices: (wgpu::Buffer, u32),
    indices: Option<(wgpu::Buffer, u32)>,
}

use std::collections::HashMap;
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::io;
use std::mem;

use nalgebra::base::Matrix4;

use wgpu::winit;

macro_rules! include_shader {
    ($name:expr) => {{
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    }};
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::D32Float;

const SHADER_MATCAP_VERT: &[u8] = include_shader!("matcap.vert.spv");
const SHADER_MATCAP_FRAG: &[u8] = include_shader!("matcap.frag.spv");

const MATCAP_TEXTURE_BYTES: &[u8] = include_bytes!("../resources/matcap.png");

/// The geometry vertex data as uploaded on the GPU.
///
/// Positions and normals are internally `[f32; 4]` with the last
/// component filled in as 0.0 or 1.0 depending on it being a position
/// or vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    // These are defined as [f32; 4] for 2 reasons:
    //
    // - point vs vector clarity
    // - padding: vec3 has the same size as vec4 in std140 layout.
    //   https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)
    //
    // We might decide to pack more useful data than just 0/1 in the
    // last component later.
    pub position: [f32; 4],
    pub normal: [f32; 4],
}

// FIXME: @Optimization Determine u16/u32 dynamically per geometry to
// save memory
/// The geometry indices as uploaded on the GPU.
pub type Index = u32;

/// The geometry containing index and vertex data in the format as
/// will be uploaded on the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    indices: Option<Vec<Index>>,
    vertex_data: Vec<Vertex>,
}

impl Geometry {
    /// Create geometry from vectors of positions and normals of same
    /// length. Does not run any validations except for length
    /// checking.
    pub fn from_positions_and_normals(
        vertex_positions: Vec<[f32; 3]>,
        vertex_normals: Vec<[f32; 3]>,
    ) -> Self {
        assert!(
            !vertex_positions.is_empty(),
            "Vertex positions must not be empty",
        );
        assert!(
            !vertex_normals.is_empty(),
            "Vertex normals must not be empty",
        );
        assert_eq!(
            vertex_positions.len(),
            vertex_normals.len(),
            "Per-vertex data must be same length",
        );

        let vertex_data = vertex_positions
            .into_iter()
            .zip(vertex_normals.into_iter())
            .map(Self::vertex)
            .collect();

        Self {
            vertex_data,
            indices: None,
        }
    }

    /// Create indexed geometry from vectors of positions and normals
    /// of same length. Does not run any validations except for length
    /// checking.
    pub fn from_positions_and_normals_indexed(
        indices: Vec<Index>,
        vertex_positions: Vec<[f32; 3]>,
        vertex_normals: Vec<[f32; 3]>,
    ) -> Self {
        assert!(!indices.is_empty(), "Indices must not be empty");
        assert!(
            !vertex_positions.is_empty(),
            "Vertex positions must not be empty",
        );
        assert!(
            !vertex_normals.is_empty(),
            "Vertex normals must not be empty",
        );
        assert_eq!(
            vertex_positions.len(),
            vertex_normals.len(),
            "Per-vertex data must be same length",
        );

        let vertex_data = vertex_positions
            .into_iter()
            .zip(vertex_normals.into_iter())
            .map(Self::vertex)
            .collect();

        Self {
            vertex_data,
            indices: Some(indices),
        }
    }

    pub fn vertex_data(&self) -> &[Vertex] {
        &self.vertex_data
    }

    pub fn indices(&self) -> Option<&[Index]> {
        if let Some(indices) = &self.indices {
            Some(&indices)
        } else {
            None
        }
    }

    fn vertex((position, normal): ([f32; 3], [f32; 3])) -> Vertex {
        Vertex {
            position: [position[0], position[1], position[2], 1.0],
            normal: [normal[0], normal[1], normal[2], 0.0],
        }
    }
}

/// Opaque handle to geometry stored in viewport renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeometryId(u64);

#[derive(Debug)]
pub enum ViewportRendererAddGeometryError {
    TooManyVertices(usize),
    TooManyIndices(usize),
}

impl fmt::Display for ViewportRendererAddGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ViewportRendererAddGeometryError as AddGeometryError;
        match self {
            AddGeometryError::TooManyVertices(given) => write!(
                f,
                "Geometry contains too many vertices {}. (max allowed is {})",
                given,
                u32::max_value(),
            ),
            AddGeometryError::TooManyIndices(given) => write!(
                f,
                "Geometry contains too many indices: {}. (max allowed is {})",
                given,
                u32::max_value(),
            ),
        }
    }
}

impl error::Error for ViewportRendererAddGeometryError {}

/// 3D renderer of the editor viewport.
///
/// Can be used to upload `Geometry` on the GPU and draw it in the
/// viewport. Currently supports just matcap rendering.
pub struct ViewportRenderer {
    geometries: HashMap<u64, GeometryDescriptor>,
    geometries_next_id: u64,
    global_matrix_buffer: wgpu::Buffer,
    global_matrix_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    matcap_texture_bind_group: wgpu::BindGroup,
    matcap_render_pipeline: wgpu::RenderPipeline,
}

impl ViewportRenderer {
    /// Create a new viewport renderer.
    ///
    /// Initializes GPU resources and the rendering pipeline to draw
    /// to a texture of `output_format`. `screen_size` and the camera
    /// matrices are the initial states of the screen and camera, and
    /// can be updated with setters later.
    pub fn new(
        device: &mut wgpu::Device,
        output_format: wgpu::TextureFormat,
        screen_size: winit::dpi::PhysicalSize,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) -> Self {
        let vs_module = device.create_shader_module(SHADER_MATCAP_VERT);
        let fs_module = device.create_shader_module(SHADER_MATCAP_FRAG);

        let global_matrix_buffer_size = wgpu_size_of::<GlobalMatrixUniforms>();
        let global_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: global_matrix_buffer_size,
            // TRANSFER_DST is here because uploading data to this
            // buffer is not done via MAP_WRITE, but rather creating
            // another mapped buffer, and issuing a transfer command.
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
        });

        let global_matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer,
                }],
            });
        let global_matrix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_matrix_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &global_matrix_buffer,
                    range: 0..global_matrix_buffer_size,
                },
            }],
        });

        let (matcap_texture_width, matcap_texture_height, matcap_texture_data) = {
            let cursor = io::Cursor::new(MATCAP_TEXTURE_BYTES);
            let decoder = png::Decoder::new(cursor);
            let (info, mut reader) = decoder
                .read_info()
                .expect("Baked matcap texture decoding must succeed");

            let mut buffer = vec![0; info.buffer_size()];
            reader
                .next_frame(&mut buffer)
                .expect("Baked matcap texture decoding must succeed");

            assert_eq!(
                info.color_type,
                png::ColorType::RGBA,
                "Baked matcap texture must be RGBA",
            );

            (info.width, info.height, buffer)
        };

        let matcap_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: matcap_texture_width,
                height: matcap_texture_height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::TRANSFER_DST,
        });
        let matcap_texture_view = matcap_texture.create_default_view();

        upload_texture(
            device,
            &matcap_texture,
            matcap_texture_width,
            matcap_texture_height,
            &matcap_texture_data,
        );

        let matcap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });

        let matcap_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture,
                    },
                    wgpu::BindGroupLayoutBinding {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler,
                    },
                ],
            });

        let matcap_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &matcap_texture_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&matcap_texture_view),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&matcap_sampler),
                },
            ],
        });

        let matcap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    &global_matrix_bind_group_layout,
                    &matcap_texture_bind_group_layout,
                ],
            });

        let matcap_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &matcap_pipeline_layout,
                vertex_stage: wgpu::PipelineStageDescriptor {
                    module: &vs_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::PipelineStageDescriptor {
                    module: &fs_module,
                    entry_point: "main",
                }),
                rasterization_state: wgpu::RasterizationStateDescriptor {
                    // We don't cull faces - even inner walls of 3d
                    // models should be visible. That does not mean
                    // the renderer is ok with non-CCW geometries. We
                    // might implement a special rendering pass for
                    // back faces one day.
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::None,
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
                    stride: wgpu_size_of::<Vertex>(),
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            offset: 0,
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            offset: wgpu_size_of::<[f32; 4]>(), // 4 bytes * 4 components
                            format: wgpu::VertexFormat::Float4,
                            shader_location: 1,
                        },
                    ],
                }],
                sample_count: 1,
            });

        let winit::dpi::PhysicalSize { width, height } = screen_size;
        let (tex_width, tex_height) = (width as u32, height as u32);

        let global_matrix_uniforms = GlobalMatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };
        upload_global_matrix_buffer(device, &global_matrix_buffer, global_matrix_uniforms);
        let depth_texture = create_depth_texture(device, tex_width, tex_height);

        Self {
            geometries: HashMap::new(),
            geometries_next_id: 0,
            global_matrix_buffer,
            global_matrix_bind_group,
            depth_texture: depth_texture.create_default_view(),
            matcap_texture_bind_group,
            matcap_render_pipeline,
        }
    }

    /// Update the camera matrix (a composition of projection matrix
    /// and view matrix).
    pub fn set_camera_matrices(
        &mut self,
        device: &mut wgpu::Device,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) {
        let global_matrix_uniforms = GlobalMatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };
        upload_global_matrix_buffer(device, &self.global_matrix_buffer, global_matrix_uniforms);
    }

    /// Update the screen size.
    ///
    /// Recreates depth texture and recomputes the scene's
    /// transformation matrix as necessary.
    pub fn set_screen_size(
        &mut self,
        device: &mut wgpu::Device,
        screen_size: winit::dpi::PhysicalSize,
    ) {
        let winit::dpi::PhysicalSize { width, height } = screen_size;
        let (tex_width, tex_height) = (width as u32, height as u32);

        let depth_texture = create_depth_texture(device, tex_width, tex_height);
        self.depth_texture = depth_texture.create_default_view();
    }

    /// Upload geometry on the GPU.
    ///
    /// Whether indexed or not, the data must be in the
    /// `TRIANGLE_LIST` format. The returned id can be used to draw
    /// the geometry, or remove it.
    pub fn add_geometry(
        &mut self,
        device: &wgpu::Device,
        geometry: &Geometry,
    ) -> Result<GeometryId, ViewportRendererAddGeometryError> {
        use ViewportRendererAddGeometryError as AddGeometryError;

        let id = GeometryId(self.geometries_next_id);

        let vertex_data = geometry.vertex_data();
        let vertex_data_count = u32::try_from(vertex_data.len())
            .map_err(|_| AddGeometryError::TooManyVertices(vertex_data.len()))?;

        let geometry_descriptor = if let Some(indices) = geometry.indices() {
            let index_count = u32::try_from(indices.len())
                .map_err(|_| AddGeometryError::TooManyIndices(indices.len()))?;

            log::debug!(
                "Adding geometry {} with {} vertices and {} indices",
                id.0,
                vertex_data_count,
                index_count,
            );

            let vertex_buffer = device
                .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
                .fill_from_slice(vertex_data);

            let index_buffer = device
                .create_buffer_mapped(indices.len(), wgpu::BufferUsage::INDEX)
                .fill_from_slice(indices);

            GeometryDescriptor {
                vertices: (vertex_buffer, vertex_data_count),
                indices: Some((index_buffer, index_count)),
            }
        } else {
            log::debug!(
                "Adding geometry {} with {} vertices",
                id.0,
                vertex_data_count
            );

            let vertex_buffer = device
                .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
                .fill_from_slice(vertex_data);

            GeometryDescriptor {
                vertices: (vertex_buffer, vertex_data_count),
                indices: None,
            }
        };

        self.geometries.insert(id.0, geometry_descriptor);
        self.geometries_next_id += 1;
        Ok(id)
    }

    /// Remove a previously uploaded geometry from the GPU.
    pub fn remove_geometry(&mut self, id: GeometryId) {
        log::debug!("Removing geometry with {}", id.0);
        // Dropping the geometry descriptor here unstreams the buffers from device memory
        self.geometries.remove(&id.0);
    }

    /// Draw previously uploaded geometries as one of the commands
    /// executed with the `command_encoder` to the
    /// `target_attachment`.
    pub fn draw_geometry(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_attachment: &wgpu::TextureView,
        ids: &[GeometryId],
    ) {
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

        rpass.set_pipeline(&self.matcap_render_pipeline);
        rpass.set_bind_group(0, &self.global_matrix_bind_group, &[]);
        rpass.set_bind_group(1, &self.matcap_texture_bind_group, &[]);

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
}

struct GeometryDescriptor {
    vertices: (wgpu::Buffer, u32),
    indices: Option<(wgpu::Buffer, u32)>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct GlobalMatrixUniforms {
    projection_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
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

fn upload_global_matrix_buffer(
    device: &mut wgpu::Device,
    global_matrix_buffer: &wgpu::Buffer,
    global_matrix_uniforms: GlobalMatrixUniforms,
) {
    let global_matrix_uniforms_size = wgpu_size_of::<GlobalMatrixUniforms>();

    let tmp_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::TRANSFER_SRC)
        .fill_from_slice(&[global_matrix_uniforms]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(
        &tmp_buffer,
        0,
        global_matrix_buffer,
        0,
        global_matrix_uniforms_size,
    );
    device.get_queue().submit(&[encoder.finish()]);
}

fn upload_texture(
    device: &mut wgpu::Device,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    data: &[u8],
) {
    let buffer = device
        .create_buffer_mapped(data.len(), wgpu::BufferUsage::TRANSFER_SRC)
        .fill_from_slice(data);

    let byte_count = u32::try_from(data.len()).expect("Texture byte length must fit in u32");
    let pixel_size = byte_count / width / height;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &buffer,
            offset: 0,
            row_pitch: pixel_size * width,
            image_height: height,
        },
        wgpu::TextureCopyView {
            texture,
            mip_level: 0,
            array_layer: 0,
            origin: wgpu::Origin3d {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
    );

    device.get_queue().submit(&[encoder.finish()]);
}

/// Applies vulkan/wgpu correction matrix to the projection matrix.
fn apply_wgpu_correction_matrix(projection_matrix: &Matrix4<f32>) -> Matrix4<f32> {
    // Vulkan (and therefore wgpu) has different NDC and
    // clip-space semantics than OpenGL: Vulkan is right-handed, Y
    // grows downwards. The easiest way to keep everything working
    // as before and use all the libraries that assume OpenGL is
    // to apply a correction to the projection matrix which
    // normally changes the right-handed OpenGL world-space to
    // left-handed OpenGL clip-space.
    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    #[rustfmt::skip]
    let wgpu_correction_matrix = Matrix4::new(
        1.0,  0.0,  0.0,  0.0,
        0.0, -1.0,  0.0,  0.0,
        0.0,  0.0,  0.5,  0.0,
        0.0,  0.0,  0.5,  1.0,
    );

    wgpu_correction_matrix * projection_matrix
}

fn wgpu_size_of<T>() -> wgpu::BufferAddress {
    let size = mem::size_of::<T>();
    wgpu::BufferAddress::try_from(size)
        .unwrap_or_else(|_| panic!("Size {} does not fit into wgpu BufferAddress", size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        #[rustfmt::skip]
        let vertex_positions = vec![
            [ -0.3, -0.5,  0.0 ],
            [  0.3, -0.5,  0.0 ],
            [  0.0,  0.5,  0.0 ],
        ];

        #[rustfmt::skip]
        let vertex_normals = vec![
            [ 0.0, 0.0, 1.0 ],
            [ 0.0, 0.0, 1.0 ],
            [ 0.0, 0.0, 1.0 ],
        ];

        (vertex_positions, vertex_normals)
    }

    fn triangle_indexed() -> (Vec<Index>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let (vertex_positions, vertex_normals) = triangle();
        let indices = vec![0, 1, 2];

        (indices, vertex_positions, vertex_normals)
    }

    #[test]
    fn test_create_valid_geometry_does_not_panic() {
        let (positions, normals) = triangle();
        let geometry = Geometry::from_positions_and_normals(positions, normals);

        assert_eq!(geometry.vertex_data, vec![
            Vertex {
                position: [-0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
        ]);
        assert_eq!(geometry.indices, None);
    }

    #[test]
    fn test_create_valid_indexed_geometry_does_not_panic() {
        let (indices, positions, normals) = triangle_indexed();
        let geometry = Geometry::from_positions_and_normals_indexed(indices, positions, normals);

        assert_eq!(geometry.vertex_data, vec![
            Vertex {
                position: [-0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
            },
        ]);
        assert_eq!(geometry.indices, Some(vec![0, 1, 2]));
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_create_geometry_from_different_length_vertex_data_panicks() {
        let (_, normals) = triangle();
        let _geometry = Geometry::from_positions_and_normals(vec![[1.0, 1.0, 1.0]], normals);
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_create_indexed_geometry_from_different_length_vertex_data_panicks() {
        let (indices, positions, _) = triangle_indexed();
        let _geometry =
            Geometry::from_positions_and_normals_indexed(indices, positions, vec![[1.0, 1.0, 1.0]]);
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_create_geometry_from_empty_vertex_positions_panicks() {
        let (_, normals) = triangle();
        let _geometry = Geometry::from_positions_and_normals(vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_create_geometry_from_empty_vertex_normals_panicks() {
        let (positions, _) = triangle();
        let _geometry = Geometry::from_positions_and_normals(positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_create_indexed_geometry_from_empty_vertex_positions_panicks() {
        let (indices, _, normals) = triangle_indexed();
        let _geometry = Geometry::from_positions_and_normals_indexed(indices, vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_create_indexed_geometry_from_empty_vertex_normals_panicks() {
        let (indices, positions, _) = triangle_indexed();
        let _geometry = Geometry::from_positions_and_normals_indexed(indices, positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Indices must not be empty")]
    fn test_create_indexed_geometry_from_empty_indices_panicks() {
        let (_, vertices, normals) = triangle_indexed();
        let _geometry = Geometry::from_positions_and_normals_indexed(vec![], vertices, normals);
    }
}

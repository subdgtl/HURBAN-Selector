use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::io;

use bitflags::bitflags;
use nalgebra::base::{Matrix4, Vector3};
use nalgebra::geometry::Point3;

use crate::convert::cast_usize;
use crate::geometry::Geometry;

use super::common::{upload_texture_rgba8_unorm, wgpu_size_of};

const SHADER_MATCAP_VERT: &[u8] = include_shader!("matcap.vert.spv");
const SHADER_MATCAP_FRAG: &[u8] = include_shader!("matcap.frag.spv");

const MATCAP_TEXTURE_BYTES: &[u8] = include_bytes!("../../resources/matcap.png");

/// The geometry containing index and vertex data in same-length
/// format as will be uploaded on the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneRendererGeometry {
    indices: Option<Vec<GeometryIndex>>,
    vertex_data: Vec<GeometryVertex>,
}

impl SceneRendererGeometry {
    /// Construct geometry with same-length per-vertex data from
    /// variable-length data `Geometry`.
    ///
    /// Duplicates vertices if same vertex is encountered multiple
    /// times paired with different per-vertex data, e.g. normals.
    pub fn from_geometry(geometry: &Geometry) -> Self {
        let vertices = geometry.vertices();
        if let Some(normals) = geometry.normals() {
            let faces_len = geometry.triangle_faces_len();
            let indices_len_estimate = faces_len * 3;

            let mut indices = Vec::with_capacity(indices_len_estimate);

            // This capacity is a lower bound estimate. Given that
            // `Geometry` contains no orphan vertices, there should be
            // at least `vertices.len()` vertices present in the
            // resulting `SceneRendererGeometry`.
            let mut vertex_data = Vec::with_capacity(vertices.len());

            // This capacity is an upper bound estimate and will
            // overshoot if indices are re-used
            let mut index_map = HashMap::with_capacity(indices_len_estimate);
            let mut next_renderer_index = 0;

            // Iterate over all faces, creating or re-using vertices
            // as we go. Vertex data identity is defined by equality
            // of the index that constructed the vertex.
            for triangle_face in geometry.triangle_faces_iter() {
                let v = triangle_face.vertices;
                let n = triangle_face
                    .normals
                    .expect("Normal indices must be present if normals are");

                for &(vi, ni) in &[(v.0, n.0), (v.1, n.1), (v.2, n.2)] {
                    match index_map.entry((vi, ni)) {
                        Entry::Occupied(occupied) => {
                            // This concrete vertex/normal combination
                            // was used before, re-use the vertex it
                            // created
                            let renderer_index = *occupied.get();

                            indices.push(renderer_index);
                        }
                        Entry::Vacant(vacant) => {
                            // We didn't see this vertex/normal
                            // combination before, we need to create a
                            // new vertex and remember the index we
                            // assigned
                            let renderer_index = next_renderer_index;
                            let position = vertices[cast_usize(vi)];
                            let normal = normals[cast_usize(ni)];
                            let vertex = Self::vertex((position, normal));

                            vacant.insert(renderer_index);
                            next_renderer_index += 1;

                            vertex_data.push(vertex);
                            indices.push(renderer_index)
                        }
                    };
                }
            }

            assert_eq!(
                indices.capacity(),
                indices_len_estimate,
                "Number of indices does not match estimate"
            );

            vertex_data.shrink_to_fit();

            SceneRendererGeometry {
                indices: Some(indices),
                vertex_data,
            }
        } else {
            // FIXME: add ability to compute normals on demand if not
            // present here

            unimplemented!("Renderer geometry needs normals")
        }
    }

    /// Create geometry from vectors of positions and normals of same
    /// length. Does not run any validations except for length
    /// checking.
    #[allow(dead_code)]
    pub fn from_positions_and_normals(
        vertex_positions: Vec<Point3<f32>>,
        vertex_normals: Vec<Vector3<f32>>,
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
            indices: None,
            vertex_data,
        }
    }

    /// Create indexed geometry from vectors of positions and normals
    /// of same length. Does not run any validations except for length
    /// checking.
    #[allow(dead_code)]
    pub fn from_positions_and_normals_indexed(
        indices: Vec<GeometryIndex>,
        vertex_positions: Vec<Point3<f32>>,
        vertex_normals: Vec<Vector3<f32>>,
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
            indices: Some(indices),
            vertex_data,
        }
    }

    fn vertex((position, normal): (Point3<f32>, Vector3<f32>)) -> GeometryVertex {
        GeometryVertex {
            position: [position[0], position[1], position[2], 1.0],
            normal: [normal[0], normal[1], normal[2], 0.0],
        }
    }
}

/// Opaque handle to geometry stored in scene renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneRendererGeometryId(u64);

#[derive(Debug)]
pub enum SceneRendererAddGeometryError {
    TooManyVertices(usize),
    TooManyIndices(usize),
}

impl fmt::Display for SceneRendererAddGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use SceneRendererAddGeometryError as AddGeometryError;
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

impl error::Error for SceneRendererAddGeometryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneRendererOptions {
    pub sample_count: u32,
    pub output_color_attachment_format: wgpu::TextureFormat,
    pub output_depth_attachment_format: wgpu::TextureFormat,
}

bitflags! {
    pub struct SceneRendererClearFlags: u8 {
        const COLOR = 0b_0000_0001;
        const DEPTH = 0b_0000_0010;
    }
}

/// 3D renderer of the editor scene.
///
/// Can be used to upload `Geometry` on the GPU and draw it in the
/// viewport. Currently supports just matcap rendering.
pub struct SceneRenderer {
    geometry_resources: HashMap<u64, GeometryResource>,
    geometry_resources_next_id: u64,
    global_matrix_buffer: wgpu::Buffer,
    global_matrix_bind_group: wgpu::BindGroup,
    matcap_texture_bind_group: wgpu::BindGroup,
    matcap_render_pipeline: wgpu::RenderPipeline,
}

impl SceneRenderer {
    /// Create a new scene renderer.
    ///
    /// Initializes GPU resources and the rendering pipeline to draw
    /// to a texture of `output_color_attachment_format`.
    pub fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
        options: SceneRendererOptions,
    ) -> Self {
        let vs_words = wgpu::read_spirv(io::Cursor::new(SHADER_MATCAP_VERT))
            .expect("Couldn't read pre-built SPIR-V");
        let fs_words = wgpu::read_spirv(io::Cursor::new(SHADER_MATCAP_FRAG))
            .expect("Couldn't read pre-built SPIR-V");
        let vs_module = device.create_shader_module(&vs_words);
        let fs_module = device.create_shader_module(&fs_words);

        let global_matrix_buffer_size = wgpu_size_of::<GlobalMatrixUniforms>();
        let global_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: global_matrix_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let global_matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
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
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        let matcap_texture_view = matcap_texture.create_default_view();

        upload_texture_rgba8_unorm(
            device,
            queue,
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
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                        },
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
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_module,
                    entry_point: "main",
                }),
                // Default rasterization state means
                // CullMode::None. We don't cull faces yet, because we
                // work with potentially non-CCW geometries. We might
                // implement special rendering for CW faces one day.
                rasterization_state: None,
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[wgpu::ColorStateDescriptor {
                    format: options.output_color_attachment_format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                }],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: options.output_depth_attachment_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                }),
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: wgpu_size_of::<GeometryVertex>(),
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
                sample_count: options.sample_count,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        let global_matrix_uniforms = GlobalMatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };
        upload_global_matrix_buffer(device, queue, &global_matrix_buffer, global_matrix_uniforms);

        Self {
            geometry_resources: HashMap::new(),
            geometry_resources_next_id: 0,
            global_matrix_buffer,
            global_matrix_bind_group,
            matcap_texture_bind_group,
            matcap_render_pipeline,
        }
    }

    /// Update camera matrices (projection matrix and view matrix).
    pub fn set_camera_matrices(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        projection_matrix: &Matrix4<f32>,
        view_matrix: &Matrix4<f32>,
    ) {
        let global_matrix_uniforms = GlobalMatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };
        upload_global_matrix_buffer(
            device,
            queue,
            &self.global_matrix_buffer,
            global_matrix_uniforms,
        );
    }

    /// Upload geometry on the GPU.
    ///
    /// Whether indexed or not, the data must be in the
    /// `TRIANGLE_LIST` format. The returned id can be used to draw
    /// the geometry, or remove it.
    pub fn add_geometry(
        &mut self,
        device: &wgpu::Device,
        geometry: &SceneRendererGeometry,
    ) -> Result<SceneRendererGeometryId, SceneRendererAddGeometryError> {
        use SceneRendererAddGeometryError as AddGeometryError;

        let id = SceneRendererGeometryId(self.geometry_resources_next_id);

        let vertex_data = &geometry.vertex_data[..];
        let vertex_data_count = u32::try_from(vertex_data.len())
            .map_err(|_| AddGeometryError::TooManyVertices(vertex_data.len()))?;

        let geometry_descriptor = if let Some(indices) = &geometry.indices {
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

            GeometryResource {
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

            GeometryResource {
                vertices: (vertex_buffer, vertex_data_count),
                indices: None,
            }
        };

        self.geometry_resources.insert(id.0, geometry_descriptor);
        self.geometry_resources_next_id += 1;
        Ok(id)
    }

    /// Remove a previously uploaded geometry from the GPU.
    pub fn remove_geometry(&mut self, id: SceneRendererGeometryId) {
        log::debug!("Removing geometry with {}", id.0);
        // Dropping the geometry descriptor here unstreams the buffers from device memory
        self.geometry_resources.remove(&id.0);
    }

    /// Optionally clear color and depth and draw previously uploaded
    /// geometries as one of the commands executed with the `encoder`
    /// to the `color_attachment`.
    pub fn draw_geometry(
        &self,
        clear_flags: SceneRendererClearFlags,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        msaa_attachment: Option<&wgpu::TextureView>,
        depth_attachment: &wgpu::TextureView,
        ids: &[SceneRendererGeometryId],
    ) {
        let color_load_op = if clear_flags.contains(SceneRendererClearFlags::COLOR) {
            wgpu::LoadOp::Clear
        } else {
            wgpu::LoadOp::Load
        };

        let depth_load_op = if clear_flags.contains(SceneRendererClearFlags::DEPTH) {
            wgpu::LoadOp::Clear
        } else {
            wgpu::LoadOp::Load
        };

        let rpass_color_attachment_descriptor = if let Some(msaa_attachment) = msaa_attachment {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: msaa_attachment,
                resolve_target: Some(color_attachment),
                load_op: color_load_op,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            }
        } else {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: color_attachment,
                resolve_target: None,
                load_op: color_load_op,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[rpass_color_attachment_descriptor],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: depth_attachment,
                depth_load_op,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
        });

        // This needs to be set for vulkan, oterwise the validation
        // layers complain about the stencil reference not being
        // set... Not sure if this is a bug or not.
        rpass.set_stencil_reference(0);

        rpass.set_pipeline(&self.matcap_render_pipeline);

        rpass.set_bind_group(0, &self.global_matrix_bind_group, &[]);
        rpass.set_bind_group(1, &self.matcap_texture_bind_group, &[]);

        for id in ids {
            if let Some(geometry) = &self.geometry_resources.get(&id.0) {
                let (vertex_buffer, vertex_count) = &geometry.vertices;
                rpass.set_vertex_buffers(0, &[(vertex_buffer, 0)]);
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

struct GeometryResource {
    vertices: (wgpu::Buffer, u32),
    indices: Option<(wgpu::Buffer, u32)>,
}

/// The geometry vertex data as uploaded on the GPU.
///
/// Positions and normals are internally `[f32; 4]` with the last
/// component filled in as 0.0 or 1.0 depending on it being a position
/// or vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct GeometryVertex {
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
type GeometryIndex = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct GlobalMatrixUniforms {
    projection_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
}

fn upload_global_matrix_buffer(
    device: &wgpu::Device,
    queue: &mut wgpu::Queue,
    global_matrix_buffer: &wgpu::Buffer,
    global_matrix_uniforms: GlobalMatrixUniforms,
) {
    let global_matrix_uniforms_size = wgpu_size_of::<GlobalMatrixUniforms>();

    let transfer_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&[global_matrix_uniforms]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(
        &transfer_buffer,
        0,
        global_matrix_buffer,
        0,
        global_matrix_uniforms_size,
    );

    queue.submit(&[encoder.finish()]);
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

#[cfg(test)]
mod tests {
    use crate::geometry::TriangleFace;

    use super::*;

    fn triangle() -> (Vec<Point3<f32>>, Vec<Vector3<f32>>) {
        #[rustfmt::skip]
        let vertex_positions = vec![
            Point3::new(-0.3, -0.5,  0.0),
            Point3::new( 0.3, -0.5,  0.0),
            Point3::new( 0.0,  0.5,  0.0),
        ];

        #[rustfmt::skip]
        let vertex_normals = vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        (vertex_positions, vertex_normals)
    }

    fn triangle_indexed() -> (Vec<GeometryIndex>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
        let (vertex_positions, vertex_normals) = triangle();
        let indices = vec![0, 1, 2];

        (indices, vertex_positions, vertex_normals)
    }

    fn triangle_geometry_same_len() -> Geometry {
        #[rustfmt::skip]
        let positions = vec![
            Point3::new(-0.3, -0.5,  0.0),
            Point3::new( 0.3, -0.5,  0.0),
            Point3::new( 0.0,  0.5,  0.0),
        ];

        #[rustfmt::skip]
        let normals = vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        #[rustfmt::skip]
        let faces = vec![
            TriangleFace { vertices: (0, 1, 2), normals: Some((0, 1, 2)) }
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, positions, normals)
    }

    fn triangle_geometry_var_len() -> Geometry {
        #[rustfmt::skip]
        let positions = vec![
            Point3::new(-0.3, -0.5,  0.0),
            Point3::new( 0.3, -0.5,  0.0),
            Point3::new( 0.0,  0.5,  0.0),
        ];

        #[rustfmt::skip]
        let normals = vec![
            Vector3::new(0.0, 0.0, 1.0),
        ];

        #[rustfmt::skip]
        let faces = vec![
            TriangleFace { vertices: (0, 1, 2), normals: Some((0, 0, 0)) }
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, positions, normals)
    }

    #[test]
    fn test_renderer_geometry_from_positions_and_normals() {
        let (positions, normals) = triangle();
        let geometry = SceneRendererGeometry::from_positions_and_normals(positions, normals);

        assert_eq!(
            geometry.vertex_data,
            vec![
                GeometryVertex {
                    position: [-0.3, -0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                },
                GeometryVertex {
                    position: [0.3, -0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                },
                GeometryVertex {
                    position: [0.0, 0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                },
            ]
        );
        assert_eq!(geometry.indices, None);
    }

    #[test]
    fn test_renderer_geometry_from_positions_and_normals_indexed() {
        let (indices, positions, normals) = triangle_indexed();
        let geometry =
            SceneRendererGeometry::from_positions_and_normals_indexed(indices, positions, normals);

        #[rustfmt::skip]
        let expected_vertex_data = vec![
            GeometryVertex { position: [-0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.0,  0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
        ];

        assert_eq!(geometry.vertex_data, expected_vertex_data);
        assert_eq!(geometry.indices, Some(vec![0, 1, 2]));
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_renderer_geometry_from_positions_and_normals_panics_on_different_length_data() {
        let (_, normals) = triangle();
        let _geometry = SceneRendererGeometry::from_positions_and_normals(
            vec![Point3::new(1.0, 1.0, 1.0)],
            normals,
        );
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_renderer_geometry_from_positions_and_normals_indexed_panics_on_different_length_data() {
        let (indices, positions, _) = triangle_indexed();
        let _geometry = SceneRendererGeometry::from_positions_and_normals_indexed(
            indices,
            positions,
            vec![Vector3::new(1.0, 1.0, 1.0)],
        );
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_renderer_geometry_from_positions_and_normals_panics_on_empty_positions() {
        let (_, normals) = triangle();
        let _geometry = SceneRendererGeometry::from_positions_and_normals(vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_renderer_geometry_from_positions_and_normals_indexed_panics_on_empty_positions() {
        let (positions, _) = triangle();
        let _geometry = SceneRendererGeometry::from_positions_and_normals(positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_renderer_geometry_from_positions_and_normals_panics_on_empty_normals() {
        let (indices, _, normals) = triangle_indexed();
        let _geometry =
            SceneRendererGeometry::from_positions_and_normals_indexed(indices, vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_renderer_geometry_from_positions_and_normals_indexed_panics_on_empty_normals() {
        let (indices, positions, _) = triangle_indexed();
        let _geometry =
            SceneRendererGeometry::from_positions_and_normals_indexed(indices, positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Indices must not be empty")]
    fn test_renderer_geometry_from_positions_and_normals_indexed_panics_on_empty_indices() {
        let (_, vertices, normals) = triangle_indexed();
        let _geometry =
            SceneRendererGeometry::from_positions_and_normals_indexed(vec![], vertices, normals);
    }

    #[test]
    fn test_renderer_geometry_from_geometry_preserves_already_same_len_geometry() {
        let geometry = SceneRendererGeometry::from_geometry(&triangle_geometry_same_len());

        #[rustfmt::skip]
        let expected_vertex_data = vec![
            GeometryVertex { position: [-0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.0,  0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
        ];

        assert_eq!(geometry.vertex_data, expected_vertex_data);
        assert_eq!(geometry.indices, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_renderer_geometry_from_geometry_duplicates_normals_in_var_len_geometry() {
        let geometry = SceneRendererGeometry::from_geometry(&triangle_geometry_var_len());

        #[rustfmt::skip]
        let expected_vertex_data = vec![
            GeometryVertex { position: [-0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.3, -0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
            GeometryVertex { position: [ 0.0,  0.5,  0.0,  1.0], normal: [ 0.0,  0.0,  1.0,  0.0] },
        ];

        assert_eq!(geometry.vertex_data, expected_vertex_data);
        assert_eq!(geometry.indices, Some(vec![0, 1, 2]));
    }
}

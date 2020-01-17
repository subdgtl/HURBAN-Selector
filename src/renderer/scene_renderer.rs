use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::io;
use std::iter;

use bitflags::bitflags;
use nalgebra::{Matrix4, Point3, Vector3};

use crate::convert::cast_usize;
use crate::mesh::{Face, Mesh};

use super::common::{upload_texture_rgba8_unorm, wgpu_size_of};

static SHADER_VIEWPORT_VERT: &[u8] = include_shader!("viewport.vert.spv");
static SHADER_VIEWPORT_FRAG: &[u8] = include_shader!("viewport.frag.spv");

static MATCAP_TEXTURE_BYTES: &[u8] = include_bytes!("../../resources/matcap.png");

/// The mesh containing index and vertex data in same-length
/// format as will be uploaded on the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuMesh {
    indices: Option<Vec<GpuMeshIndex>>,
    vertex_data: Vec<GpuMeshVertex>,
}

impl GpuMesh {
    /// Creates mesh with same-length per-vertex data from
    /// variable-length data `Mesh`.
    ///
    /// Duplicates vertices if same vertex is encountered multiple
    /// times paired with different per-vertex data, e.g. normals.
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let vertices = mesh.vertices();
        let normals = mesh.normals();

        let faces_len = mesh.faces().len();
        let indices_len_estimate = faces_len * 3;

        let mut indices = Vec::with_capacity(indices_len_estimate);

        // This capacity is a lower bound estimate. Given that
        // `Mesh` contains no orphan vertices, there should be
        // at least `vertices.len()` vertices present in the
        // resulting `GpuMesh`.
        let mut vertex_data = Vec::with_capacity(vertices.len());

        // This capacity is an upper bound estimate and will
        // overshoot if indices are re-used
        let mut index_map = HashMap::with_capacity(indices_len_estimate);
        let mut next_renderer_index = 0;

        // Iterate over all faces, creating or re-using vertices
        // as we go. Vertex data identity is defined by equality
        // of the index that constructed the vertex.
        for face in mesh.faces() {
            match face {
                Face::Triangle(triangle_face) => {
                    let v = triangle_face.vertices;
                    let n = triangle_face.normals;

                    for &(vertex_index, normal_index, barycentric) in
                        &[(v.0, n.0, 0x01), (v.1, n.1, 0x02), (v.2, n.2, 0x04)]
                    {
                        match index_map.entry((vertex_index, normal_index, barycentric)) {
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
                                let position = vertices[cast_usize(vertex_index)];
                                let normal = normals[cast_usize(normal_index)];
                                let vertex = Self::vertex(position, normal, barycentric);

                                vacant.insert(renderer_index);
                                next_renderer_index += 1;

                                vertex_data.push(vertex);
                                indices.push(renderer_index)
                            }
                        };
                    }
                }
            }
        }

        assert_eq!(
            indices.capacity(),
            indices_len_estimate,
            "Number of indices does not match estimate"
        );

        vertex_data.shrink_to_fit();

        GpuMesh {
            indices: Some(indices),
            vertex_data,
        }
    }

    /// Creates mesh from vectors of positions and normals of same
    /// length.
    ///
    /// Does not run any validations except for length checking.
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
            .zip(barycentric_sequence_iter())
            .map(|((position, normal), barycentric)| Self::vertex(position, normal, barycentric))
            .collect();

        Self {
            indices: None,
            vertex_data,
        }
    }

    /// Create indexed mesh from vectors of positions and normals
    /// of same length. Does not run any validations except for length
    /// checking.
    #[allow(dead_code)]
    pub fn from_positions_and_normals_indexed(
        indices: Vec<GpuMeshIndex>,
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
            .zip(barycentric_sequence_iter())
            .map(|((position, normal), barycentric)| Self::vertex(position, normal, barycentric))
            .collect();

        Self {
            indices: Some(indices),
            vertex_data,
        }
    }

    fn vertex(position: Point3<f32>, normal: Vector3<f32>, barycentric: u32) -> GpuMeshVertex {
        GpuMeshVertex {
            position: [position[0], position[1], position[2], 1.0],
            normal: [normal[0], normal[1], normal[2], 0.0],
            barycentric,
        }
    }
}

/// Opaque handle to mesh stored in scene renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMeshId(u64);

#[derive(Debug)]
pub enum AddMeshError {
    TooManyVertices(usize),
    TooManyIndices(usize),
}

impl fmt::Display for AddMeshError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AddMeshError::TooManyVertices(given) => write!(
                f,
                "Mesh contains too many vertices {}. (max allowed is {})",
                given,
                u32::max_value(),
            ),
            AddMeshError::TooManyIndices(given) => write!(
                f,
                "Mesh contains too many indices: {}. (max allowed is {})",
                given,
                u32::max_value(),
            ),
        }
    }
}

impl error::Error for AddMeshError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub clear_color: [f64; 4],
    pub sample_count: u32,
    pub output_color_attachment_format: wgpu::TextureFormat,
    pub output_depth_attachment_format: wgpu::TextureFormat,
}

bitflags! {
    pub struct ClearFlags: u8 {
        const COLOR = 0b_0000_0001;
        const DEPTH = 0b_0000_0010;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawMeshMode {
    Shaded,
    Edges,
    ShadedEdges,
    ShadedEdgesXray,
}

/// 3D renderer of the editor scene.
///
/// Can be used to upload meshes on the GPU and draw it in the
/// viewport. Currently supports just shaded (matcap) and wireframe
/// rendering, and their combinations.
pub struct SceneRenderer {
    mesh_resources: HashMap<u64, MeshResource>,
    mesh_resources_next_id: u64,
    matrix_buffer: wgpu::Buffer,
    matrix_bind_group: wgpu::BindGroup,
    shading_bind_group_shaded: wgpu::BindGroup,
    shading_bind_group_edges: wgpu::BindGroup,
    shading_bind_group_shaded_edges: wgpu::BindGroup,
    matcap_texture_bind_group: wgpu::BindGroup,
    render_pipeline_opaque: wgpu::RenderPipeline,
    render_pipeline_transparent: wgpu::RenderPipeline,
    options: Options,
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
        options: Options,
    ) -> Self {
        let vs_words = wgpu::read_spirv(io::Cursor::new(SHADER_VIEWPORT_VERT))
            .expect("Couldn't read pre-built SPIR-V");
        let fs_words = wgpu::read_spirv(io::Cursor::new(SHADER_VIEWPORT_FRAG))
            .expect("Couldn't read pre-built SPIR-V");
        let vs_module = device.create_shader_module(&vs_words);
        let fs_module = device.create_shader_module(&fs_words);

        let matrix_buffer_size = wgpu_size_of::<MatrixUniforms>();
        let matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: matrix_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                }],
            });
        let matrix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &matrix_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &matrix_buffer,
                    range: 0..matrix_buffer_size,
                },
            }],
        });
        let matrix_uniforms = MatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };

        let shading_buffer_size = wgpu_size_of::<ShadingUniforms>();
        let shading_buffer_shaded = device.create_buffer(&wgpu::BufferDescriptor {
            size: shading_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        let shading_buffer_edges = device.create_buffer(&wgpu::BufferDescriptor {
            size: shading_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        let shading_buffer_shaded_edges = device.create_buffer(&wgpu::BufferDescriptor {
            size: shading_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let shading_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                }],
            });
        let shading_bind_group_shaded = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shading_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &shading_buffer_shaded,
                    range: 0..shading_buffer_size,
                },
            }],
        });
        let shading_bind_group_edges = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shading_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &shading_buffer_edges,
                    range: 0..shading_buffer_size,
                },
            }],
        });
        let shading_bind_group_shaded_edges =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shading_bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &shading_buffer_shaded_edges,
                        range: 0..shading_buffer_size,
                    },
                }],
            });

        upload_matrix_buffer(device, queue, &matrix_buffer, matrix_uniforms);
        upload_shading_buffer(
            device,
            queue,
            &shading_buffer_shaded,
            ShadingUniforms {
                edge_color_and_face_alpha: [0.0, 0.0, 0.0, 1.0],
                shading_mode: ShadingMode::SHADED,
            },
        );
        upload_shading_buffer(
            device,
            queue,
            &shading_buffer_edges,
            ShadingUniforms {
                edge_color_and_face_alpha: [0.239, 0.306, 0.400, 1.0],
                shading_mode: ShadingMode::EDGES,
            },
        );
        upload_shading_buffer(
            device,
            queue,
            &shading_buffer_shaded_edges,
            ShadingUniforms {
                edge_color_and_face_alpha: [0.239, 0.306, 0.400, 1.0],
                shading_mode: ShadingMode::SHADED | ShadingMode::EDGES,
            },
        );

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
                    resource: wgpu::BindingResource::TextureView(
                        &matcap_texture.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&matcap_sampler),
                },
            ],
        });

        upload_texture_rgba8_unorm(
            device,
            queue,
            &matcap_texture,
            matcap_texture_width,
            matcap_texture_height,
            &matcap_texture_data,
        );

        let render_pipeline_opaque = create_pipeline(
            device,
            &vs_module,
            &fs_module,
            &matrix_bind_group_layout,
            &shading_bind_group_layout,
            &matcap_texture_bind_group_layout,
            false,
            options,
        );
        let render_pipeline_transparent = create_pipeline(
            device,
            &vs_module,
            &fs_module,
            &matrix_bind_group_layout,
            &shading_bind_group_layout,
            &matcap_texture_bind_group_layout,
            true,
            options,
        );

        Self {
            mesh_resources: HashMap::new(),
            mesh_resources_next_id: 0,
            matrix_buffer,
            matrix_bind_group,
            shading_bind_group_shaded,
            shading_bind_group_edges,
            shading_bind_group_shaded_edges,
            matcap_texture_bind_group,
            render_pipeline_opaque,
            render_pipeline_transparent,
            options,
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
        let matrix_uniforms = MatrixUniforms {
            projection_matrix: apply_wgpu_correction_matrix(projection_matrix).into(),
            view_matrix: view_matrix.clone().into(),
        };
        upload_matrix_buffer(device, queue, &self.matrix_buffer, matrix_uniforms);
    }

    /// Upload mesh on the GPU.
    ///
    /// Whether indexed or not, the data must be in the
    /// `TRIANGLE_LIST` format. The returned id can be used to draw
    /// the mesh, or remove it.
    pub fn add_mesh(
        &mut self,
        device: &wgpu::Device,
        mesh: &GpuMesh,
    ) -> Result<GpuMeshId, AddMeshError> {
        let id = GpuMeshId(self.mesh_resources_next_id);

        let vertex_data = &mesh.vertex_data[..];
        let vertex_data_count = u32::try_from(vertex_data.len())
            .map_err(|_| AddMeshError::TooManyVertices(vertex_data.len()))?;

        let mesh_descriptor = if let Some(indices) = &mesh.indices {
            let index_count = u32::try_from(indices.len())
                .map_err(|_| AddMeshError::TooManyIndices(indices.len()))?;

            log::debug!(
                "Adding mesh with ID {}, {} vertices and {} indices",
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

            MeshResource {
                vertices: (vertex_buffer, vertex_data_count),
                indices: Some((index_buffer, index_count)),
            }
        } else {
            log::debug!(
                "Adding mesh with ID {} and {} vertices",
                id.0,
                vertex_data_count
            );

            let vertex_buffer = device
                .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
                .fill_from_slice(vertex_data);

            MeshResource {
                vertices: (vertex_buffer, vertex_data_count),
                indices: None,
            }
        };

        self.mesh_resources.insert(id.0, mesh_descriptor);
        self.mesh_resources_next_id += 1;
        Ok(id)
    }

    /// Remove a previously uploaded mesh from the GPU.
    pub fn remove_mesh(&mut self, id: GpuMeshId) {
        log::debug!("Removing mesh with ID {}", id.0);
        // Dropping the mesh descriptor here unstreams the buffers from device memory
        self.mesh_resources.remove(&id.0);
    }

    /// Optionally clear color and depth and draw previously uploaded
    /// meshes as one of the commands executed with the `encoder`
    /// to the `color_attachment`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_mesh<'a, I>(
        &self,
        mode: DrawMeshMode,
        clear_flags: ClearFlags,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        msaa_attachment: Option<&wgpu::TextureView>,
        depth_attachment: &wgpu::TextureView,
        ids: I,
    ) where
        I: Iterator<Item = &'a GpuMeshId> + Clone,
    {
        let color_load_op = if clear_flags.contains(ClearFlags::COLOR) {
            wgpu::LoadOp::Clear
        } else {
            wgpu::LoadOp::Load
        };

        let depth_load_op = if clear_flags.contains(ClearFlags::DEPTH) {
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
                    r: self.options.clear_color[0],
                    g: self.options.clear_color[1],
                    b: self.options.clear_color[2],
                    a: self.options.clear_color[3],
                },
            }
        } else {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: color_attachment,
                resolve_target: None,
                load_op: color_load_op,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: self.options.clear_color[0],
                    g: self.options.clear_color[1],
                    b: self.options.clear_color[2],
                    a: self.options.clear_color[3],
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

        // FIXME: The current renderer architecture is enough for our
        // current needs, but has some serious downsides.
        //
        // - We should be doing object sorting. We currently live
        //   without it as the only transparent objects we have are
        //   the edges and those do not show mis-blending artifacts,
        //   because the area that of their fragments that is neither
        //   fully transparent nor fully opaque is very small. If we
        //   ever want to support transparency, we need to sort
        //   objects.
        //
        // - We don't mitigate self-transparency issues (because we
        //   don't experience them much). The simplest mitigation
        //   would be to draw back and front faces separately, with 2
        //   different pipelines (for culling settings). There are
        //   also more advanced techniques, such as:
        //
        //   * Weighted Blended Order-Independent Transparency
        //     http://jcgt.org/published/0002/02/09/
        //   * Stochastic Transparency
        //     http://www.cse.chalmers.se/~d00sint/StochasticTransparency_I3D2010.pdf
        //   * Adaptive Transparency
        //     https://software.intel.com/en-us/articles/adaptive-transparency-hpg-2011

        match mode {
            DrawMeshMode::Shaded => {
                rpass.set_pipeline(&self.render_pipeline_opaque);
                rpass.set_bind_group(0, &self.matrix_bind_group, &[]);
                rpass.set_bind_group(1, &self.shading_bind_group_shaded, &[]);
                rpass.set_bind_group(2, &self.matcap_texture_bind_group, &[]);

                self.record(&mut rpass, ids);
            }
            DrawMeshMode::Edges => {
                rpass.set_pipeline(&self.render_pipeline_transparent);
                rpass.set_bind_group(0, &self.matrix_bind_group, &[]);
                rpass.set_bind_group(1, &self.shading_bind_group_edges, &[]);
                rpass.set_bind_group(2, &self.matcap_texture_bind_group, &[]);

                self.record(&mut rpass, ids);
            }
            DrawMeshMode::ShadedEdges => {
                rpass.set_pipeline(&self.render_pipeline_opaque);
                rpass.set_bind_group(0, &self.matrix_bind_group, &[]);
                rpass.set_bind_group(1, &self.shading_bind_group_shaded_edges, &[]);
                rpass.set_bind_group(2, &self.matcap_texture_bind_group, &[]);

                self.record(&mut rpass, ids);
            }
            DrawMeshMode::ShadedEdgesXray => {
                rpass.set_pipeline(&self.render_pipeline_opaque);
                rpass.set_bind_group(0, &self.matrix_bind_group, &[]);
                rpass.set_bind_group(1, &self.shading_bind_group_shaded, &[]);
                rpass.set_bind_group(2, &self.matcap_texture_bind_group, &[]);

                self.record(&mut rpass, ids.clone());

                rpass.set_pipeline(&self.render_pipeline_transparent);
                rpass.set_bind_group(1, &self.shading_bind_group_edges, &[]);

                self.record(&mut rpass, ids);
            }
        }
    }

    fn record<'a, I>(&self, rpass: &mut wgpu::RenderPass, ids: I)
    where
        I: IntoIterator<Item = &'a GpuMeshId>,
    {
        for id in ids {
            if let Some(mesh) = &self.mesh_resources.get(&id.0) {
                let (vertex_buffer, vertex_count) = &mesh.vertices;
                rpass.set_vertex_buffers(0, &[(vertex_buffer, 0)]);
                if let Some((index_buffer, index_count)) = &mesh.indices {
                    rpass.set_index_buffer(&index_buffer, 0);
                    rpass.draw_indexed(0..*index_count, 0, 0..1);
                } else {
                    rpass.draw(0..*vertex_count, 0..1);
                }
            } else {
                log::warn!("Mesh with id {} does not exist in this renderer.", id.0);
            }
        }
    }
}

struct MeshResource {
    vertices: (wgpu::Buffer, u32),
    indices: Option<(wgpu::Buffer, u32)>,
}

/// The mesh vertex data as uploaded on the GPU.
///
/// Positions and normals are internally `[f32; 4]` with the last
/// component filled in as 1.0 or 0.0 for points and vectors
/// respectively.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct GpuMeshVertex {
    /// The position of the vertex in world-space. Last component is 1.
    pub position: [f32; 4],

    /// The normal of the vertex in world-space. Last component is 0.
    pub normal: [f32; 4],

    /// Barycentric coordinates of the current vertex within the
    /// triangle primitive. First bit means `(1, 0, 0)`, second `(0,
    /// 1, 0)`, and the third `(0, 0, 1)`. The rest of the bits are 0.
    pub barycentric: u32,
}

// FIXME: @Optimization Determine u16/u32 dynamically per mesh to
// save memory
type GpuMeshIndex = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct MatrixUniforms {
    projection_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct ShadingUniforms {
    edge_color_and_face_alpha: [f32; 4],
    shading_mode: ShadingMode,
}

bitflags! {
    pub struct ShadingMode: u32 {
        const SHADED = 0x01;
        const EDGES = 0x02;
    }
}

fn upload_matrix_buffer(
    device: &wgpu::Device,
    queue: &mut wgpu::Queue,
    matrix_buffer: &wgpu::Buffer,
    matrix_uniforms: MatrixUniforms,
) {
    let matrix_uniforms_size = wgpu_size_of::<MatrixUniforms>();

    let transfer_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&[matrix_uniforms]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(&transfer_buffer, 0, matrix_buffer, 0, matrix_uniforms_size);

    queue.submit(&[encoder.finish()]);
}

fn upload_shading_buffer(
    device: &wgpu::Device,
    queue: &mut wgpu::Queue,
    shading_buffer: &wgpu::Buffer,
    shading_uniforms: ShadingUniforms,
) {
    let shading_uniforms_size = wgpu_size_of::<ShadingUniforms>();

    let transfer_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&[shading_uniforms]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(
        &transfer_buffer,
        0,
        shading_buffer,
        0,
        shading_uniforms_size,
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

/// Produces an infinite iterator over bit-packed barycentric
/// coordinates of triangle vertices.
///
/// Barycentric coords (1, 0, 0), (0, 1, 0) and (0, 0, 1) are
/// bit-packed into a single u32 to save space (possibly, depending on
/// attribute data layout and alignment). They are unpacked on the
/// vertex shader. Usage is to zip this iterator with other data
/// iterators to produce vertex attributes for renderer mesh.
fn barycentric_sequence_iter() -> impl Iterator<Item = u32> {
    iter::successors(Some(0x01), |predecessor| match predecessor {
        0x01 => Some(0x02),
        0x02 => Some(0x04),
        0x04 => Some(0x01),
        _ => unreachable!(),
    })
}

#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    device: &wgpu::Device,
    vs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    matrix_bind_group_layout: &wgpu::BindGroupLayout,
    shading_bind_group_layout: &wgpu::BindGroupLayout,
    matcap_texture_bind_group_layout: &wgpu::BindGroupLayout,
    support_transparency: bool,
    options: Options,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[
            &matrix_bind_group_layout,
            &shading_bind_group_layout,
            &matcap_texture_bind_group_layout,
        ],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: fs_module,
            entry_point: "main",
        }),
        // FIXME: @Correctness Draw backfaces differently.
        //
        // Default rasterization state means CullMode::None. We don't
        // cull faces yet, because we work with potentially non-CCW
        // meshes. We might implement special rendering for CW
        // faces one day.
        rasterization_state: None,
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: options.output_color_attachment_format,
            color_blend: if support_transparency {
                wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                }
            } else {
                wgpu::BlendDescriptor::REPLACE
            },
            alpha_blend: if support_transparency {
                wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                }
            } else {
                wgpu::BlendDescriptor::REPLACE
            },
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: options.output_depth_attachment_format,
            depth_write_enabled: !support_transparency,
            depth_compare: if support_transparency {
                wgpu::CompareFunction::Always
            } else {
                wgpu::CompareFunction::Less
            },
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }),
        index_format: wgpu::IndexFormat::Uint32,
        vertex_buffers: &[wgpu::VertexBufferDescriptor {
            stride: wgpu_size_of::<GpuMeshVertex>(),
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    format: wgpu::VertexFormat::Float4,
                    shader_location: 0,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: wgpu_size_of::<[f32; 4]>(), // 4 bytes * 4 components * 1 attrib
                    format: wgpu::VertexFormat::Float4,
                    shader_location: 1,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: wgpu_size_of::<[f32; 4]>() * 2, // 4 bytes * 4 components * 2 attribs
                    format: wgpu::VertexFormat::Uint,
                    shader_location: 2,
                },
            ],
        }],
        sample_count: options.sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

#[cfg(test)]
mod tests {
    use crate::mesh::TriangleFace;

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

    fn triangle_indexed() -> (Vec<GpuMeshIndex>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
        let (vertex_positions, vertex_normals) = triangle();
        let indices = vec![0, 1, 2];

        (indices, vertex_positions, vertex_normals)
    }

    fn triangle_mesh_same_len() -> Mesh {
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
            TriangleFace::from_same_vertex_and_normal_index(0, 1, 2)
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, positions, normals)
    }

    fn triangle_mesh_var_len() -> Mesh {
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
            TriangleFace::new(0, 1, 2, 0, 0, 0)
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, positions, normals)
    }

    #[test]
    fn test_gpu_mesh_from_positions_and_normals() {
        let (positions, normals) = triangle();
        let mesh = GpuMesh::from_positions_and_normals(positions, normals);

        assert_eq!(
            mesh.vertex_data,
            vec![
                GpuMeshVertex {
                    position: [-0.3, -0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    barycentric: 0x01,
                },
                GpuMeshVertex {
                    position: [0.3, -0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    barycentric: 0x02,
                },
                GpuMeshVertex {
                    position: [0.0, 0.5, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    barycentric: 0x04,
                },
            ]
        );
        assert_eq!(mesh.indices, None);
    }

    #[test]
    fn test_gpu_mesh_from_positions_and_normals_indexed() {
        let (indices, positions, normals) = triangle_indexed();
        let mesh = GpuMesh::from_positions_and_normals_indexed(indices, positions, normals);

        let expected_vertex_data = vec![
            GpuMeshVertex {
                position: [-0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x01,
            },
            GpuMeshVertex {
                position: [0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x02,
            },
            GpuMeshVertex {
                position: [0.0, 0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x04,
            },
        ];

        assert_eq!(mesh.vertex_data, expected_vertex_data);
        assert_eq!(mesh.indices, Some(vec![0, 1, 2]));
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_gpu_mesh_from_positions_and_normals_panics_on_different_length_data() {
        let (_, normals) = triangle();
        GpuMesh::from_positions_and_normals(vec![Point3::new(1.0, 1.0, 1.0)], normals);
    }

    #[test]
    #[should_panic(expected = "Per-vertex data must be same length")]
    fn test_gpu_mesh_from_positions_and_normals_indexed_panics_on_different_length_data() {
        let (indices, positions, _) = triangle_indexed();
        GpuMesh::from_positions_and_normals_indexed(
            indices,
            positions,
            vec![Vector3::new(1.0, 1.0, 1.0)],
        );
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_gpu_mesh_from_positions_and_normals_panics_on_empty_positions() {
        let (_, normals) = triangle();
        GpuMesh::from_positions_and_normals(vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_gpu_mesh_from_positions_and_normals_indexed_panics_on_empty_positions() {
        let (positions, _) = triangle();
        GpuMesh::from_positions_and_normals(positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Vertex positions must not be empty")]
    fn test_gpu_mesh_from_positions_and_normals_panics_on_empty_normals() {
        let (indices, _, normals) = triangle_indexed();
        GpuMesh::from_positions_and_normals_indexed(indices, vec![], normals);
    }

    #[test]
    #[should_panic(expected = "Vertex normals must not be empty")]
    fn test_gpu_mesh_from_positions_and_normals_indexed_panics_on_empty_normals() {
        let (indices, positions, _) = triangle_indexed();
        GpuMesh::from_positions_and_normals_indexed(indices, positions, vec![]);
    }

    #[test]
    #[should_panic(expected = "Indices must not be empty")]
    fn test_gpu_mesh_from_positions_and_normals_indexed_panics_on_empty_indices() {
        let (_, vertices, normals) = triangle_indexed();
        GpuMesh::from_positions_and_normals_indexed(vec![], vertices, normals);
    }

    #[test]
    fn test_gpu_mesh_from_mesh_preserves_already_same_len_mesh() {
        let mesh = GpuMesh::from_mesh(&triangle_mesh_same_len());

        let expected_vertex_data = vec![
            GpuMeshVertex {
                position: [-0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x01,
            },
            GpuMeshVertex {
                position: [0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x02,
            },
            GpuMeshVertex {
                position: [0.0, 0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x04,
            },
        ];

        assert_eq!(mesh.vertex_data, expected_vertex_data);
        assert_eq!(mesh.indices, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_gpu_mesh_from_mesh_duplicates_normals_in_var_len_mesh() {
        let mesh = GpuMesh::from_mesh(&triangle_mesh_var_len());

        let expected_vertex_data = vec![
            GpuMeshVertex {
                position: [-0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x01,
            },
            GpuMeshVertex {
                position: [0.3, -0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x02,
            },
            GpuMeshVertex {
                position: [0.0, 0.5, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0, 0.0],
                barycentric: 0x04,
            },
        ];

        assert_eq!(mesh.vertex_data, expected_vertex_data);
        assert_eq!(mesh.indices, Some(vec![0, 1, 2]));
    }
}

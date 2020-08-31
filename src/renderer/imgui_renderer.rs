use imgui::internal::RawWrapper as _;
use zerocopy::AsBytes as _;

use super::common;

static SHADER_IMGUI_VERT: &[u8] = include_shader!("imgui.vert.spv");
static SHADER_IMGUI_FRAG: &[u8] = include_shader!("imgui.frag.spv");

#[derive(Debug, Clone)]
pub enum Error {
    BadTexture(imgui::TextureId),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub output_color_attachment_format: wgpu::TextureFormat,
}

pub struct ImguiRenderer {
    texture_resources: imgui::Textures<Texture>,
    sampler: wgpu::Sampler,
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
}

impl ImguiRenderer {
    pub fn new(
        mut imgui_font_atlas: imgui::FontAtlasRefMut,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        options: Options,
    ) -> Result<ImguiRenderer, Error> {
        let vs_source = wgpu::util::make_spirv(SHADER_IMGUI_VERT);
        let fs_source = wgpu::util::make_spirv(SHADER_IMGUI_FRAG);
        let vs_module = device.create_shader_module(vs_source);
        let fs_module = device.create_shader_module(fs_source);

        // Create transform uniform buffer bind group
        let transform_buffer_size = common::wgpu_size_of::<TransformUniforms>();
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: transform_buffer_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        // TODO(yanchith): Provide this to optimize
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(transform_buffer.slice(..)),
            }],
        });

        // Create texture uniform bind group
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                ],
            });

        // Create render pipeline

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&transform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Setup render state: alpha-blending enabled, no face
        // culling, no depth testing

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: None,
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: options.output_color_attachment_format,
                // Enable alpha blending
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],
            // Disable depth test
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: common::wgpu_size_of::<imgui::DrawVert>(),
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            offset: 0,
                            format: wgpu::VertexFormat::Float2,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            offset: 8,
                            format: wgpu::VertexFormat::Float2,
                            shader_location: 1,
                        },
                        wgpu::VertexAttributeDescriptor {
                            offset: 16,
                            format: wgpu::VertexFormat::Uint,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Create the font texture and add it to the font atlas

        let font_atlas_image = imgui_font_atlas.build_rgba32_texture();
        let font_atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: font_atlas_image.width,
                height: font_atlas_image.height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        common::upload_texture_rgba8_unorm(
            queue,
            &font_atlas_texture,
            font_atlas_image.width,
            font_atlas_image.height,
            font_atlas_image.data,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let font_atlas_texture_resource = Texture::new(
            device,
            &texture_bind_group_layout,
            &font_atlas_texture,
            &sampler,
        );
        let mut texture_resources = imgui::Textures::new();
        let font_atlas_texture_id = texture_resources.insert(font_atlas_texture_resource);
        imgui_font_atlas.tex_id = font_atlas_texture_id;

        Ok(ImguiRenderer {
            texture_resources,
            sampler,
            render_pipeline,
            transform_buffer,
            transform_bind_group,
            texture_bind_group_layout,
            vertex_buffers: Vec::new(),
            index_buffers: Vec::new(),
        })
    }

    // FIXME: Return our own `TextureHandle` instead of imgui's
    // `TextureId` so that we can disallow `Clone` and `Copy` and
    // properly track it's ownership in the type system. Imgui's
    // `TextureId` can be converted to/from usize, so this should be
    // possible.

    pub fn add_texture_rgba8_unorm(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> imgui::TextureId {
        assert_eq!(data.len() % 4, 0);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        common::upload_texture_rgba8_unorm(queue, &texture, width, height, data);

        let texture_resource = Texture::new(
            device,
            &self.texture_bind_group_layout,
            &texture,
            &self.sampler,
        );
        self.texture_resources.insert(texture_resource)
    }

    pub fn remove_texture(&mut self, id: imgui::TextureId) {
        self.texture_resources.remove(id);
    }

    pub fn draw_ui(
        &mut self,
        color_needs_clearing: bool,
        clear_color: [f64; 4],
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        draw_data: &imgui::DrawData,
    ) -> Result<(), Error> {
        // This is mostly a transcript of the following:
        // https://github.com/ocornut/imgui/blob/master/examples/imgui_impl_opengl3.cpp
        // https://github.com/ocornut/imgui/blob/master/examples/imgui_impl_vulkan.cpp
        // https://github.com/Gekkio/imgui-rs/blob/master/imgui-glium-renderer/src/lib.rs

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if fb_width <= 0.0 && fb_height <= 0.0 {
            return Ok(());
        }

        let transform_matrix = {
            // Setup orthographic projection matrix. Our visible imgui space
            // lies from `draw_data.display_pos` (top left) to
            // `draw_data->display_pos + data_data->display_size` (bottom
            // right).
            let l = draw_data.display_pos[0];
            let r = draw_data.display_pos[0] + draw_data.display_size[0];
            let t = draw_data.display_pos[1];
            let b = draw_data.display_pos[1] + draw_data.display_size[1];

            #[rustfmt::skip]
            let matrix = [
                [2.0 / (r - l)    , 0.0              , 0.0, 0.0],
                [0.0              , 2.0 / (t - b)    , 0.0, 0.0],
                [0.0              , 0.0              , 0.5, 0.0],
                [(r + l) / (l - r), (t + b) / (b - t), 0.5, 1.0],
            ];

            matrix
        };

        queue.write_buffer(
            &self.transform_buffer,
            0,
            [TransformUniforms {
                matrix: transform_matrix,
            }]
            .as_bytes(),
        );

        // Will project scissor/clipping rectangles into framebuffer space
        let clip_off = draw_data.display_pos; // (0,0) unless using multi-viewports
        let clip_scale = draw_data.framebuffer_scale; // (1,1) unless using hidpi

        // The rendering process is as follows:
        //
        // 1) We begin the render pass and set the pipeline and bind
        // group for the already uploaded uniforms as those don't
        // change for the whole frame.
        //
        // 2) For each processed draw list, we create new vertex and
        // index buffer, and set them to the render pass as they stay
        // the same for the entire draw list.
        //
        // 3) For each draw command in a draw list, we figure out
        // clipping (and don't draw anything if it would be clipped),
        // the current texture to use, and our current index window
        // `idx_start..idx_end` and set those to the render pass, and
        // finally draw.

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: color_attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if color_needs_clearing {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0],
                            g: clear_color[1],
                            b: clear_color[2],
                            a: clear_color[3],
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.transform_bind_group, &[]);

        self.vertex_buffers.clear();
        self.index_buffers.clear();

        // We enumerate twice, because we need to keep the created buffers alive
        // for the duration of the render pass.

        for draw_list in draw_data.draw_lists() {
            let vtx_buffer = ImguiVertex::cast(draw_list.vtx_buffer());
            let vertex_buffer =
                common::create_buffer(device, wgpu::BufferUsage::VERTEX, vtx_buffer);

            let idx_buffer = draw_list.idx_buffer();
            let index_buffer = common::create_buffer(device, wgpu::BufferUsage::INDEX, idx_buffer);

            self.vertex_buffers.push(vertex_buffer);
            self.index_buffers.push(index_buffer);
        }

        for (draw_list_index, draw_list) in draw_data.draw_lists().enumerate() {
            rpass.set_vertex_buffer(0, self.vertex_buffers[draw_list_index].slice(..));
            rpass.set_index_buffer(self.index_buffers[draw_list_index].slice(..));

            let mut idx_start = 0;
            for cmd in draw_list.commands() {
                match cmd {
                    imgui::DrawCmd::Elements {
                        count,
                        cmd_params:
                            imgui::DrawCmdParams {
                                clip_rect,
                                texture_id,
                                ..
                            },
                    } => {
                        let idx_end = idx_start + count as u32;

                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            let texture = self
                                .texture_resources
                                .get(texture_id)
                                .ok_or_else(|| Error::BadTexture(texture_id))?;

                            rpass.set_bind_group(1, texture.bind_group(), &[]);
                            rpass.set_scissor_rect(
                                clip_rect[0].max(0.0).min(fb_width).round() as u32,
                                clip_rect[1].max(0.0).min(fb_height).round() as u32,
                                (clip_rect[2] - clip_rect[0]).abs().min(fb_width).round() as u32,
                                (clip_rect[3] - clip_rect[1]).abs().min(fb_height).round() as u32,
                            );
                            rpass.draw_indexed(idx_start..idx_end, 0, 0..1);

                            idx_start = idx_end;
                        }
                    }

                    // Our render state is mostly predefined in the
                    // render pipeline, not much to reset here
                    imgui::DrawCmd::ResetRenderState => (),

                    imgui::DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd)
                    },
                }
            }
        }

        Ok(())
    }
}

struct Texture {
    bind_group: wgpu::BindGroup,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: &wgpu::Texture,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Texture { bind_group }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, zerocopy::AsBytes)]
struct ImguiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    col: [u8; 4],
}

static_assertions::assert_eq_size!(ImguiVertex, imgui::DrawVert);
static_assertions::assert_eq_align!(ImguiVertex, imgui::DrawVert);

impl ImguiVertex {
    /// Transmutes `&[imgui::DrawVert]` to `&[ImguiVertex]`.
    pub fn cast(slice: &[imgui::DrawVert]) -> &[ImguiVertex] {
        let len = slice.len();
        let ptr = slice.as_ptr() as *const ImguiVertex;

        // SAFETY: This is only safe when `imgui::DrawVert` and `ImguiVertex`
        // have the same memory layout, size and alignment.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, zerocopy::AsBytes)]
struct TransformUniforms {
    matrix: [[f32; 4]; 4],
}

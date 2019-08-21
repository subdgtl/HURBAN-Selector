use std::convert::TryFrom;
use std::mem;

use imgui;
use imgui::internal::RawWrapper;

use crate::include_shader;

#[derive(Debug, Clone)]
pub enum ImguiRendererError {
    BadTexture(imgui::TextureId),
}

pub struct ImguiRenderer {
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    textures: imgui::Textures<Texture>,
    clear_color: Option<wgpu::Color>,
}

impl ImguiRenderer {
    pub fn new(
        imgui: &mut imgui::Context,
        device: &mut wgpu::Device,
        format: wgpu::TextureFormat,
        clear_color: Option<wgpu::Color>,
    ) -> Result<ImguiRenderer, ImguiRendererError> {
        // Link shaders

        let vs_spv = include_shader!("imgui.vert.spv");
        let fs_spv = include_shader!("imgui.frag.spv");
        let vs_module = device.create_shader_module(vs_spv);
        let fs_module = device.create_shader_module(fs_spv);

        // Create ortho projection matrix uniform buffer, layout and bind group

        let uniform_buffer_size = wgpu_size_of::<TransformUniforms>();
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: uniform_buffer_size,
            // FIXME(yanchith): `TRANSFER_DST` is required because the
            // only way to upload the buffer currently is by issueing
            // the transfer command in `upload_buffer_immediate`. We
            // can remove the flag once we get rid of the hack and
            // learn to write to mapped buffers correctly.
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    range: 0..uniform_buffer_size,
                },
            }],
        });

        // Create texture uniforms layout

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // Create render pipeline

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_layout, &texture_layout],
        });

        // Setup render state: alpha-blending enabled, no face
        // culling, no depth testing

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
                cull_mode: wgpu::CullMode::None,
                // Depth test is disabled, no need for these to mean anything
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_scale: 0.0,
            },
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format,
                // Enable alpha blending
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],
            // Disabled depth test
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16, // FIXME(yanchith): may need 32bit indices!
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: u64::try_from(wgpu_size_of::<imgui::DrawVert>()).unwrap(),
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
            sample_count: 1,
        });

        // Create the font atlas texture

        let mut fonts = imgui.fonts();
        let font_texture = fonts.build_rgba32_texture();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: font_texture.width,
                height: font_texture.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::TRANSFER_DST,
        });

        upload_texture_immediate(
            device,
            &texture,
            font_texture.width,
            font_texture.height,
            font_texture.data,
        );

        let sampler = create_sampler(device);
        let pair = Texture::new(device, &texture_layout, texture, sampler);
        let mut textures = imgui::Textures::new();
        let atlas_id = textures.insert(pair);
        fonts.tex_id = atlas_id;

        Ok(ImguiRenderer {
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            texture_layout,
            textures,
            clear_color,
        })
    }

    pub fn add_rgba32_texture(
        &mut self,
        device: &mut wgpu::Device,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> imgui::TextureId {
        assert_eq!(data.len() % 4, 0);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::TRANSFER_DST,
        });

        upload_texture_immediate(device, &texture, width, height, data);

        let sampler = create_sampler(device);
        let pair = Texture::new(device, &self.texture_layout, texture, sampler);
        self.textures.insert(pair)
    }

    pub fn render(
        &mut self,
        device: &mut wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_attachment: &wgpu::TextureView,
        draw_data: &imgui::DrawData,
    ) -> Result<(), ImguiRendererError> {
        // This is mostly a transcript of the following:
        // https://github.com/ocornut/imgui/blob/master/examples/imgui_impl_opengl3.cpp
        // https://github.com/ocornut/imgui/blob/master/examples/imgui_impl_vulkan.cpp
        // https://github.com/Gekkio/imgui-rs/blob/master/imgui-glium-renderer/src/lib.rs

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if fb_width <= 0.0 && fb_height <= 0.0 {
            return Ok(());
        }

        let scale = [
            2.0 / draw_data.display_size[0],
            2.0 / draw_data.display_size[1],
        ];
        let translate = [
            -1.0 - draw_data.display_pos[0] * scale[0],
            -1.0 - draw_data.display_pos[1] * scale[1],
        ];

        // FIXME(yanchith): try to use map_write_async here... but
        // figure out how to use it correctly beforehand. Currently,
        // even calling unmap does not force the callback in
        // map_write_async.

        // self.uniform_buffer.map_write_async(0, UNIFORM_BUFFER_SIZE, move |target| {
        //     println!("MAP");
        //     if let Ok(t) = target {
        //         t.data[0] = translate[0];
        //         t.data[1] = translate[1];
        //         t.data[2] = scale[0];
        //         t.data[3] = scale[1];
        //     }
        // });
        // println!("UNMAP starting");
        // self.uniform_buffer.unmap();
        // println!("UNMAP done");

        upload_buffer_immediate(
            device,
            &self.uniform_buffer,
            TransformUniforms { translate, scale },
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
                attachment: target_attachment,
                resolve_target: None,
                load_op: match self.clear_color {
                    Some(_) => wgpu::LoadOp::Clear,
                    None => wgpu::LoadOp::Load,
                },
                store_op: wgpu::StoreOp::Store,
                clear_color: self.clear_color.unwrap_or(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.uniform_bind_group, &[]);

        for draw_list in draw_data.draw_lists() {
            let vtx_buffer = draw_list.vtx_buffer();
            let vertex_buffer = device
                .create_buffer_mapped(vtx_buffer.len(), wgpu::BufferUsage::VERTEX)
                .fill_from_slice(vtx_buffer);

            let idx_buffer = draw_list.idx_buffer();
            let index_buffer = device
                .create_buffer_mapped(idx_buffer.len(), wgpu::BufferUsage::INDEX)
                .fill_from_slice(idx_buffer);

            rpass.set_vertex_buffers(&[(&vertex_buffer, 0)]);
            rpass.set_index_buffer(&index_buffer, 0);

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
                                .textures
                                .get(texture_id)
                                .ok_or_else(|| ImguiRendererError::BadTexture(texture_id))?;

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
        texture: wgpu::Texture,
        sampler: wgpu::Sampler,
    ) -> Self {
        let view = texture.create_default_view();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
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
#[derive(Debug, Clone, Copy)]
struct TransformUniforms {
    translate: [f32; 2],
    scale: [f32; 2],
}

fn upload_buffer_immediate(
    device: &mut wgpu::Device,
    buffer: &wgpu::Buffer,
    transform_uniforms: TransformUniforms,
) {
    let transform_uniforms_size = wgpu_size_of::<TransformUniforms>();
    let source_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::TRANSFER_SRC)
        .fill_from_slice(&[transform_uniforms]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

    encoder.copy_buffer_to_buffer(&source_buffer, 0, buffer, 0, transform_uniforms_size);

    device.get_queue().submit(&[encoder.finish()]);
}

fn upload_texture_immediate(
    device: &mut wgpu::Device,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    data: &[u8],
) {
    let count = data.len();
    let buffer = device
        .create_buffer_mapped(count, wgpu::BufferUsage::TRANSFER_SRC)
        .fill_from_slice(data);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

    let pixel_size = u32::try_from(count).unwrap() / width / height;
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

fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare_function: wgpu::CompareFunction::Always,
    })
}

fn wgpu_size_of<T>() -> wgpu::BufferAddress {
    let size = mem::size_of::<T>();
    wgpu::BufferAddress::try_from(size)
        .unwrap_or_else(|_| panic!("Size {} does not fit into wgpu BufferAddress", size))
}

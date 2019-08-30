use std::convert::TryFrom;
use std::mem;

use crate::convert::cast_u32;

#[macro_export]
macro_rules! include_shader {
    ($name:expr) => {{
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    }};
}

pub fn wgpu_size_of<T>() -> wgpu::BufferAddress {
    let size = mem::size_of::<T>();
    wgpu::BufferAddress::try_from(size)
        .unwrap_or_else(|_| panic!("Size {} does not fit into wgpu BufferAddress", size))
}

pub fn upload_texture_rgba8_unorm(
    device: &mut wgpu::Device,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    data: &[u8],
) {
    let buffer = device
        .create_buffer_mapped(data.len(), wgpu::BufferUsage::TRANSFER_SRC)
        .fill_from_slice(data);

    let byte_count = cast_u32(data.len());
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

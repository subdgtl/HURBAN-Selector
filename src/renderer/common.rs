use std::convert::TryFrom;
use std::iter;
use std::mem;

use crate::convert::{cast_u32, cast_u64};

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

pub fn create_buffer<T: zerocopy::AsBytes>(
    device: &wgpu::Device,
    usage: wgpu::BufferUsage,
    data: &[T],
) -> wgpu::Buffer {
    use zerocopy::AsBytes as _;

    let bytes = data.as_bytes();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cast_u64(bytes.len()),
        usage,
        mapped_at_creation: true,
    });

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytes);

    buffer
}

// FIXME: @Refactoring Take encoder instead of queue
pub fn upload_texture_rgba8_unorm(
    device: &wgpu::Device,
    queue: &mut wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    data: &[u8],
) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cast_u64(data.len()),
        usage: wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: true,
    });

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(data);

    let byte_count = cast_u32(data.len());
    let pixel_size = byte_count / width / height;

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &buffer,
            layout: wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: pixel_size * width,
                rows_per_image: height,
            },
        },
        wgpu::TextureCopyView {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        },
        wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
    );

    queue.submit(iter::once(encoder.finish()));
}

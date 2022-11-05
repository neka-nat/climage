use super::ClContext;
use image;
use ocl::enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType};
use ocl::Image;

pub struct ClImageBuffer {
    pub data: Image<u8>,
    dims: (u32, u32),
}

impl ClImageBuffer {
    pub fn from_host_image(
        context: &ClContext,
        img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        flag: ocl::flags::MemFlags,
    ) -> ClImageBuffer {
        let data = ocl::Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(img.dimensions())
            .flags(flag)
            .copy_host_slice(&img)
            .queue(context.queue.clone())
            .build()
            .unwrap();
        ClImageBuffer {
            data,
            dims: img.dimensions(),
        }
    }
    pub fn from_readonly_host_image(
        context: &ClContext,
        img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> ClImageBuffer {
        ClImageBuffer::from_host_image(
            context,
            img,
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
    }
    pub fn from_writeonly_host_image(
        context: &ClContext,
        img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> ClImageBuffer {
        ClImageBuffer::from_host_image(
            context,
            img,
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
    }
    pub fn dimensions(&self) -> (u32, u32) {
        self.dims
    }
}

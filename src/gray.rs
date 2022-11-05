use super::context::ClContext;
use super::imageproc::ImageProc;
use ocl::{Kernel, Program};

static KERNEL_SRC: &'static str = r#"
__constant sampler_t sampler_const = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;
__kernel void grayscale(
    __read_only image2d_t src_image,
    __write_only image2d_t dst_image) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int2 size = get_image_dim(src_image);

    if(all(gid < size)){
        uint4 pixel = read_imageui(src_image, sampler_const, gid);
        float4 color = convert_float4(pixel) / 255;
        color.xyz = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
        pixel = convert_uint4_rte(color * 255);
        write_imageui(dst_image, gid, pixel);
    }
}
"#;

pub struct Gray<'a> {
    context: &'a ClContext,
    program: Program,
    kernel: Option<Kernel>,
}

impl<'a> Gray<'a> {
    pub fn new(context: &'a ClContext) -> Self {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .build(&context.context)
            .unwrap();
        Gray {
            context,
            program,
            kernel: None,
        }
    }
}

impl<'a> ImageProc for Gray<'a> {
    fn build_kernel(
        &mut self,
        src_img: &crate::ClImageBuffer,
        dst_img: &mut crate::ClImageBuffer,
    ) -> &mut Self {
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name("grayscale")
            .queue(self.context.queue.clone())
            .global_work_size(src_img.dimensions())
            .arg(&src_img.data)
            .arg(&dst_img.data)
            .build()
            .unwrap();
        self.kernel = Some(kernel);
        self
    }
    fn run(&self) {
        unsafe { self.kernel.as_ref().unwrap().enq().unwrap() };
    }
}

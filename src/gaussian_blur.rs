use super::context::ClContext;
use super::imageproc::ImageProc;
use ocl::{Buffer, Kernel, Program};

static KERNEL_SRC: &'static str = r#"
__constant sampler_t sampler_const = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;
__kernel void gaussian_blur(
    __read_only image2d_t src_image,
    __write_only image2d_t dst_image,
    __constant float *weights,
    __private int const sigma) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int2 size = get_image_dim(src_image);

    if(all(gid < size)){
        float4 sum = 0;
        for(int i = -sigma; i < sigma + 1; i++) {
           for(int j = -sigma; j < sigma + 1; j++) {
               sum += weights[i + sigma + (j + sigma) * (sigma * 2 + 1)]
                   *read_imagef(src_image, sampler_const, gid + (int2)(i, j));
           }
        }
        write_imagef(dst_image, gid, sum);
    }
}
"#;

pub struct GaussianBlur<'a> {
    context: &'a ClContext,
    program: Program,
    kernel: Option<Kernel>,
    weight_buf: Buffer<f32>,
    sigma: i32,
}

impl<'a> GaussianBlur<'a> {
    pub fn new(context: &'a ClContext, sigma: i32) -> Self {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .build(&context.context)
            .unwrap();
        let weight_size = (sigma * 2 + 1) * (sigma * 2 + 1);
        let mut weights: Vec<f32> = vec![0.0; weight_size as usize];
        for i in -sigma..(sigma + 1) {
            for j in -sigma..(sigma + 1) {
                let weight = (-((i * i + j * j) as f32) / (2.0 * ((sigma * sigma) as f32))).exp();
                weights[(i + sigma + (j + sigma) * (sigma * 2 + 1)) as usize] = weight;
            }
        }
        let sum_weight = weights.iter().sum::<f32>();
        for i in 0..weight_size {
            weights[i as usize] /= sum_weight;
        }
        let weight_buf = Buffer::builder()
            .queue(context.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(weight_size)
            .copy_host_slice(&weights)
            .build()
            .unwrap();
        GaussianBlur {
            context,
            program,
            kernel: None,
            weight_buf,
            sigma,
        }
    }
}

impl<'a> ImageProc for GaussianBlur<'a> {
    fn build_kernel(
        &mut self,
        src_img: &crate::ClImageBuffer,
        dst_img: &mut crate::ClImageBuffer,
    ) -> &mut Self {
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name("gaussian_blur")
            .queue(self.context.queue.clone())
            .global_work_size(src_img.dimensions())
            .arg(&src_img.data)
            .arg(&dst_img.data)
            .arg(&self.weight_buf)
            .arg(&self.sigma)
            .build()
            .unwrap();
        self.kernel = Some(kernel);
        self
    }
    fn run(&self) {
        unsafe { self.kernel.as_ref().unwrap().enq().unwrap() };
    }
}

use super::context::ClContext;
use super::imageproc::ImageProc;
use ocl::{Buffer, Kernel, Program};

static KERNEL_SRC: &'static str = r#"
__constant sampler_t sampler_const = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;
__kernel void bilateral_filter(
    __read_only image2d_t src_image,
    __write_only image2d_t dst_image,
    __constant float *weights,
    __private int const sigma_color,
    __private int const sigma_space) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int2 size = get_image_dim(src_image);

    if(all(gid < size)){
        float4 sum = 0;
        float4 total_w = 0;
        float4 center_p = read_imagef(src_image, sampler_const, gid);
        for(int i = -sigma_space; i < sigma_space + 1; i++) {
            for(int j = -sigma_space; j < sigma_space + 1; j++) {
                if (gid.x + i < 0 || gid.x + i >= size.x || gid.y + j < 0 || gid.y + j >= size.y) {
                    continue;
                }
                float4 cur_p = read_imagef(src_image, sampler_const, gid + (int2)(i, j));
                float4 dp = center_p - cur_p;
                float4 weight = weights[i + sigma_space + (j + sigma_space) * (sigma_space * 2 + 1)] * exp(-(dp * dp) / (float4)(2.0 * sigma_color * sigma_color));
                sum += weight * cur_p;
                total_w += weight;
            }
        }
        sum /= total_w;
        write_imagef(dst_image, gid, sum);
    }
}
"#;

pub struct BilateralFilter<'a> {
    context: &'a ClContext,
    program: Program,
    kernel: Option<Kernel>,
    weight_buf: Buffer<f32>,
    sigma_color: i32,
    sigma_space: i32,
}

impl<'a> BilateralFilter<'a> {
    pub fn new(context: &'a ClContext, sigma_color: i32, sigma_space: i32) -> Self {
        let program = Program::builder()
            .src(KERNEL_SRC)
            .build(&context.context)
            .unwrap();
        let weight_size = (sigma_space * 2 + 1) * (sigma_space * 2 + 1);
        let mut weights: Vec<f32> = vec![0.0; weight_size as usize];
        for i in -sigma_space..(sigma_space + 1) {
            for j in -sigma_space..(sigma_space + 1) {
                let weight = (-((i * i + j * j) as f32)
                    / (2.0 * ((sigma_space * sigma_space) as f32)))
                    .exp();
                weights[(i + sigma_space + (j + sigma_space) * (sigma_space * 2 + 1)) as usize] =
                    weight;
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
        BilateralFilter {
            context,
            program,
            kernel: None,
            weight_buf,
            sigma_color,
            sigma_space,
        }
    }
}

impl<'a> ImageProc for BilateralFilter<'a> {
    fn build_kernel(
        &mut self,
        src_img: &crate::ClImageBuffer,
        dst_img: &mut crate::ClImageBuffer,
    ) -> &mut Self {
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name("bilateral_filter")
            .queue(self.context.queue.clone())
            .global_work_size(src_img.dimensions())
            .arg(&src_img.data)
            .arg(&dst_img.data)
            .arg(&self.weight_buf)
            .arg(&self.sigma_color)
            .arg(&self.sigma_space)
            .build()
            .unwrap();
        self.kernel = Some(kernel);
        self
    }
    fn run(&self) {
        unsafe { self.kernel.as_ref().unwrap().enq().unwrap() };
    }
}

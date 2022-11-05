use super::ClImageBuffer;

pub trait ImageProc {
    fn build_kernel(&mut self, src_img: &ClImageBuffer, dst_img: &mut ClImageBuffer) -> &mut Self;
    fn run(&self);
}

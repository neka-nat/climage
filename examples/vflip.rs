use climage;
use climage::ImageProc;

fn main() {
    let context = climage::ClContext::default();
    let mut vflip = climage::VFlip::new(&context);
    let img = climage::ClImageBuffer::from_readonly_host_image(
        &context,
        image::open("examples/lenna.png").unwrap().into_rgba8(),
    );
    let mut out = climage::ClImageBuffer::from_writeonly_host_image(
        &context,
        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(img.dimensions().0, img.dimensions().1),
    );
    vflip.build_kernel(&img, &mut out).run();
    let mut out_img =
        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(out.dimensions().0, out.dimensions().1);
    out.data.read(&mut out_img).enq().unwrap();
    out_img.save("examples/lena_vflip.png").unwrap();
}

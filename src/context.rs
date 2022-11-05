use ocl::{Context, Device, Queue};

pub struct ClContext {
    pub context: Context,
    pub device: Device,
    pub queue: Queue,
}

impl Default for ClContext {
    fn default() -> Self {
        let context = Context::builder()
            .devices(Device::specifier().first())
            .build()
            .unwrap();
        let device = context.devices()[0];
        let queue = Queue::new(&context, device, None).unwrap();
        ClContext {
            context,
            device,
            queue,
        }
    }
}

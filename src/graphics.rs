use crate::{
    shaders::{self, VertexInput, VertexInputs},
    DynResult,
};
use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::GpuFuture,
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::{Window, WindowBuilder},
};

pub struct Runtime {
    event_loop: EventLoop<()>,
    device: Arc<Device>,
    window: Arc<Window>,
    viewport: Viewport,
    swapchain: Arc<Swapchain>,
    frame_buffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[VertexInput]>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl Runtime {
    pub fn new() -> DynResult<Self> {
        let event_loop = EventLoop::new()?;

        let context = Self::load_library(&event_loop)?;
        let (window, surface) = Self::spawn_window(context.clone(), &event_loop)?;
        let (device, queue) = Self::select_device(context.clone(), surface.clone())?;

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())?;

            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            let composite_alpha = surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap();

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    ..Default::default()
                },
            )?
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertex_buffer = VertexInputs::new(vec![
            VertexInput::new([0.0, -0.5, 0.0], [1.0, 0.0, 0.0]),
            VertexInput::new([0.5, 0.5, 0.0], [0.0, 1.0, 0.0]),
            VertexInput::new([-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
        ])
        .into_buffer(memory_allocator.clone())?;

        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
               color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },

            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .unwrap();

        let pipeline = {
            let vertex_shader = shaders::vertex::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let fragment_shader = shaders::fragment::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state =
                VertexInput::per_vertex().definition(&vertex_shader.info().input_interface)?;
            // Position::per_vertex().definition(&vertex_shader.info().input_interface)?;
            let stages = [
                PipelineShaderStageCreateInfo::new(vertex_shader),
                PipelineShaderStageCreateInfo::new(fragment_shader),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };

        let frame_buffers =
            window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let recreate_swapchain = false;
        let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());

        Ok(Self {
            event_loop,
            device,
            window,
            viewport,
            swapchain,
            frame_buffers,
            command_buffer_allocator,
            queue,
            render_pass,
            pipeline,
            vertex_buffer,
            recreate_swapchain,
            previous_frame_end,
        })
    }

    fn spawn_window(
        context: Arc<Instance>,
        event_loop: &EventLoop<()>,
    ) -> DynResult<(Arc<Window>, Arc<Surface>)> {
        let window = Arc::new(WindowBuilder::new().build(event_loop)?);
        let surface = Surface::from_window(context.clone(), window.clone())?;

        Ok((window, surface))
    }

    pub fn load_library(event_loop: &EventLoop<()>) -> DynResult<Arc<Instance>> {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);

        Ok(Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?)
    }

    fn select_device(
        context: Arc<Instance>,
        surface: Arc<Surface>,
    ) -> DynResult<(Arc<Device>, Arc<Queue>)> {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = context
            .enumerate_physical_devices()?
            .map(|device| {
                println!(
                    "Discovered device: {} (type {:?})",
                    device.properties().device_name,
                    device.properties().device_type
                );

                device
            })
            .filter(|physical_device| {
                physical_device
                    .supported_extensions()
                    .contains(&device_extensions)
            })
            .filter_map(|physics_device| {
                physics_device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, family_properties)| {
                        family_properties
                            .queue_flags
                            .intersects(QueueFlags::GRAPHICS)
                            && physics_device
                                .surface_support(i as u32, &surface)
                                .unwrap_or(false)
                    })
                    .map(|i| (physics_device, i as u32))
            })
            .min_by_key(
                |(physics_device, _)| match physics_device.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                },
            )
            .unwrap();

        println!(
            "Selected device: {} (type {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        let queue = queues.next().unwrap();

        Ok((device, queue))
    }

    fn split_internal(self) -> (EventLoop<()>, RuntimeInternal) {
        let Runtime {
            event_loop,
            device,
            window,
            viewport,
            swapchain,
            frame_buffers,
            command_buffer_allocator,
            queue,
            render_pass,
            pipeline,
            vertex_buffer,
            recreate_swapchain,
            previous_frame_end,
        } = self;

        (
            event_loop,
            RuntimeInternal {
                device,
                window,
                viewport,
                swapchain,
                frame_buffers,
                command_buffer_allocator,
                queue,
                render_pass,
                pipeline,
                vertex_buffer,
                recreate_swapchain,
                previous_frame_end,
            },
        )
    }

    pub fn run(self) -> DynResult<()> {
        let (event_loop, mut internal) = self.split_internal();
        event_loop.run(move |event, window_target| internal.tick(event, window_target))?;

        Ok(())
    }
}

struct RuntimeInternal {
    device: Arc<Device>,
    window: Arc<Window>,
    viewport: Viewport,
    swapchain: Arc<Swapchain>,
    frame_buffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[VertexInput]>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl RuntimeInternal {
    pub fn tick(&mut self, event: Event<()>, window_target: &EventLoopWindowTarget<()>) {
        window_target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => todo!(),
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = self.window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if self.recreate_swapchain {
                    let (new_swapchain, new_images) = self
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..self.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    self.swapchain = new_swapchain;

                    self.frame_buffers = window_size_dependent_setup(
                        &new_images,
                        self.render_pass.clone(),
                        &mut self.viewport,
                    );

                    self.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(self.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            self.recreate_swapchain = true;

                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    self.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                self.frame_buffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [self.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(self.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .draw(self.vertex_buffer.len() as u32, 1, 0, 0) // We add a draw command.
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = self
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => self.previous_frame_end = Some(future.boxed()),

                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        self.previous_frame_end =
                            Some(vulkano::sync::now(self.device.clone()).boxed());
                    }
                    Err(error) => panic!("failed to flush future: {error}"),
                }
            }
            Event::AboutToWait => self.window.request_redraw(),
            _ => {}
        }
    }
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

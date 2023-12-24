use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
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
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod shaders;

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> DynResult<()> {
    let event_loop = EventLoop::new()?;

    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )?;

    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let surface = Surface::from_window(instance.clone(), window.clone())?;

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()?
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
        "Using device: {} (type {:?})",
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

    let (mut swapchain, images) = {
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

    // let position_buffer = VertexInputs::new(vec![
    //     Position::new(0.0, -0.5),
    //     Position::new(0.5, 0.5),
    //     Position::new(-0.5, 0.5),
    // ])
    // .as_buffer(memory_allocator.clone())?;

    // let color_buffer = VertexInputs::new(vec![
    //     Color::new(1.0, 0.0, 0.0),
    //     Color::new(0.0, 1.0, 0.0),
    //     Color::new(0.0, 0.0, 1.0),
    // ])
    // .as_buffer(memory_allocator.clone())?;

    let vertex_buffer = VertexInputs::new(vec![
        VertexInput::new([0.0, -0.5], [1.0, 0.0, 0.0]),
        VertexInput::new([0.5, 0.5], [0.0, 1.0, 0.0]),
        VertexInput::new([-0.5, 0.5], [0.0, 0.0, 1.0]),
    ])
    .as_buffer(memory_allocator.clone())?;

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

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());

    event_loop.run(move |event, window_target| {
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
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );

                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0) // We add a draw command.
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::AboutToWait => window.request_redraw(),
            Event::Suspended => {
                println!("suspended");
            }
            Event::LoopExiting => {
                println!("loop exiting");
            }
            _ => {}
        }
    })?;

    Ok(())
}

struct VertexInputs<T: BufferContents>(Vec<T>);

impl<T: BufferContents> VertexInputs<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self(values)
    }

    pub fn as_buffer(
        self,
        allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    ) -> DynResult<Subbuffer<[T]>> {
        Ok(Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.0,
        )?)
    }
}

// #[derive(BufferContents, Vertex)]
// #[repr(C)]
// struct Position {
//     #[format(R32G32_SFLOAT)]
//     position: [f32; 2],
// }

// impl Position {
//     fn new(x: f32, y: f32) -> Self {
//         Self { position: [x, y] }
//     }
// }

// #[derive(BufferContents, Vertex)]
// #[repr(C)]
// struct Color {
//     #[format(R32G32B32_SFLOAT)]
//     color: [f32; 3],
// }

// impl Color {
//     fn new(r: f32, g: f32, b: f32) -> Self {
//         Self { color: [r, g, b] }
//     }
// }

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertexInput {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

impl VertexInput {
    pub fn new(position: [f32; 2], color: [f32; 3]) -> Self {
        Self { position, color }
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

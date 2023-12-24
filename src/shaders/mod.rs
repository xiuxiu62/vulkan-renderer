use crate::DynResult;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    },
    pipeline::graphics::vertex_input::Vertex,
};

pub mod fragment;
pub mod vertex;

pub struct VertexInputs<T: BufferContents>(Vec<T>);

impl<T: BufferContents> VertexInputs<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self(values)
    }

    pub fn into_buffer(
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

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VertexInput {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

impl VertexInput {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, color }
    }
}

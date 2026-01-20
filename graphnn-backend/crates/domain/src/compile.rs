use burn::{prelude::Backend, tensor::Shape};

use crate::ops::Operation;

pub struct CompiledGraph<B: Backend> {
    pub nodes: Vec<Operation<B>>,
    pub mem_shapes: Vec<Shape>,
    pub param_shapes: Vec<Shape>,
    pub output_ids: Vec<usize>,
}

use burn::tensor::Shape;

#[derive(Debug)]
pub struct OpId(u64);

pub struct CompiledNode {
    pub op_id: OpId,
    pub input_ids: Vec<usize>,
    pub param_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
}

pub struct CompiledGraph {
    pub nodes: Vec<CompiledNode>,
    pub mem_shapes: Vec<Shape>,
    pub param_shapes: Vec<Shape>,
    pub output_ids: Vec<usize>,
}

impl CompiledGraph {
    pub fn mem_size(&self) -> usize {
        self.mem_shapes.len()
    }

    pub fn param_size(&self) -> usize {
        self.param_shapes.len()
    }
}

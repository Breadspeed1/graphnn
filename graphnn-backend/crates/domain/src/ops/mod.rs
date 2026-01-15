use burn::{prelude::Backend, tensor::Shape};

use crate::tensor::TensorAny;

#[derive(Debug)]
pub enum OpError {
    CannotInferShapes(&'static str),
}

pub trait Op<B: Backend> {
    fn name(&self) -> &'static str;

    fn arity(&self) -> (usize, usize);

    fn init_params(&self, device: &B::Device) -> Vec<TensorAny<B>>;

    fn infer_shapes(
        &self,
        input_shapes: &[Shape],
        param_shapes: &[Shape],
    ) -> Result<Vec<Shape>, OpError>;

    fn evaluate(&self, inputs: &[TensorAny<B>], params: &[TensorAny<B>]) -> Vec<TensorAny<B>>;
}

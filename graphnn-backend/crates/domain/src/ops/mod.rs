use std::marker::PhantomData;

use burn::{prelude::Backend, tensor::Shape};
use enum_dispatch::enum_dispatch;

use crate::tensor::TensorAny;

pub struct OpArity {
    inputs: u32,
    outputs: u32,
    params: u32,
}

#[derive(Debug)]
pub enum OpError {
    CannotInferShapes(&'static str),
}

pub struct NoOp<B: Backend>(PhantomData<B>);

#[enum_dispatch]
pub trait Op<B: Backend> {
    fn name(&self) -> &'static str;

    fn arity(&self) -> OpArity;

    fn init_params(&self, device: &B::Device, input_shapes: &[Shape]) -> Vec<TensorAny<B>>;

    fn infer_shapes(&self, input_shapes: &[Shape]) -> Result<Vec<Shape>, OpError>;

    fn evaluate(&self, inputs: &[TensorAny<B>], params: &[TensorAny<B>]) -> Vec<TensorAny<B>>;
}

impl<B: Backend> Op<B> for NoOp<B> {
    fn name(&self) -> &'static str {
        "No Operation"
    }

    fn arity(&self) -> OpArity {
        OpArity {
            inputs: 1,
            outputs: 1,
            params: 0,
        }
    }

    fn init_params(
        &self,
        _device: &<B as Backend>::Device,
        _input_shapes: &[Shape],
    ) -> Vec<TensorAny<B>> {
        vec![]
    }

    fn infer_shapes(&self, input_shapes: &[Shape]) -> Result<Vec<Shape>, OpError> {
        Ok(vec![input_shapes[0].clone()])
    }

    fn evaluate(&self, inputs: &[TensorAny<B>], _params: &[TensorAny<B>]) -> Vec<TensorAny<B>> {
        vec![inputs[0].clone()]
    }
}

#[enum_dispatch(Op<B>)]
pub enum Operation<B: Backend> {
    NoOp(NoOp<B>),
}

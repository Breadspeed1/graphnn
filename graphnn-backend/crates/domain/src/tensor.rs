use burn::{Tensor, prelude::Backend};

#[derive(Debug)]
pub enum TensorTypeError {
    WrongRank { expected: usize, got: usize },
}

#[derive(Debug)]
pub enum ArenaError {
    NoUsesLeft(usize),
    SlotEmpty(usize),
}

#[derive(Clone)]
pub enum TensorAny<B: Backend> {
    D1(Tensor<B, 1>),
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
    D4(Tensor<B, 4>),
}

pub struct Arena<B: Backend> {
    mem: Vec<Option<TensorAny<B>>>,
    uses_left: Vec<u32>,
}

impl<B: Backend> Arena<B> {
    pub fn new(slots: usize, use_count: Vec<u32>) -> Self {
        Self {
            mem: vec![None; slots],
            uses_left: use_count,
        }
    }

    pub fn read(&mut self, slot: usize) -> Result<TensorAny<B>, ArenaError> {
        let left = self.uses_left[slot];

        if left <= 0 {
            return Err(ArenaError::NoUsesLeft(slot));
        }

        self.uses_left[slot] -= 1;

        if left == 1 {
            self.mem[slot].take().ok_or(ArenaError::SlotEmpty(slot))
        } else {
            self.mem[slot]
                .as_ref()
                .cloned()
                .ok_or(ArenaError::SlotEmpty(slot))
        }
    }

    pub fn write(&mut self, slot: usize, value: TensorAny<B>) {
        self.mem[slot] = Some(value);
    }
}

impl<B: Backend> TensorAny<B> {
    pub fn rank(&self) -> usize {
        match self {
            TensorAny::D1(_) => 1,
            TensorAny::D2(_) => 2,
            TensorAny::D3(_) => 3,
            TensorAny::D4(_) => 4,
        }
    }

    pub fn expect_d2(self) -> Result<Tensor<B, 2>, TensorTypeError> {
        match self {
            TensorAny::D2(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 2,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_ref_d2(&self) -> Result<&Tensor<B, 2>, TensorTypeError> {
        match self {
            TensorAny::D2(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 2,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_d1(self) -> Result<Tensor<B, 1>, TensorTypeError> {
        match self {
            TensorAny::D1(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 1,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_ref_d1(&self) -> Result<&Tensor<B, 1>, TensorTypeError> {
        match self {
            TensorAny::D1(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 1,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_d3(self) -> Result<Tensor<B, 3>, TensorTypeError> {
        match self {
            TensorAny::D3(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 3,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_ref_d3(&self) -> Result<&Tensor<B, 3>, TensorTypeError> {
        match self {
            TensorAny::D3(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 3,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_d4(self) -> Result<Tensor<B, 4>, TensorTypeError> {
        match self {
            TensorAny::D4(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 4,
                got: other.rank(),
            }),
        }
    }

    pub fn expect_ref_d4(&self) -> Result<&Tensor<B, 4>, TensorTypeError> {
        match self {
            TensorAny::D4(t) => Ok(t),
            other => Err(TensorTypeError::WrongRank {
                expected: 4,
                got: other.rank(),
            }),
        }
    }
}

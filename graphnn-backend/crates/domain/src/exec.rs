use burn::prelude::Backend;

use crate::tensor::TensorAny;

#[derive(Debug)]
pub enum ArenaError {
    NoUsesLeft(usize),
    SlotEmpty(usize),
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

use burn::prelude::Backend;

use crate::{
    compile::{CompiledGraph, CompiledNode},
    ops::Op,
    tensor::TensorAny,
};

#[derive(Debug)]
pub enum ArenaError {
    NoUsesLeft(usize),
    SlotEmpty(usize),
    SlotDone(usize),
}

#[derive(Debug)]
pub enum MemError {
    NoUsesLeft,
    Empty,
    CellDone,
}

pub struct MemSlot<B: Backend> {
    data: Option<TensorAny<B>>,

    /// uses until free for each use of this slot
    use_counts: Vec<usize>,
    use_pointer: usize,
}

pub struct Arena<B: Backend> {
    mem: Vec<MemSlot<B>>,
}

pub struct Executor<B: Backend> {
    mem: Arena<B>,
    graph: CompiledGraph,
    ops: Vec<Box<dyn Op<B>>>,
}

impl ArenaError {
    pub fn from_mem_error(err: MemError, slot: usize) -> Self {
        match err {
            MemError::NoUsesLeft => Self::NoUsesLeft(slot),
            MemError::Empty => Self::SlotEmpty(slot),
            MemError::CellDone => Self::SlotDone(slot),
        }
    }
}

impl<B: Backend> MemSlot<B> {
    pub fn new(use_counts: Vec<usize>) -> Self {
        Self {
            use_counts,
            data: None,
            use_pointer: 0,
        }
    }

    pub fn read(&mut self) -> Result<TensorAny<B>, MemError> {
        if self.use_pointer >= self.use_counts.len() {
            return Err(MemError::CellDone);
        }

        let reads_left = self.use_counts[self.use_pointer];

        match reads_left {
            0 => Err(MemError::NoUsesLeft),
            1 => {
                self.use_counts[self.use_pointer] -= 1;
                self.use_pointer += 1;

                self.data.take().ok_or(MemError::Empty)
            }
            2.. => self.data.as_ref().cloned().ok_or(MemError::Empty),
        }
    }

    pub fn write(&mut self, data: TensorAny<B>) {
        self.data = Some(data);
    }
}

impl<B: Backend> Arena<B> {
    pub fn new(use_count: Vec<Vec<usize>>) -> Self {
        Self {
            mem: use_count
                .into_iter()
                .map(|uses| MemSlot::new(uses))
                .collect(),
        }
    }

    pub fn read(&mut self, slot: usize) -> Result<TensorAny<B>, ArenaError> {
        self.mem[slot]
            .read()
            .map_err(|e| ArenaError::from_mem_error(e, slot))
    }

    pub fn read_all(&mut self, slots: &[usize]) -> Result<Vec<TensorAny<B>>, ArenaError> {
        slots.iter().map(|&s| self.read(s)).collect()
    }

    pub fn write_all(&mut self, slots: &[usize], values: Vec<TensorAny<B>>) {
        slots
            .iter()
            .zip(values)
            .for_each(|(&slot, value)| self.write(slot, value));
    }

    pub fn write(&mut self, slot: usize, data: TensorAny<B>) {
        self.mem[slot].write(data);
    }
}

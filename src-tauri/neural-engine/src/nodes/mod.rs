use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::Arc,
};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use dummy::DummyNode;

pub mod dummy;

#[derive(Serialize, Deserialize)]
pub struct GraphBlueprint {
    pub nodes: HashMap<Id, NodeConfig>,
    pub edges: Vec<Edge>,
}

//TODO: make this an enum dispatch to a config trait
#[enum_dispatch(Config)]
#[derive(Serialize, Deserialize, Clone)]
pub enum NodeConfig {
    DummyNode,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct Id(Arc<str>);

#[derive(Serialize, Deserialize, Clone)]
pub struct Edge {
    source: HandleRef,
    target: HandleRef,
    id: Id,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HandleRef {
    node_id: Id,
    handle_id: Id,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum DataType {
    IntTensor,
    FloatTensor,
    Int,
    Float,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HandleDef {
    pub id: Id,
    pub dtype: DataType,
}

enum GraphBlueprintError {
    NodeExists(Id),
    NodeDoesNotExist(Id),
}

#[enum_dispatch]
trait Config {
    fn in_handles(&self) -> Vec<HandleDef>;
    fn out_handles(&self) -> Vec<HandleDef>;
}

impl Deref for Id {
    type Target = Arc<str>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GraphBlueprint {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    fn add_node(&mut self, id: Id, config: NodeConfig) -> Result<(), GraphBlueprintError> {
        if self.nodes.contains_key(&id) {
            return Err(GraphBlueprintError::NodeExists(id));
        }

        self.nodes.insert(id, config);

        Ok(())
    }

    fn remove_node(&mut self, id: Id) -> Result<(), GraphBlueprintError> {
        let result = self.nodes.remove(&id);

        match result {
            Some(_) => Ok(()),
            None => Err(GraphBlueprintError::NodeDoesNotExist(id)),
        }
    }

    fn connect_nodes(
        &mut self,
        source: HandleRef,
        target: HandleRef,
    ) -> Result<(), GraphBlueprintError> {
        if !self.nodes.contains_key(&source.node_id) {
            return Err(GraphBlueprintError::NodeDoesNotExist(source.node_id));
        }

        if !self.nodes.contains_key(&target.node_id) {
            return Err(GraphBlueprintError::NodeDoesNotExist(target.node_id));
        }

        // check if handles exist
        // check if edge already exists

        todo!()
    }
}

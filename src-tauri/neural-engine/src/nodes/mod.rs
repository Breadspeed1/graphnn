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

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
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
    HandleDoesNotExist(HandleRef),
    HandleMismatch(HandleRef, HandleRef),
    EdgeExists(Id),
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
        id: Id,
    ) -> Result<(), GraphBlueprintError> {
        if self.edges.iter().any(|e| e.id == id) {
            return Err(GraphBlueprintError::EdgeExists(id));
        }

        match (
            self.nodes.get(&source.node_id),
            self.nodes.get(&target.node_id),
        ) {
            (None, _) => Err(GraphBlueprintError::NodeDoesNotExist(source.node_id)),
            (_, None) => Err(GraphBlueprintError::NodeDoesNotExist(target.node_id)),
            (Some(source_node), Some(target_node)) => match (
                source_node
                    .out_handles()
                    .iter()
                    .find(|h| h.id == source.handle_id),
                target_node
                    .out_handles()
                    .iter()
                    .find(|h| h.id == target.handle_id),
            ) {
                (None, _) => Err(GraphBlueprintError::HandleDoesNotExist(source)),
                (_, None) => Err(GraphBlueprintError::HandleDoesNotExist(target)),
                (Some(source_handle), Some(target_handle)) => {
                    if source_handle.dtype == target_handle.dtype {
                        self.edges.push(Edge { source, target, id });
                        Ok(())
                    } else {
                        Err(GraphBlueprintError::HandleMismatch(source, target))
                    }
                }
            },
        }
    }
}

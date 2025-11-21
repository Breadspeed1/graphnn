use std::{collections::HashMap, ops::Deref, sync::Arc};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use dummy::DummyNode;

pub mod dummy;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphBlueprint {
    pub nodes: HashMap<Id, NodeConfig>,
    pub edges: Vec<Edge>,
}

//TODO: make this an enum dispatch to a config trait
#[enum_dispatch(Config)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NodeConfig {
    DummyNode,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Id(Arc<str>);

impl<T: Into<Arc<str>>> From<T> for Id {
    fn from(s: T) -> Self {
        Id(s.into())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Edge {
    source: HandleRef,
    target: HandleRef,
    id: Id,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HandleRef {
    node_id: Id,
    handle_id: Id,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    IntTensor,
    FloatTensor,
    Int,
    Float,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HandleDef {
    pub id: Id,
    pub dtype: DataType,
}

#[derive(Debug, PartialEq)]
pub enum GraphBlueprintError {
    NodeExists(Id),
    NodeDoesNotExist(Id),
    HandleDoesNotExist(HandleRef),
    HandleMismatch(HandleRef, HandleRef),
    EdgeExists(Id),
    EdgeDoesNotExist(Id),
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

    fn disconnect(&mut self, id: Id) -> Result<(), GraphBlueprintError> {
        if let Some(idx) = self.edges.iter().position(|e| e.id == id) {
            self.edges.remove(idx);
            Ok(())
        } else {
            Err(GraphBlueprintError::EdgeDoesNotExist(id))
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
                    .in_handles()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_node() {
        let mut graph = GraphBlueprint::new();
        let id = Id::from("node1");
        let config = NodeConfig::DummyNode(DummyNode);

        assert!(graph.add_node(id.clone(), config.clone()).is_ok());
        assert!(graph.nodes.contains_key(&id));

        // Test duplicate add
        assert_eq!(
            graph.add_node(id.clone(), config),
            Err(GraphBlueprintError::NodeExists(id.clone()))
        );

        assert!(graph.remove_node(id.clone()).is_ok());
        assert!(!graph.nodes.contains_key(&id));

        // Test remove non-existent
        assert_eq!(
            graph.remove_node(id.clone()),
            Err(GraphBlueprintError::NodeDoesNotExist(id))
        );
    }

    #[test]
    fn test_connect_nodes_success() {
        let mut graph = GraphBlueprint::new();
        let node1_id = Id::from("node1");
        let node2_id = Id::from("node2");

        graph
            .add_node(node1_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();
        graph
            .add_node(node2_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();

        let source = HandleRef {
            node_id: node1_id.clone(),
            handle_id: Id::from("out_1"),
        };
        let target = HandleRef {
            node_id: node2_id.clone(),
            handle_id: Id::from("in_1"),
        };
        let edge_id = Id::from("edge1");

        assert!(
            graph
                .connect_nodes(source.clone(), target.clone(), edge_id.clone())
                .is_ok()
        );
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].id, edge_id);
    }

    #[test]
    fn test_connect_nodes_failures() {
        let mut graph = GraphBlueprint::new();
        let node1_id = Id::from("node1");
        let node2_id = Id::from("node2");

        graph
            .add_node(node1_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();
        // node2 not added

        let source = HandleRef {
            node_id: node1_id.clone(),
            handle_id: Id::from("out_1"),
        };
        let target = HandleRef {
            node_id: node2_id.clone(),
            handle_id: Id::from("in_1"),
        };
        let edge_id = Id::from("edge1");

        // NodeDoesNotExist (target)
        assert_eq!(
            graph.connect_nodes(source.clone(), target.clone(), edge_id.clone()),
            Err(GraphBlueprintError::NodeDoesNotExist(node2_id.clone()))
        );

        graph
            .add_node(node2_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();

        // HandleDoesNotExist (source handle wrong)
        let bad_source = HandleRef {
            node_id: node1_id.clone(),
            handle_id: Id::from("wrong_handle"),
        };
        assert_eq!(
            graph.connect_nodes(bad_source.clone(), target.clone(), edge_id.clone()),
            Err(GraphBlueprintError::HandleDoesNotExist(bad_source))
        );

        // HandleDoesNotExist (target handle wrong)
        let bad_target = HandleRef {
            node_id: node2_id.clone(),
            handle_id: Id::from("wrong_handle"),
        };
        assert_eq!(
            graph.connect_nodes(source.clone(), bad_target.clone(), edge_id.clone()),
            Err(GraphBlueprintError::HandleDoesNotExist(bad_target))
        );

        // EdgeExists
        graph
            .connect_nodes(source.clone(), target.clone(), edge_id.clone())
            .unwrap();
        assert_eq!(
            graph.connect_nodes(source.clone(), target.clone(), edge_id.clone()),
            Err(GraphBlueprintError::EdgeExists(edge_id.clone()))
        );
    }

    #[test]
    fn test_disconnect() {
        let mut graph = GraphBlueprint::new();
        let node1_id = Id::from("node1");
        let node2_id = Id::from("node2");

        graph
            .add_node(node1_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();
        graph
            .add_node(node2_id.clone(), NodeConfig::DummyNode(DummyNode))
            .unwrap();

        let source = HandleRef {
            node_id: node1_id.clone(),
            handle_id: Id::from("out_1"),
        };
        let target = HandleRef {
            node_id: node2_id.clone(),
            handle_id: Id::from("in_1"),
        };
        let edge_id = Id::from("edge1");

        graph
            .connect_nodes(source, target, edge_id.clone())
            .unwrap();
        assert_eq!(graph.edges.len(), 1);

        assert!(graph.disconnect(edge_id.clone()).is_ok());
        assert_eq!(graph.edges.len(), 0);

        // Disconnect non-existent
        assert_eq!(
            graph.disconnect(edge_id.clone()),
            Err(GraphBlueprintError::EdgeDoesNotExist(edge_id))
        );
    }
}

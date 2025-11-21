use std::{
    collections::{HashMap, VecDeque},
    ops::Deref,
    sync::Arc,
};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use dummy::DummyNode;

use crate::nodes::lint::LintOutput;

pub mod dummy;
pub mod lint;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphBlueprint {
    pub nodes: HashMap<Id, NodeConfig>,
    pub edges: Vec<Edge>,
}

pub struct SortedGraphBlueprint {
    pub nodes: Vec<(Id, NodeConfig)>,
    pub edges: Vec<(Edge, TensorShape)>,
    pub lint: LintOutput,
}

#[enum_dispatch(Config)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NodeConfig {
    Dummy(DummyNode),
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
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum RankConstraint {
    Any,
    Fixed(usize),
    MatchHandle(Id),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorShape(Arc<[usize]>);

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
    UnableToInferTensorShape(HandleRef),
}

#[derive(Debug, PartialEq)]
pub enum ConfigError {
    MissingInputShape(Id),
}

#[enum_dispatch]
trait Config {
    fn in_handles(&self) -> Vec<HandleDef>;
    fn out_handles(&self) -> Vec<HandleDef>;
    fn infer_output_shapes(
        &self,
        input_shapes: &HashMap<Id, TensorShape>,
    ) -> Result<HashMap<Id, TensorShape>, ConfigError>;
}

impl Deref for Id {
    type Target = Arc<str>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GraphBlueprint {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, id: Id, config: NodeConfig) -> Result<(), GraphBlueprintError> {
        if self.nodes.contains_key(&id) {
            return Err(GraphBlueprintError::NodeExists(id));
        }

        self.nodes.insert(id, config);

        Ok(())
    }

    pub fn remove_node(&mut self, id: Id) -> Result<(), GraphBlueprintError> {
        let result = self.nodes.remove(&id);

        match result {
            Some(_) => Ok(()),
            None => Err(GraphBlueprintError::NodeDoesNotExist(id)),
        }
    }

    pub fn disconnect(&mut self, id: Id) -> Result<(), GraphBlueprintError> {
        if let Some(idx) = self.edges.iter().position(|e| e.id == id) {
            self.edges.remove(idx);
            Ok(())
        } else {
            Err(GraphBlueprintError::EdgeDoesNotExist(id))
        }
    }

    pub fn connect_nodes(
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

    pub fn sort(&self) -> SortedGraphBlueprint {
        let mut sorted_nodes = Vec::new();
        let mut lint = LintOutput::default();

        let mut adj: HashMap<Id, Vec<&Edge>> = HashMap::new();
        let mut in_degree: HashMap<Id, usize> = HashMap::new();

        for id in self.nodes.keys() {
            in_degree.insert(id.clone(), 0);
            adj.insert(id.clone(), Vec::new());
        }

        for edge in &self.edges {
            if let Some(list) = adj.get_mut(&edge.source.node_id) {
                list.push(edge);
            }
            if let Some(count) = in_degree.get_mut(&edge.target.node_id) {
                *count += 1;
            }
        }

        let mut queue: VecDeque<Id> = in_degree
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(id, _)| id.clone())
            .collect();

        while let Some(node_id) = queue.pop_front() {
            if let Some(config) = self.nodes.get(&node_id) {
                sorted_nodes.push((node_id.clone(), config.clone()));
            }

            if let Some(outgoing_edges) = adj.get(&node_id) {
                for edge in outgoing_edges {
                    let target = &edge.target.node_id;

                    if let Some(count) = in_degree.get_mut(target) {
                        *count -= 1;
                        // If this was the last dependency, add to queue
                        if *count == 0 {
                            queue.push_back(target.clone());
                        }
                    }
                }
            }
        }

        if sorted_nodes.len() != self.nodes.len() {
            let cycle_edges: Vec<Id> = self
                .edges
                .iter()
                .filter(|e| *in_degree.get(&e.target.node_id).unwrap_or(&0) > 0)
                .map(|e| e.id.clone())
                .collect();

            lint.push_error(lint::LintError::Cycle { edges: cycle_edges });
        }

        SortedGraphBlueprint {
            nodes: sorted_nodes,
            edges: todo!(),
            lint,
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
        let config = NodeConfig::Dummy(DummyNode);

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
            .add_node(node1_id.clone(), NodeConfig::Dummy(DummyNode))
            .unwrap();
        graph
            .add_node(node2_id.clone(), NodeConfig::Dummy(DummyNode))
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
            .add_node(node1_id.clone(), NodeConfig::Dummy(DummyNode))
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
            .add_node(node2_id.clone(), NodeConfig::Dummy(DummyNode))
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
            .add_node(node1_id.clone(), NodeConfig::Dummy(DummyNode))
            .unwrap();
        graph
            .add_node(node2_id.clone(), NodeConfig::Dummy(DummyNode))
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

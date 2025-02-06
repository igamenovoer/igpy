from .Snapshot import SnapshotNode
from typing import Tuple, Iterable
import numpy as np
import networkx as nx
from enum import IntEnum

class LabelAssignment:
    def __init__(self) -> None:
        # this label is assigned due to which node? (consider parent assign label to child)
        self.m_src_node : SnapshotNode = None
        
        # the label is assigned to this node
        self.m_node : SnapshotNode = None
        self.m_label : LabelDef = None # the label
        
    @property
    def assign_by_who(self) -> SnapshotNode:
        return self.m_src_node
    
    @property
    def label(self) -> "LabelDef":
        return self.m_label
    
    @property
    def node(self) -> SnapshotNode:
        return self.m_node
    
    @property
    def include_children(self) -> bool:
        ''' this assignment is also applied to the children of the node?
        '''
        if self.m_label:
            return self.m_label.include_children
        else:
            return False

class LabelerBase:
    ''' responsible for assigning a label to an object
    '''
    
    def get_label_assignments(self, node : SnapshotNode) -> list[LabelAssignment]:
        ''' get the label assignments related to a node
        '''
        return NotImplementedError()
    
    @property
    def background_label(self) -> "LabelDef":
        ''' the label for background objects
        
        return
        ---------
        lb : LabelDef
            the background label
        '''
        raise NotImplementedError()
    
class LabelDef:
    ''' defines a semantic label
    '''
    def __init__(self) -> None:
        self.m_name : str = None    # unique name of the label
        self.m_code : int = None    # non-unique code of the label
        
        # defines precedence of labels when multiple labels are assigned to one object
        # the label with lower rank value has higher priority, and it takes precedence.
        self.m_rank : int = np.iinfo(int).max   # by default, use lowest priority
        
        # this label definition also applies to the children nodes
        self.m_include_children : bool = False
        
    @property
    def name(self)->str:
        ''' name of the label, must be unique
        '''
        return self.m_name
    
    @property
    def code(self)->int:
        ''' code of the label, non-unique, different labels can have the same code, designating these
        labels are considered having the same semantics
        '''
        return self.m_code
    
    @property
    def rank(self)->int:
        return self.m_rank
    
    @property
    def include_children(self) -> bool:
        ''' the label definition also applicable to children?
        '''
        return self.m_include_children
    
    def __repr__(self) -> str:
        return 'LabelDef(name={}, code={})'.format(self.name, self.code)
    
class LabelPrecedenceGraph:
    ''' A graph that defines partial ordering of all labels.
    An edge between label u->v means label u has high priority over v
    '''
    NodeKey = "label_def"
        
    def __init__(self) -> None:
        self.m_graph = nx.DiGraph()
        
    @property
    def graph(self) -> nx.DiGraph:
        return self.m_graph
    
    # def sort_label_by_precedency(self, labels : list[LabelDef], strict : bool = False):
    #     ''' sort the labels based on the partial ordering in the graph, from high to low in terms of precedence
        
    #     return
    #     ----------
    #     labels
    #         a list of labels to sort
    #     strict
    #         if True, 
        
    #     '''
    
    def is_precedent(self, u:str, v:str, strict : bool = False) -> bool:
        ''' check if u>v in terms of label precedency. Note that if u and v are unrelated,
        return arbitrary True/False, unless strict=True.
        
        parameters
        -------------
        u, v
            label names.
        strict : bool
            If True, guarantee to return False if u and v are not related.
            
        returns
        ----------
        u_pred_v
            True if u>v in terms of label precedency, otherwise False.
        '''
        g = self.m_graph
        NodeKey = self.__class__.NodeKey
        if strict:
            # there must be a path from u to v in order for u>v
            u_to_v : bool = nx.has_path(g, u, v)
            return u_to_v
        else:
            # just check rank. If u,v not connected, the result is meaning less
            u_label : LabelDef = g.nodes[u].get(NodeKey)
            v_label : LabelDef = g.nodes[v].get(NodeKey)
            return u_label.rank < v_label.rank
            
    def _update_rank(self, src_node_name : str):
        ''' update all ranks from a source. 
        The source itself must have rank value assigned
        '''
        g = self.m_graph
        NodeKey = self.__class__.NodeKey
        src : LabelDef = g.nodes[src_node_name][NodeKey]
        assert src.rank is not None, 'source node must have rank value assigned'
        
        for parent, children in nx.bfs_successors(g, src_node_name):
            # early return: has rank updated in children ? if no update is applicable, we are done
            has_update : bool = False
            
            # update ranks in children
            p_label : LabelDef = g.nodes[parent].get(NodeKey)
            new_rank = p_label.rank + 1
            for c in children:
                c_label : LabelDef = g.nodes[c].get(NodeKey)
                if c_label.rank is None or c_label.rank < new_rank:
                    c_label.m_rank = new_rank
                    has_update = True
            
            if not has_update:
                break
                
    def add_label(self, lb : LabelDef):
        ''' add a label to the precedence graph. The label must have unique name.
        '''
        self.m_graph.add_node(lb, {__class__.NodeKey : lb})
    
    def add_precedence_by_pair(self, u : str, v : str):
        ''' define precedence relationship between two labels, such that u takes precedence over v.
        
        parameters
        -------------
        u, v
            label names, such that in terms of precedence, label u > label v
        '''
        NodeKey = self.__class__.NodeKey
        g = self.m_graph
        
        assert u in g.nodes and v in g.nodes, 'u or v is not in the graph, use add_label() first'
        g.add_edge(u, v)
        
        # all nodes in the precedence graph must have rank assigned
        # update ranks
        u_node : LabelDef = g.nodes[u].get(NodeKey)
        if u_node.rank is None:
            u_node.m_rank = 0
        self._update_rank(u)
        
    def add_precedence_by_ordering(self, labels : list[str]):
        ''' define precedence relationship among N labels, such that labels[i] takes precedence over labels[i+1].
        
        parameters
        ------------
        labels
            a list of label names, such that in terms of precedence, labels[i]>labels[i+1]
        '''
        NodeKey = self.__class__.NodeKey
        g = self.m_graph
        
        # check
        for name in labels:
            assert name in g.nodes, '{} is not in graph, call add_label first'.format(name)
            
        # define edges
        for i in range(len(labels)-1):
            g.add_edge(labels[i], labels[i+1])
        root_node : LabelDef = g.nodes[labels[0]].get(NodeKey)
        if root_node.rank is None:
            root_node.m_rank = 0
        self._update_rank(root_node.name)
            
class AssignmentFilter:
    ''' functions to select assignments for each node
    '''
    def __init__(self) -> None:
        self.m_node2assignments : dict[SnapshotNode, list[LabelAssignment]] = None
    
    @staticmethod
    def create_from_assignments(assignments : dict[SnapshotNode, list[LabelAssignment]]) -> "AssignmentFilter":
        out = AssignmentFilter()
        out.m_node2assignments = assignments
        return out
        
    def get_label_by_highest_priority(self) -> dict[SnapshotNode, LabelDef]:
        ''' for each snapshot node, select the assignment with lowest rank (highest priority)
        '''
        output = {}
        for node, asnlist in self.m_node2assignments.items():
            asn = min(asnlist, key=lambda x : x.label.rank)
            output[node] = asn.label
        return output
    
    def get_label_by_root(self) -> dict[SnapshotNode, LabelDef]:
        ''' for each node, select the assignment assigned by the root node in hierarchy
        '''
        output = {}
        for node, asnlist in self.m_node2assignments.items():
            asn = min(asnlist, key= lambda x : (x.assign_by_who.depth_in_graph, x.label.rank))
            output[node] = asn.label
        return output
    
    def get_label_by_leaf(self) -> dict[SnapshotNode, LabelDef]:
        ''' for each node, select the assignment assigned by the node closest to it
        in terms for graph depth
        '''
        output = {}
        for node, asnlist in self.m_node2assignments.items():
            asn = min(asnlist, key= lambda x : (-x.assign_by_who.depth_in_graph, x.label.rank))
            output[node] = asn.label
        return output


class LabelPreference(IntEnum):
    ''' how to resolve labeling when multiple labels are assigned to one object
    '''
    MaxPriority = 1    # label with lowest rank (highest priority)
    ParentFirst = 2   # prefer root in hierarchy
    ChildFirst = 3   # prefer leaf in hierarchy
####
DefaultBackgroundLabel = LabelDef()
DefaultBackgroundLabel.m_code = 0
DefaultBackgroundLabel.m_name = '__default_background'
DefaultBackgroundLabel.m_rank = np.iinfo(int).max
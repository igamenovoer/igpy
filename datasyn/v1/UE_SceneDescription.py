import numpy as np
from typing import Union, Tuple, Callable
import networkx as nx
from .Constants import \
    UE_SceneDescKeys, UE_ClassName, UE_MetaClassType
from .SceneDescription import SceneDescription
import re

class UE_DescNodeSelector:
    ''' information to select nodes by pattern match. 
    Given a selector, the target must match all given patterns in order to be selected.
    A default selector will match any node. 
    
    A node will be selected if it satisfies ALL pattern matching requirements.
    As such, individual patterns and combined by logical "AND".
    '''
    def __init__(self) -> None:
        self.m_name_ptn : re.Pattern = None
        self.m_name_str : str = None
        self.m_name_inv_select : bool = False # if True, select unmatched entry
        
        self.m_asset_path_ptn : re.Pattern = None
        self.m_asset_path_str : str = None
        self.m_asset_path_inv_select : bool = False # if True, select unmatched entry
        
        # if True, all strings will be converted to lower case
        self.m_case_sensitive: bool = False
        
        # if True, the final selection will be inverted
        self.m_inv_select: bool = False   
        
    def _make_pattern(self, ptn_string : str) -> re.Pattern:
        if ptn_string is None:
            return None
        elif self.m_case_sensitive:
            return re.compile(ptn_string)
        else:
            return re.compile(ptn_string.lower())
        
    def _unify_case(self, s : str) -> str:
        if self.m_case_sensitive:
            return s
        else:
            return s.lower()
            
    def set_name_pattern(self, ptn_string : str, inv_select : bool = False):
        ''' name pattern to include or exclude
        '''
        self.m_name_str = ptn_string
        self.m_name_ptn = self._make_pattern(ptn_string)
        self.m_name_inv_select = inv_select
        
    def set_asset_path_pattern(self, ptn_string:str, inv_select : bool = False):
        ''' pattern to filter by asset path
        '''
        self.m_asset_path_str = ptn_string
        self.m_asset_path_ptn = self._make_pattern(ptn_string)
        self.m_asset_path_inv_select = inv_select
        
    def set_case_sensitive(self, case_sensitive : bool):
        if case_sensitive == self.m_case_sensitive:
            return
        
        self.m_case_sensitive = case_sensitive
        self.m_name_ptn = self._make_pattern(self.m_name_str)
        self.m_asset_path_ptn = self._make_pattern(self.m_asset_path_str)
        
    def __call__(self, node : "UE_SceneDescNode") -> bool:
        ''' given the patterns, check if a node should be selected or not
        '''
        return self.match(node)
        
    def match(self, node : "UE_SceneDescNode") -> bool:
        ''' given the patterns, check if a node should be selected or not
        '''
            
        ok : bool = True
        if self.m_name_ptn:
            m = self.m_name_ptn.match(self._unify_case(node.name))
            if self.m_name_inv_select:
                ok &= ~bool(m)
            else:
                ok &= bool(m)
            
        if self.m_asset_path_ptn:
            p = node.get_asset_path()
            if p is not None:
                m = self.m_asset_path_ptn.match(self._unify_case(p))
                if self.m_asset_path_inv_select:
                    ok &= ~bool(m)
                else:
                    ok &= bool(m)
                
        if self.m_inv_select:
            ok = ~ok
        
        return ok

class UE_SceneDescNode:
    ''' represent an object defined by scene description
    '''
    def __init__(self) -> None:
        self.m_name : str = None   # display name in ue, unique for each scene object
        self.m_content : dict[str, Union[str, dict[str,str]]] = None    # descriptive contents
        self.m_owner : UE_SceneDescription = None
        self.m_meta_class_type : str = None # unreal meta class type
        
    @property
    def name(self) -> str:
        return self.m_name
    
    @property
    def content(self):
        return self.m_content
    
    @property
    def meta_class(self)->str:
        return self.m_meta_class_type
    
    @property
    def owner(self) -> "UE_SceneDescription":
        return self.m_owner
    
    def __repr__(self) -> str:
        return 'UE_DescNode(name={}, metaclass={})'.format(self.name, self.meta_class)
    
    def get_parent_node(self):
        ''' get parent of this node, None if no parent exists
        '''
        g = self.m_owner.graph
        nodekey = self.m_owner.NodeKey
        parent : UE_SceneDescNode = None
        for p in g.predecessors(self.m_name):
            parent = g.nodes[p].get(nodekey)
        return parent
    
    def get_direct_children(self) -> list["UE_SceneDescNode"]:
        ''' get a list of direct children of this node
        '''
        g = self.m_owner.graph
        nodekey = self.m_owner.NodeKey
        output : list[UE_SceneDescNode] = []
        for p in g.successors(self.m_name):
            output.append(g.nodes[p].get(nodekey))
            
        return output
    
    def get_parent_actor_name(self) -> str:
        ''' get the name of parent actor. If not exists, return None
        '''
        return self.m_content.get(UE_SceneDescKeys.ParentActor)
    
    def get_class_name(self) -> str:
        return self.m_content.get(UE_SceneDescKeys.UEClassName)
    
    def get_root_geometry_node_name(self) -> str:
        ''' get the geometry node name of the root component. None if not exists.
        '''
        name, root_comp = self.get_root_component()
        if name is None:
            return None
        return root_comp.get(UE_SceneDescKeys.GLTFNode)
    
    def get_root_component(self) -> Tuple[str, dict[str,str]]:
        ''' get root component information
        
        return
        -----------
        name : str
            the root component name, None if not exists
        content : dict[str,str]
            root component description content
        '''
        name = self.m_content.get(UE_SceneDescKeys.UERootCompName)
        if name is None:
            return None, {}
        content = self.m_content[UE_SceneDescKeys.Components][name]
        return name, content
    
    def get_gltf_name(self) -> str:
        return self.m_content.get(UE_SceneDescKeys.GLTFNode)
    
    def get_asset_path(self) -> str:
        return self.m_content.get(UE_SceneDescKeys.AssetPath)
    
    def get_components(self, ue_class_name : str = None) -> dict[str, Union[str, dict[str,str]]]:
        ''' get all components or components of a specific ue class
        
        parameters
        --------------
        ue_class_name
            if specified, only include components of this UE class
            
        return
        -----------
        output : dict
            components content indexed by their display name
        '''
        components = self.m_content.get(UE_SceneDescKeys.Components)
        if not components:
            components = {}
            
        if ue_class_name is None:
            return components
        else:
            output = {}
            for name, comp in components.items():
                if comp.get(UE_SceneDescKeys.UEClassName) == ue_class_name:
                    output[name] = comp
            return output
        
class UE_SceneDescription(SceneDescription):
    def __init__(self) -> None:
        super().__init__()
        self.m_scene_structure : dict[str, Union[str, dict]] = None
        self.m_full_graph : nx.DiGraph = None   # the graph that contains all the nodes
        self.m_filtered_graph : nx.DiGraph = None    # graph contains selected nodes
        
    @property
    def graph(self) -> nx.DiGraph:
        ''' the filtered graph
        '''
        return self.m_filtered_graph
    
    @property
    def full_graph(self) -> nx.DiGraph:
        ''' the graph containing all nodes
        '''
        return self.m_full_graph
    
    @property
    def scene_structure(self):
        return self.m_scene_structure
        
    def init(self, scene_struct : dict):
        self.m_scene_structure = scene_struct
        self.m_full_graph = None
        self.m_filtered_graph = None
        
    @staticmethod
    def create_from_desc_file(fn_scene_struct : str, parse_now : bool = False) -> "UE_SceneDescription":
        import json
        with open(fn_scene_struct, 'r') as fid:
            s = json.load(fid)
        out = UE_SceneDescription()
        out.init(s)
        
        if parse_now:
            out.parse()
        
        return out
    
    def parse(self):
        ''' parse the scene structure into graph
        '''
        K = UE_SceneDescKeys
        
        ss_actors : dict = self.m_scene_structure[K.RootActors]
        actor_info : dict = None
        g = nx.DiGraph()
        name2node : dict[str, UE_SceneDescNode]= {}
        
        # create actor nodes
        for disp_name, actor_info in ss_actors.items():
            # this actor does not have component, skip
            # update a scene object if it exists, otherwise create it
            obj = UE_SceneDescNode()
                
            # set up basic info
            obj.m_name = disp_name
            obj.m_content = actor_info
            obj.m_owner = self
            obj.m_meta_class_type = UE_MetaClassType.Actor
            
            # add to graph
            g.add_node(disp_name, **{self.NodeKey: obj})
            name2node[disp_name] = obj
            
        # link actors
        for name, node in name2node.items():
            p = node.get_parent_actor_name()
            if p is not None and p in g.nodes:
                g.add_edge(p, name)
                
        # create nodes for static mesh components
        for name, node in name2node.items():
            meshcomps = node.get_components(ue_class_name=UE_ClassName.StaticMeshComponent)
            for comp_name, comp_info in meshcomps.items():
                obj = UE_SceneDescNode()
                obj.m_name = comp_name
                obj.m_content = comp_info
                obj.m_owner = self
                obj.m_meta_class_type = UE_MetaClassType.Component
                g.add_node(comp_name, **{self.NodeKey: obj})
                g.add_edge(name, comp_name)
                
        self.m_full_graph = g
        self.m_filtered_graph = g
    
    def get_nodes(self, root_only : bool = False, graph : nx.DiGraph = None) -> list[UE_SceneDescNode]:
        ''' get all description nodes in a given graph. If the graph is None, use self.graph
        
        parametesr
        -------------
        root_only
            if True, only get root nodes
        graph : nx.DiGraph
            the target description graph. By default, it is self.graph
        
        return
        ---------
        nodes
            a list of UE_SceneDescNode
        '''
        output : list[UE_SceneDescNode] = []
        if graph is None:
            graph = self.graph
            
        for name, content in graph.nodes.items():
            if root_only and graph.in_degree[name] > 0:
                continue
            node = content.get(self.NodeKey)
            output.append(node)
            
        return output
    
    def get_node_by_name(self, name : str) -> UE_SceneDescNode:
        ''' get a node by its name. If not exists, return None
        '''
        content = self.m_full_graph.nodes.get(name)
        if content is None:
            return None
        else:
            return content.get(self.NodeKey)
    
    def get_asset2nodes(self, graph : nx.DiGraph = None) -> dict[str, list[UE_SceneDescNode]]:
        ''' get a mapping between asset path and nodes
        
        parameters
        --------------
        graph
            scene description graph, self.graph or self.full_graph. By default, is self.graph.
        
        return
        ----------
        asset2nodelist : dict[str, list[UE_SceneDescNode]]
            scene description nodes groupped by their used assets
        '''
        if graph is None:
            graph = self.graph
        assert graph is self.m_filtered_graph or graph is self.m_full_graph, 'graph must belong to this object'
            
        nodelist = self.get_nodes(graph = self.m_full_graph)
        output : dict[str, list[UE_SceneDescNode]] = {}
        for node in nodelist:
            p = node.get_asset_path()
            if p is None:
                continue
            
            if p in output:
                output[p].append(node)
            else:
                output[p] = [node]
                
        return output
        
    
    def remove_node_selection(self):
        ''' remove all applied node selection, so that self.graph == self.full_graph
        '''
        self.m_filtered_graph = self.m_full_graph
    
    def apply_node_selection(self, selector : Callable[[UE_SceneDescNode], bool], 
                             include_child : bool = False) -> list[str]:
        ''' Apply node selection to current description graph. After that, self.graph will be changed.
        
        parameters
        ------------
        selector : func(node) -> bool
            a selection function which takes a description node and returns whether the node should be selected or not
        include_child : bool
            if True, if a node is selected, all the children will be selected too
            
        return
        --------
        selected_node_names : list[str]
            names of the selected nodes
        '''
        
        # filter by nodes
        select_func = selector
        
        g = self.m_filtered_graph
        node_selection = set()
        for name, content in g.nodes.items():
            if name in node_selection:
                continue
            node : UE_SceneDescNode = content.get(self.NodeKey)
            select_this : bool = select_func(node)
            if select_this:
                node_selection.add(name)
                if include_child:
                    node_selection.update(nx.dfs_preorder_nodes(g, name))
        
        self.m_filtered_graph = g.subgraph(node_selection)
        return list(node_selection)

import numpy as np
import networkx as nx
from .SceneGeometry import SceneGeometryBase, SceneMesh
from .Constants import CoordinateType
import trimesh.scene as triscene
import trimesh

from igpy.geometry.Box_3 import Box_3

class GLTF_Node:
    def __init__(self) -> None:
        self.m_name : str = None    # gltf node name
        self.m_transmat_wrt_parent : np.ndarray = np.eye(4) # transform wrt to parent, 4x4 right-mul transform
        self.m_owner : GLTF_SceneGeometry = None    # the scene that contains this node
        self.m_geometry : str = None   # this node contains geometry data, refers to the geometry entry
        
    @property
    def name(self) -> str:
        return self.m_name
    
    @property
    def has_local_mesh(self) -> bool:
        return self.m_geometry is not None
    
    @property
    def has_geometry_in_tree(self) -> bool:
        ''' whether there is a geometry exists in the tree rooted at this node
        '''
        g = self.m_owner.graph
        nodekey = self.m_owner.NodeKey
        for node_name in nx.dfs_preorder_nodes(g, self.m_name):
            node : GLTF_Node = g.nodes[node_name].get(nodekey)
            if node.has_local_mesh:
                return True
        return False
    
    @property
    def transmat_wrt_parent(self) -> np.ndarray:
        ''' right-multiply transformation matrix relative to parent
        '''
        return self.m_transmat_wrt_parent
    
    def get_bounding_box(self, 
                         include_child : bool = True,
                         coordinate:str = CoordinateType.World) -> Box_3:
        ''' get the 3d bounding box of this node. Return None if no gemetry exists
        '''
        meshlist = self.get_meshes(include_child=include_child, coordinate=CoordinateType.Local)
        if not meshlist:
            return None
        
        # compute local bounding box
        minc = np.zeros(3)
        minc.flat[:] = np.inf
        maxc = np.zeros(3)
        maxc.flat[:] = -np.inf
        
        for meshobj in meshlist:
            v = meshobj.get_transformed_vertices()
            minc = np.minimum(minc, v.min(axis=0))
            maxc = np.maximum(maxc, v.max(axis=0))
        
        box = Box_3()
        box.set_lengths(*(maxc - minc))
        
        # bounding box at center of local origin, which is NOT local coordiante
        # we need to shift the points to local coordinate by T_center
        T_center = np.eye(4)
        T_center[-1,:-1] = (maxc + minc)/2
        T_target = self.get_transform(coordinate=coordinate)
        T_box = T_center.dot(T_target)
        box.set_transmat(T_box)
        return box
    
    def get_parent_node(self):
        ''' get parent node if exists, otherwise return None
        '''
        g = self._get_scene_graph()
        nodekey = self.m_owner.NodeKey
        obj : GLTF_Node = g.pred[self.m_name].get(nodekey)
        return obj
    
    def get_merged_mesh(self, coordinate : str = CoordinateType.World) -> SceneMesh:
        ''' combine all meshes into one single mesh. Texture will be discarded.
        
        parameters
        --------------
        coordinate : str in CoordinateType
            the returned mesh's transmat will be in this coordinate
        '''
        
        import igpy.common.util_trimesh as ut_tri
        meshlist = self.get_meshes(include_child=True, coordinate=CoordinateType.Local)
        if not meshlist:
            return None
        tmesh = ut_tri.merge_scene_meshes(meshlist)
        out = SceneMesh()
        
        out.m_geometry_name = None
        out.m_instance_name = self.m_name
        out.m_mesh_object = tmesh
        out.m_transform = self.get_transform(coordinate)
        return out
        
    
    def get_meshes(self, 
                   include_child : bool = True, 
                   coordinate : str = CoordinateType.World) -> list[SceneMesh]:
        ''' get the all the meshes rooted at this node
        
        parameters
        ----------
        include_child
            If True, include all meshes of all children in all depths. 
            If False, only include the mesh of this node
        coorindate
            The output meshes are at which coordinate
            
        return
        ---------
        meshlist
            a list of meshes
        '''
        
        base_tmat : np.ndarray = self.get_transform(coordinate)
        
        # transmat of other nodes rooted at this node
        node2transmat : dict[str, np.ndarray] = {
            self.m_name : base_tmat
        }
        # node2mesh : dict[str, SceneMesh] = {}
        meshlist = []
        
        # read mesh of this node
        mesh_this = self._get_local_mesh()
        if mesh_this is not None:
            mesh_this.m_transform = base_tmat
            meshlist.append(mesh_this)
            
        nodekey = self.m_owner.NodeKey
        if include_child:
            # traverse all children, compute transforms
            g = self._get_scene_graph()
            src : GLTF_Node = None
            dst : GLTF_Node = None
            for e in nx.dfs_edges(g, self.m_name):
                u, v = e
                src = g.nodes[u].get(nodekey)
                dst = g.nodes[v].get(nodekey)
                
                tmat_src = node2transmat[src.name]
                tmat_dst_wrt_src = dst.transmat_wrt_parent
                tmat_dst = tmat_dst_wrt_src.dot(tmat_src)
                node2transmat[dst.name] = tmat_dst
                
                # read mesh of dst
                mesh_dst = dst._get_local_mesh()
                if mesh_dst is not None:
                    mesh_dst.m_transform = tmat_dst
                    meshlist.append(mesh_dst)
                    
        return meshlist
    
    def get_transform(self, coordinate : str = CoordinateType.World) -> np.ndarray:
        ''' get 4x4 right-mul transform with respect to the given coordinate
        
        parameters
        --------------
        coordinate : str
            the transformation relative to this coordinate, see CoordinateType for more information.
            
        return
        ----------
        transmat : np.ndarray
            4x4 right-multiply transformation matrix
        '''
        nodekey = self.m_owner.NodeKey
        if coordinate == CoordinateType.Local:
            return np.eye(4)
        elif coordinate == CoordinateType.Parent:
            return self.m_transmat_wrt_parent
        elif coordinate == CoordinateType.World:
            # traverse to the root
            g = self._get_scene_graph()
            tmat = self.m_transmat_wrt_parent
            for e in nx.edge_dfs(g, self.m_name, orientation='reverse'):
                parent_name, _, _ = e
                parent_obj : GLTF_Node = g.nodes[parent_name].get(nodekey)
                tmat_parent = parent_obj.transmat_wrt_parent
                tmat = tmat.dot(tmat_parent)
            return tmat
        else:
            raise ValueError("invalid coordinate type")
        
    
    def _get_scene_graph(self) -> nx.DiGraph:
        ''' get the scene graph that contains this node
        '''
        return self.m_owner.m_scene_graph
    
    def _get_local_mesh(self) -> SceneMesh:
        ''' get the geometry of this node, if exists. Otherwise return None
        '''
        if self.m_owner is None or self.m_geometry is None:
            return None
        mesh_obj = self.m_owner.m_scene_static.geometry[self.m_geometry]
        
        output = SceneMesh()
        output.m_geometry_name = self.m_geometry
        output.m_instance_name = self.m_name
        output.m_mesh_object = mesh_obj
        output.m_transform = self.m_transmat_wrt_parent
        return output
    
class GLTF_SceneGeometry(SceneGeometryBase):
    ''' the raw scene data is stored in a GLTF file
    ''' 
        
    def __init__(self) -> None:
        super().__init__()
        
        self.m_scene_static : triscene.Scene = None # static scene
        self.m_filepath : str = None    # the gltf file
        
        # the scene graph, where the nodes are names of GLTF nodes, 
        # and the node attribute contains a GLTF_Node object
        self.m_scene_graph : nx.DiGraph = None    
        
    @property
    def graph(self) -> nx.DiGraph:
        return self.m_scene_graph
        
    def load_scene_from_file(self, fn_gltf : str):
        self.m_scene_static = trimesh.load(fn_gltf)
        self.m_scene_graph = None
        self.m_filepath = fn_gltf
        
    def init(self, scene : triscene.Scene):
        self.m_scene_static = scene
        self.m_scene_graph = None
        self.m_filepath = None
        
    @staticmethod
    def create_from_gltf_file(fn_gltf : str, parse_now : bool = False) -> "GLTF_SceneGeometry":
        ''' create scene geometry from gltf scene file
        '''
        out = GLTF_SceneGeometry()
        out.load_scene_from_file(fn_gltf)
        if parse_now:
            out.parse()
            
        return out
        
    def parse(self):
        ''' create all additional internal structures to represent the scene
        '''
        self._parse_for_scene_graph()
        
    def get_gltf_node_by_name(self, node_name) -> GLTF_Node:
        ''' get GLTF node given the name, return None if such node does not exist
        '''
        g = self.m_scene_graph
        if node_name not in g.nodes:
            return None
        
        out : GLTF_Node = self.m_scene_graph.nodes[node_name].get(self.NodeKey)
        return out
    
    def get_all_nodes(self) -> list[GLTF_Node]:
        ''' get all GLTF nodes in the scene graph
        
        return
        ---------
        name2node : dict[str, GLTF_Node]
            gltf nodes indexed by their names
        '''
        assert self.m_scene_graph is not None
        g = self.m_scene_graph
        nodelist = []
        for node_name, node_data in g.nodes.items():
            obj : GLTF_Node = node_data.get(self.NodeKey)
            nodelist.append(obj)
        return nodelist
        
    def _parse_for_scene_graph(self):
        assert self.m_scene_static is not None, 'load or set scene first'
        
        # create nodes
        scene = self.m_scene_static
        nodes = scene.graph.nodes
        g = nx.DiGraph()
        for x in nodes:
            obj = GLTF_Node()
            obj.m_name = x
            obj.m_owner = self
            g.add_node(x, **{self.NodeKey : obj})
        
        # create edges
        edges = self.m_scene_static.graph.to_edgelist()
        edge_data : dict = None
        for u, v, edge_data in edges:
            g.add_edge(u,v)
            obj : GLTF_Node = g.nodes[v][self.NodeKey]
            
            tmat = edge_data.get('matrix')
            if tmat is not None:
                obj.m_transmat_wrt_parent = np.array(tmat, dtype=float).T    # use right-mul transformation matrix
                
            obj.m_geometry = edge_data.get('geometry')
            
        self.m_scene_graph = g
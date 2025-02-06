import numpy as np
import networkx as nx
from .Snapshot import Snapshot, SnapshotNode
from . import UE_SceneDescription as ue_desc
from . import UE_SceneRendering as ue_render
from . import GLTF_SceneGeometry as ue_geom
from .Constants import UE_MetaClassType, UE_ClassName
from igpy.geometry.Rectangle_2 import Rectangle_2
from typing import Callable

class UE_SnapshotNodeSelector:
    ''' select SnapshotNode by their scene description
    '''
    def __init__(self) -> None:
        self.m_desc_selectors : list[ue_desc.UE_DescNodeSelector] = None
        self.m_logical_op : bool = 'or' # 'and'|'or', how to combine selectors?
    
    def set_desc_selectors(self, selectors : list[ue_desc.UE_DescNodeSelector]):
        self.m_desc_selectors = selectors
        
    def set_logical_op(self, op_name:str):
        ''' use 'and' or 'or' to combine selectors?
        
        parameters
        -------------
        op_name : 'and'|'or'
            the logical operation or combine selection results. If 'and', then a node
            is selected only if it is selected by all selectors. If 'or', then one selector
            match suffices.
        '''
        assert op_name.lower() in ('and','or')
        self.m_logical_op = op_name.lower()
        
    def __call__(self, target : "UE_SnapshotNode")->bool:
        ''' return whether the target node should be selected by this selector
        '''
        return self.match(target)
        
    def match(self, target: "UE_SnapshotNode") -> bool:
        assert self.m_desc_selectors, 'empty selector is not allowed'
        
        select_result = np.zeros(len(self.m_desc_selectors), dtype=bool)
        for i, sel in enumerate(self.m_desc_selectors):
            ok = sel(target.description_node)
            select_result[i] = ok
            
        if self.m_logical_op == 'and':
            return np.all(select_result)
        else:   #or
            return np.any(select_result)

class UE_SnapshotNode(SnapshotNode):
    ''' objects in snapshot that we care about. It must correspond to a description node
    '''
    def __init__(self) -> None:
        super().__init__()
        
        self.m_name : str = None    #unique name of this object
        self.m_owner : UE_Snapshot = None  # source object
        
        # store the hierarchy of snapshot nodes
        self.m_graph : nx.DiGraph = None
        
        # a description node associated with this snapshot node
        self.m_desc_node : ue_desc.UE_SceneDescNode = None

        # additional data attached to this node
        self.m_user_data : dict[str, object] = {}
    
    @property
    def name(self) -> str:
        ''' unique name of this node
        '''
        return self.m_name
    
    @property
    def user_data(self) -> dict[str, object]:
        return self.m_user_data
        
    @property
    def is_ue_actor(self) -> bool:
        ''' is this snapshot node representing an actor?
        '''
        return self.m_desc_node.meta_class == UE_MetaClassType.Actor
    
    @property
    def is_ue_component(self) -> bool:
        ''' is this snapshot node representing a component?
        '''
        return self.m_desc_node.meta_class == UE_MetaClassType.Component
    
    @property
    def code(self) -> int:
        ''' id code of this object, which is defined by the rendering. If not exists, return None
        '''
        rd = self.get_rendering()
        if not rd:
            return None
        return rd.code
    
    @property
    def description_node(self) -> ue_desc.UE_SceneDescNode:
        return self.m_desc_node
    
    @property
    def visible(self) -> bool:
        ''' whether this node is visible in rendering
        '''
        rd = self.get_rendering()
        if rd is not None:
            if rd.pixel_indices is not None:
                return True
        
        nodes = self.get_children()
        for n in nodes:
            rd = n.get_rendering()
            if rd is not None and rd.pixel_indices is not None:
                return True
            
        return False
    
    @property
    def owner(self) -> "UE_Snapshot":
        return self.m_owner
    
    def __repr__(self) -> str:
        return 'UE_SnapshotNode(name={}, code={})'.format(self.name, self.code)
    
    def get_geometry_node(self) -> ue_geom.GLTF_Node:
        ''' get the geometry node associated with this snapshot, 
        which defines its world transformation. If not exists, return None
        '''
        
        # get the geometry node name
        geom_node_name = self.m_desc_node.get_root_geometry_node_name()
        if geom_node_name is None:
            return None
        
        # find node
        g_geom : nx.DiGraph = self.owner.scene_geometry.graph
        node = g_geom.nodes.get(geom_node_name)
        if not node:    # geometry does not exist in geometry graph
            return None
        output : ue_geom.GLTF_Node = node.get(ue_geom.SceneGeometryBase.NodeKey)
        return output
    
    def get_rendering(self) -> ue_render.UE_RenderedObject:
        ''' get the rendering of this node. If not exists, return None
        '''
        rd = self.owner.scene_rendering
        return rd.get_rendered_object_by_name(self.m_name)
    
    def get_bbox2_visible(self) -> Rectangle_2:
        ''' get the axis-aligned bounding box of the visible part of this node,
        including children. If not pixel is covered by this node, return None.
        '''
        idxpix = self.get_pixel_indices()
        imgsize_hw = self.get_image_size_hw()
        
        output : Rectangle_2 = None
        if idxpix is not None:
            yy, xx = np.unravel_index(idxpix, imgsize_hw)
            minc = np.array([xx.min(), yy.min()])
            maxc = np.array([xx.max(), yy.max()])
            x,y = minc
            w,h = maxc - minc
            output = Rectangle_2.create_from_xywh(x,y,w,h)
        return output
    
    def get_bbox2_complete(self, near_clip_distance : float = 1e-4) -> Rectangle_2:
        ''' get the axis-aligned bounding box which encloses the whole object,
        including parts in occlusion.
        
        parameters
        ---------------
        near_clip_distance
            If a vertex of the object is this near to the camera optical center,
        it will be discarded and not contribute to the bounding box.
        
        return
        ---------
        bbox : Rectangle_2
            the 2d bounding box. If a valid bbox cannot be constructed 
            (missing geometry, or all points are clipped), return None
        '''
        cam = self.owner.m_scene_render.camera
        assert cam is not None, 'the rendering should have camera information'
            
        gnode = self.get_geometry_node()
        if not gnode:
            return None
        meshes = gnode.get_meshes()
        
        # collect vertices
        vts = []
        for m in meshes:
            v = m.get_transformed_vertices()
            vts.append(v)
        if not vts:
            return None
        vts = np.row_stack(vts)
        
        # project to camera
        pts_camera = cam.convert_pts_world_to_camera(vts)
        
        # clip near-distance points
        mask = pts_camera[:,-1] >= near_clip_distance
        if not np.any(mask):
            return None
        pts_use = pts_camera[mask]
        pts_image = cam.convert_pts_camera_to_image(pts_use)
        
        out = Rectangle_2.create_as_bounding_box_of_points(pts_image)
        return out
        
    def get_children(self, depth : int = None) -> list["UE_SnapshotNode"]:
        ''' get all children of this node (not including this node)
        
        parameters
        -------------
        depth : int
            only get children within this number of hops from this node.
            If None, get all children with unlimited depth.
        '''
        nodekey = self.owner.NodeKey
        g = self.owner.graph
        output : list[UE_SnapshotNode]= []
        for x in nx.dfs_preorder_nodes(g, self.m_name, depth_limit=depth):
            snp = g.nodes[x].get(nodekey)
            output.append(snp)
        output = output[1:] # remove the first node, which is itself
        return output
    
    def get_all_parents(self) -> list["UE_SnapshotNode"]:
        ''' get all parent nodes, sorted from nearest (direct predecessor) to root
        '''
        g = self.owner.graph
        nodekey = self.owner.NodeKey
        output : list[UE_SnapshotNode] = []
        for parent, child in nx.edge_dfs(g, self.m_name, orientation='reverse'):
            node = g.nodes[parent].get(nodekey)
            output.append(node)
        return output
        
    def get_pixel_indices(self) -> np.ndarray:
        ''' get all pixel indices covered by this node, including children.
        Return None if no pixel is covered
        '''
        all_nodes = self.get_children()
        all_nodes.append(self)
        
        pixlist = []
        for node in all_nodes:
            rd = node.get_rendering()
            if not rd:
                continue
            
            if rd.pixel_indices is not None and len(rd.pixel_indices) > 0:
                pixlist.append(rd.pixel_indices)
        
        if pixlist:
            output : np.ndarray = np.concatenate(pixlist)
        else:
            output = None
        return output
                    
    def get_image_size_hw(self) -> np.ndarray:
        ''' image size in (height, width) where this snapshot is rendered
        '''
        if self.owner:
            return self.owner.image_size_hw
        else:
            return None

class UE_Snapshot(Snapshot):
    ''' snapshot got from unreal engine
    '''
    def __init__(self) -> None:
        super().__init__()

        self.m_graph : nx.DiGraph = None
        self.m_scene_render : ue_render.UE_SceneRendering = None
        self.m_scene_desc : ue_desc.UE_SceneDescription = None
        self.m_scene_geom : ue_geom.GLTF_SceneGeometry = None
    
    @property
    def graph(self) -> nx.DiGraph:
        return self.m_graph
    
    @property
    def scene_rendering(self) ->ue_render.UE_SceneRendering:
        return self.m_scene_render
    
    @property
    def scene_description(self) -> ue_desc.UE_SceneDescription:
        return self.m_scene_desc
    
    @property
    def scene_geometry(self) -> ue_geom.GLTF_SceneGeometry:
        return self.m_scene_geom
    
    @property
    def image_size_hw(self) -> np.ndarray:
        ''' image size in (height, width)
        '''
        if self.m_scene_render:
            return self.m_scene_render.image_size_hw
        else:
            return None
        
    def init(self, scene_rendering : ue_render.UE_SceneRendering,
             scene_description : ue_desc.UE_SceneDescription,
             scene_geometry : ue_geom.GLTF_SceneGeometry = None):
        self.m_scene_render = scene_rendering
        self.m_scene_desc = scene_description
        self.m_scene_geom = scene_geometry
        
    def get_node_by_name(self, node_name : str) -> UE_SnapshotNode:
        ''' get a snapshot node by its name, None if the name does not exists
        '''
        node_data = self.m_graph.nodes.get(node_name)
        if not node_data:
            return None
        x = node_data.get(self.NodeKey)
        return x
    
    def get_nodes(self, root_only : bool = False) -> list[UE_SnapshotNode]:
        return super().get_nodes(root_only=root_only)

    def get_asset2nodes(self) -> dict[str, list[UE_SnapshotNode]]:
        ''' get a mapping between asset path and nodes
        
        return
        ----------
        asset2nodelist : dict[str, list[UE_SnapshotNode]]
            nodes groupped by their used assets
        '''
        g = self.graph
         
        nodelist = self.get_nodes()
        output : dict[str, list[UE_SnapshotNode]] = {}
        for node in nodelist:
            p = node.description_node.get_asset_path()
            if p is None:
                continue
            
            if p in output:
                output[p].append(node)
            else:
                output[p] = [node]
                
        return output


    def parse(self):
        ''' parse everything into snapshot nodes
        '''
        
        # find all description nodes with some geometry in the descendants,
        # and establish hierarchy among them
        g_snapshot = nx.DiGraph()
        self.m_graph = g_snapshot
        
        g_desc = self.m_scene_desc.graph
            
        # select description nodes that have some geometry attached to them, and create snapshot nodes based on them
        desc_node : ue_desc.UE_SceneDescNode = None # for better IDE code completion
        desc_nodelist = self._find_desc_nodes_with_geometry()
        if not desc_nodelist:
            return  # nothing to parse
        
        snp_nodekey = self.NodeKey
        for name in desc_nodelist:
            desc_node : ue_desc.UE_SceneDescNode = g_desc.nodes[name].get(ue_desc.SceneDescription.NodeKey)
            snp_node = UE_SnapshotNode()
            snp_node.m_desc_node = desc_node
            snp_node.m_name = name
            snp_node.m_owner = self    
            g_snapshot.add_node(name, **{snp_nodekey: snp_node})
            
        g_desc_sub : nx.DiGraph = g_desc.subgraph(desc_nodelist)
        for name in desc_nodelist:
            for x in g_desc_sub.successors(name):
                g_snapshot.add_edge(name, x)
            
        # check, for root node, it must have a geometry node
        # for each snapshot node, find its geometry node which defines its transformation
        for name in g_snapshot.nodes:
            if g_snapshot.in_degree[name] > 0: 
                continue
            x : UE_SnapshotNode = g_snapshot.nodes[name].get(self.NodeKey)
            geom_node = x.get_geometry_node()
            assert geom_node is not None, 'snapshot node must have geometry node'
        
    
    def _find_desc_nodes_with_geometry(self) -> list[str]:
        ''' find all scene description nodes that have geometry information in their descendants
        
        return
        ---------
        desc_node_names : list[str]
            names of all found description nodes.
        '''
        g_desc = self.m_scene_desc.graph
        g_geom = self.m_scene_geom.graph
        
        # for each description node, find its geometry node
        # node selection : a node is selected only if its descendants have geometry
        desc_node : ue_desc.UE_SceneDescNode = None # for better IDE code completion
        selection : dict[str, bool] = {}
        
        # select nodes that have mesh components, and the geometry must exists
        for x in g_desc.nodes:
            desc_node = g_desc.nodes[x].get(ue_desc.SceneDescription.NodeKey)
            meshcomps = desc_node.get_components(ue_class_name=UE_ClassName.StaticMeshComponent)
            accept_this_node = False
            if meshcomps:
                for name, info in meshcomps.items():
                    geom_node_name = info.get(ue_desc.UE_SceneDescKeys.GLTFNode)
                    if geom_node_name is not None and geom_node_name in g_geom.nodes:
                        gnode : ue_geom.GLTF_Node = g_geom.nodes[geom_node_name].get(ue_geom.SceneGeometryBase.NodeKey)
                        if gnode.has_geometry_in_tree:
                            accept_this_node = True
                            break
            selection[x] = accept_this_node
                
        # propogate selection to parents
        for x in nx.dfs_postorder_nodes(g_desc):
            if g_desc.pred[x] and selection[x]:
                p = list(g_desc.pred[x])[0]
                selection[p] = True
        
        output = [name for name, sel in selection.items() if sel]
        return output
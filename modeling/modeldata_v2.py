import numpy as np
import trimesh as tri
import igpy.common.shortfunc as sf
from igpy.geometry.Box_3 import Box_3
from igpy.geometry.geometry import decompose_transmat
import igpy.common.util_trimesh as ut_trimesh

class CoordinateType:
    World = 'world'
    Local = 'local'

class ModelData:
    def __init__(self) -> None:
        # name of this model data
        self.m_name : str = None
        
        # the underlying mesh data
        self.m_mesh_local : tri.Trimesh = tri.Trimesh()
        
        # 4x4 right-multiply transformation matrix
        # that is, points.dot(transmat) = new_points
        self.m_transmat : np.ndarray = np.eye(4)
        
    @classmethod
    def from_vertex_face(cls, vertices : np.ndarray, faces : np.ndarray, name : str= None) -> "ModelData":
        ''' initialize with vertex-face data
        '''
        tmesh = tri.Trimesh(vertices=vertices, faces=faces)
        out = cls.from_trimesh(tmesh, name = name)
        return out
        
    @classmethod
    def from_trimesh(cls, obj : tri.Trimesh, name : str = None) -> "ModelData":
        ''' initialize with a trimesh object
        '''
        out = cls()
        out.m_mesh_local = obj
        out.m_transmat = np.eye(4)
        out.m_name = name
        return out
        
    @classmethod
    def from_scene_mesh(cls, obj : ut_trimesh.SceneMesh) -> "ModelData":
        ''' initialize with a scene mesh, loaded from a gltf file
        '''
        out = cls()
        out.m_mesh_local = obj.mesh_local.copy()
        out.m_name = obj.instance_name
        out.m_transmat = obj.transmat.copy()
        return out
        
    @property    
    def name(self) -> np.ndarray:
        return self.m_name
    
    @property
    def transmat(self) -> np.ndarray:
        return self.m_transmat
    
    @property
    def mesh_local(self) -> tri.Trimesh:
        return self.m_mesh_local
    
    @property
    def faces(self) -> np.ndarray:
        ''' get faces of the mesh
        '''
        if self.m_mesh_local:
            return self.m_mesh_local.faces
        
    @property
    def vertices_local(self) -> np.ndarray:
        ''' get vertices of the mesh in local coordinate
        '''
        if self.m_mesh_local:
            return self.m_mesh_local.vertices
        
    @property
    def vertices_world(self) -> np.ndarray:
        ''' get vertices of the mesh in world coordinate
        '''
        return self.get_vertices(CoordinateType.World)
    
    @property
    def boundary_edges(self) -> np.ndarray:
        ''' get boundary edges of the mesh, which are edges that are only included in one face
        
        return
        -------
        edges : (N,2) array
            edges[i]=(u,v) where u and v are vertex indices, defining a boundary edge
        '''
        if self.m_mesh_local is not None:
            edges = self.m_mesh_local.edges_sorted
            u_edges, counts = np.unique(edges, return_counts=True, axis=0)
            return u_edges[counts == 1]
    
    def set_name(self, name : str):
        self.m_name = name
        
    def set_transmat(self, transmat : np.ndarray):
        assert transmat.shape == (4,4)
        self.m_transmat = transmat
    
    def get_trimesh(self, coordinate_type : str = CoordinateType.World) -> tri.Trimesh:
        if coordinate_type == CoordinateType.Local:
            return self.m_mesh_local
        elif coordinate_type == CoordinateType.World:
            out = self.m_mesh_local.copy()
            out.apply_transform(self.m_transmat.T)
            return out
        
    def get_vertices(self, coordinate_type : str = CoordinateType.World) -> np.ndarray:
        ''' get vertices of the mesh in the specified coordinate type
        
        parameters
        --------------
        coordinate_type : str
            CoordinateType.* constant
        '''
        v = np.array(self.m_mesh_local.vertices)
        if coordinate_type == CoordinateType.Local:
            return v
        else:
            pts = tri.transform_points(v, self.m_transmat.T)
            return pts
        
    @property
    def position(self) -> np.ndarray:
        ''' get position of the mesh in world coordinate
        '''
        return np.array(self.m_transmat[-1,:3])
    
    @property
    def orientation(self) -> np.ndarray:
        ''' get orientation of the mesh in world coordinate, 3x3 matrix of [xdir,ydir,zdir]
        '''
        R,_,_ = decompose_transmat(self.m_transmat)
        return np.array(R)
        
    def get_aabb(self, coordinate_type : str = CoordinateType.World) -> Box_3:
        ''' get axis-aligned bounding box
        '''
        v = self.get_vertices(coordinate_type)
        out = Box_3.create_as_bounding_box_of_points(v)
        return out
    
    def get_obb(self, coordinate_type : str = CoordinateType.World) -> Box_3:
        ''' get oriented bounding box, which bounds the mesh in local space
        '''
        v = self.m_mesh_local.vertices
        box = Box_3.create_as_bounding_box_of_points(v)
        if coordinate_type == CoordinateType.World:
            box.apply_transform(self.m_transmat)
        return box
    
    def apply_transform(self, transmat : np.ndarray, coordinate_type : str = CoordinateType.World):
        ''' apply a transformation to the mesh
        '''
        if coordinate_type == CoordinateType.Local:
            self.m_transmat = transmat @ self.m_transmat
        elif coordinate_type == CoordinateType.World:
            self.m_transmat = self.m_transmat @ transmat
            
    def clone(self, with_texture : bool = True) -> "ModelData":
        out = self.__class__()
        out.m_name = self.m_name
        out.m_mesh_local = self.m_mesh_local.copy()
        if not with_texture and isinstance(out.m_mesh_local.visual, tri.visual.texture.TextureVisuals):
            out.m_mesh_local.visual.material = tri.visual.material.empty_material()
        out.m_transmat = np.array(self.m_transmat)
        return out
    
    def clone_for_transformation(self) -> "ModelData":
        ''' shallow copy of the mesh so that only transformation is independent
        '''
        out = self.__class__()
        out.__dict__.update(self.__dict__)
        out.m_transmat = np.array(self.m_transmat)
        return out
    
    def show(self, plt = None):
        ''' show the mesh
        '''
        import igpy.myplot.vistaplot as vp
        use_internal_plt = plt is None
        if use_internal_plt:
            plt = vp.ExPlotter.init_with_std_plotter(title=self.name)
        plt.add_mesh_with_trimesh_object(self.get_trimesh(), show_edges=True, opacity=0.9)
        
        if use_internal_plt:
            plt.show()
    
def load_from_gltf(filename : str, 
                   model_class : type = ModelData) -> list[ModelData]:
    ''' load model data from a gltf file
    
    parameters
    ------------
    filename : str
        filename of the gltf file
    model_class : type
        class of the model data, must be a subclass of ModelData
        
    return
    ----------
    out : list[ModelData]
        list of model data, each element is an instance of model_class
    '''
    assert issubclass(model_class, ModelData), "model_class must be a subclass of ModelData"
    
    meshlist = ut_trimesh.load_and_flatten_gltf(filename)
    out = []
    for m in meshlist:
        x = model_class.from_scene_mesh(m)
        out.append(x)
    return out
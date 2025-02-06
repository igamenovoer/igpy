# utilities related to trimesh library
import trimesh
import trimesh.visual
import numpy as np

class SceneMesh:
    ''' an instanced mesh of a gltf scene
    '''
    def __init__(self) -> None:
        self.m_mesh_object : trimesh.Trimesh = None
        self.m_geometry_name : str = None
        self.m_instance_name : str = None   # if loaded from gltf, this is the name of the node
        self.m_transform : np.ndarray = None    # right-multiplication transformation
    
    @property
    def instance_name(self) -> str:
        return self.m_instance_name
    
    @property
    def geometry_name(self) -> str:
        return self.m_geometry_name
    
    @property
    def transmat(self) -> np.ndarray:
        ''' 4x4 right-multiply transformation matrix
        '''
        return self.m_transform
    
    @property
    def mesh_local(self) -> trimesh.Trimesh:
        ''' mesh before transformation
        '''
        return self.m_mesh_object
        
    def get_transformed_mesh(self, with_texture = True) -> trimesh.Trimesh:
        if self.m_mesh_object is None:
            return None
        
        obj = self.m_mesh_object.copy()
        if isinstance(obj.visual, trimesh.visual.TextureVisuals):
            if with_texture:
                obj.visual.material = self.m_mesh_object.visual.material
            else:   # clear texture image
                obj.visual.material = trimesh.visual.material.empty_material()
            
        obj.apply_transform(self.m_transform.T) # the api uses left-mul transformation
        return obj
    
    def get_transformed_vertices(self) -> np.ndarray:
        ''' get the transformed vertices only
        
        return
        ----------
        vertices : Nx3 array
            transformed vertices
        '''
        if self.m_mesh_object is None or self.m_transform is None:
            return
        
        from . import shortfunc as sf
        pts = sf.transform_points(self.m_mesh_object.vertices, self.m_transform)
        return pts
    
    def get_transformed_simple_mesh(self) -> trimesh.Trimesh:
        ''' get transformed mesh that only contains vertex and face data
        '''
        v = self.get_transformed_vertices()
        if v is None:
            return
        
        f = self.m_mesh_object.faces
        output = trimesh.Trimesh(v,f)
        return output
    
def merge_trimesh_list(meshlist : list[trimesh.Trimesh]) -> trimesh.Trimesh:
    ''' merge a list of Trimesh into one, discarding texture
    '''
    vertices = []
    faces = []
    vertex_normals = []
    face_normals = []
    v_next = 0  # next starting vertex index
        
    for m in meshlist:
        vertices.append(m.vertices)
        faces.append(m.faces + v_next)
        face_normals.append(m.face_normals)
        vertex_normals.append(m.vertex_normals)
        v_next += len(m.vertices)
        
    v = np.row_stack(vertices)
    f = np.row_stack(faces)
    vn = np.row_stack(vertex_normals)
    fn = np.row_stack(face_normals)
    
    output = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=vn, face_normals=fn)
    return output
    
def merge_scene_meshes(scenemeshlist : list[SceneMesh], verbose = False) -> trimesh.Trimesh:
    ''' merge a list of SceneMeshes into a single trimesh, texture will be discarded
    '''
    meshlist = []
    if verbose:
        from tqdm import tqdm
        thelist = tqdm(scenemeshlist)
    else:
        thelist = scenemeshlist
        
    for x in thelist:
        m = x.get_transformed_mesh(with_texture=False)
        meshlist.append(m)
    out = trimesh.util.concatenate(meshlist)
    return out

def load_and_flatten_gltf(fn_gltf : str, 
                          exclude_geometry_regex:str = None,
                          include_geometry_regex:str = None) -> list[SceneMesh]:
    ''' load gltf scene and flatten it into meshes
    
    parameters
    --------------
    fn_gltf
        the gltf filename
    exclude_geometry_regex
        exclude the geometry if its name matches this pattern
    include_geometry_regex
        include the geometry only if its name matches this pattern
        
    return
    ---------
    meshlist
        a list of meshes
    '''
    scene = trimesh.load(fn_gltf)
    output = flatten_trimesh_scene(scene, exclude_geometry_regex=exclude_geometry_regex,
                          include_geometry_regex=include_geometry_regex)
    return output
    

def flatten_trimesh_scene(scene : trimesh.Scene, 
                          exclude_geometry_regex:str = None,
                          include_geometry_regex:str = None) -> list[SceneMesh]:
    ''' flatten trimesh scene into a list of meshes
    
    parameters
    --------------
    scene
        the trimesh scene
    exclude_geometry_regex
        exclude the geometry if its name matches this pattern
    include_geometry_regex
        include the geometry only if its name matches this pattern
        
    return
    ---------
    meshlist
        a list of meshes
    '''
    import re
    name2node : dict = scene.graph.to_flattened()
    ptn_exclude : re.Pattern = None
    ptn_include : re.Pattern = None
    if exclude_geometry_regex is not None:
        ptn_exclude = re.compile(exclude_geometry_regex)
    if include_geometry_regex is not None:
        ptn_include = re.compile(include_geometry_regex)
        
    output = []
    for name, node in name2node.items():
        geom_name = node['geometry']
        if not geom_name:
            continue
        
        if exclude_geometry_regex is not None:
            if ptn_exclude.match(name):
                continue
        
        if include_geometry_regex is not None:
            if not ptn_include.match(name):
                continue
        
        thismesh = SceneMesh()
        thismesh.m_geometry_name = geom_name
        thismesh.m_instance_name = name
        thismesh.m_transform = np.array(node['transform']).T
        thismesh.m_mesh_object = scene.geometry[geom_name]
        output.append(thismesh)
        
    return output

def trimesh_to_o3d_mesh_t(obj : trimesh.Trimesh):
    ''' convert trimesh.Trimesh to open3d.t.geometry.TriangleMesh
    '''
    import open3d as o3d
    
    omesh : o3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh()
    omesh.vertex.positions = o3d.core.Tensor(obj.vertices.astype(np.float32))
    omesh.vertex.normals = o3d.core.Tensor(obj.vertex_normals.astype(np.float32))
    omesh.triangle.indices = o3d.core.Tensor(obj.faces.astype(np.int32))
    omesh.material.material_name='defaultLit'
    
    if isinstance(obj.visual, trimesh.visual.TextureVisuals):
        if obj.visual.uv is not None:
            omesh.triangle.triangle_uvs = o3d.core.Tensor(obj.visual.uv[obj.faces].reshape(-1,2).astype(np.float32))
            
        if obj.visual.material.baseColorTexture is not None:
            diffuse_texture = np.asarray(obj.visual.material.baseColorTexture)
            omesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(diffuse_texture)
    
    return omesh

def trimesh_scene_to_o3d_mesh_t(obj : trimesh.Scene) -> dict:
    ''' convert trimesh.Scene into a list of open3d.t.geometry.TriangleMesh
    
    returns
    -------------
    named_meshes : dict[str, open3d.t.geometry.TriangleMesh]
        the meshes keyed by names
    '''
    output = {}
    for name, geom in obj.geometry.items():
        print('converting mesh {}'.format(name), flush=True)
        omesh = trimesh_to_o3d_mesh_t(geom)
        output[name] = omesh
    return output
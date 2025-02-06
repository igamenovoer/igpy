import numpy as np
import trimesh as tri
from igpy.geometry.Box_3 import Box_3
from igpy.modeling.modeldata_v2 import ModelData
import igpy.common.shortfunc as sf

class VoxelizeMethod:
    ''' voxelization method
    '''
    # voxelization method
    trimesh = 'trimesh'
    igfast = 'igfast'
    
class VoxelizeParams:
    igfast_precision_ratio = 4.0
    
class VoxelizedMesh:
    ''' voxel representation of a mesh
    '''
    
    VoxelPivot = np.zeros(3)    # pivot of the voxel, can be center (0.5,0.5,0.5) or corner (0,0,0)
    
    def __init__(self) -> None:
        # voxelization bounding box of the mesh in local coordinate, where all contents in the box are voxelized
        self.m_box_local : Box_3 = None
        
        # 3d voxel grid
        self.m_voxdata : tri.voxel.VoxelGrid = None
        self.m_shape : np.ndarray = None    # shape of the voxel grid at direction 0
        
        # mesh unit per voxel
        self.m_unit : float = None
        
        # used voxelization method
        self.m_method : str = None
        
        # caches
        self.m_dvox_original : np.ndarray = None
        self.m_dvox_filled : np.ndarray = None
        self.m_dvox_hollow : np.ndarray = None
        
    @property
    def shape(self) -> np.ndarray:
        ''' get the shape of the voxel grid, considering the canonical direction
        '''
        if self.m_voxdata is not None:
            return np.array(self.m_voxdata.shape)
    
    @property
    def voxel_unit(self) -> float:
        return self.m_unit
        
    @property
    def transmat_local_to_voxel(self) -> np.ndarray:
        ''' get the transformation matrix that converts points from local coordinate to voxel coordinate.
        Note that the minimum corner of the mesh is assumed to be at (0.5,0.5,0.5) in voxel coordinate,
        that is, at the center of the first voxel.
        '''
        pts_local = self.m_box_local.get_points()
        
        # local to box coordinate
        tmat_local2box = np.linalg.inv(self.m_box_local.transmat)
        tmat_box2origin = np.eye(4)
        tmat_box2origin[-1,:3] = self.m_box_local.lengths / 2
        
        # transform the box to pre-vox frame
        tmat_origin2prevox = np.eye(4)
        # tmat_origin2prevox[:3,:3] = GC.CanonicalDirections.get_rotmat_by_direction_type(self.m_direction_type, dtype=float)
        pts_prevox = sf.transform_points(pts_local, tmat_local2box @ tmat_box2origin @ tmat_origin2prevox)
        prevox_minc = pts_prevox.min(axis=0)
        tmat_origin2prevox[-1,:3] = -prevox_minc
        
        # transform pre-vox frame to voxel frame
        tmat_prevox2voxel = np.eye(4)
        tmat_prevox2voxel[:3,:3] = np.diag([1/self.m_unit, 1/self.m_unit, 1/self.m_unit])
        tmat_prevox2voxel[-1,:3] = self.VoxelPivot
        
        # combine all transforms
        tmat_local2voxel = tmat_local2box @ tmat_box2origin @ tmat_origin2prevox @ tmat_prevox2voxel
        return tmat_local2voxel
    
    @property
    def transmat_voxel_to_local(self) -> np.ndarray:
        return np.linalg.inv(self.transmat_local_to_voxel)
        
    @property
    def method(self) -> str:
        ''' get the used voxelization method
        '''
        return self.m_method
        
    @classmethod
    def from_model_data(cls, modeldata : ModelData, 
                        vox_unit : float,
                        vox_method : str = VoxelizeMethod.trimesh, 
                        use_cache : bool = False) -> "VoxelizedMesh":
        ''' create a voxelized object from model data, the model will be voxelized in world space
        '''
        out = cls()
        mesh = modeldata.get_trimesh()
        voxproc = MeshVoxelizer.from_model_data(modeldata, vox_unit)
        voxproc.process(method = vox_method)
        out.m_voxdata = voxproc.voxel_grid
        out.m_shape = np.array(voxproc.voxel_grid.shape)
        out.m_method = voxproc.method
        
        # voxgrid : tri.voxel.VoxelGrid = mesh.voxelized(vox_unit, method = GC.ComponentParams.VoxelizationMethod)
        # out.m_voxdata = voxgrid        
        out.m_unit = vox_unit
        
        out.m_box_local = Box_3.create_as_bounding_box_of_points(mesh.vertices)
        out.m_box_local.set_transmat(out.m_box_local.transmat @ np.linalg.inv(modeldata.transmat))
        
        if use_cache:
            out.update_cache()
        return out
    
    def remove_voxel_data(self):
        ''' remove all voxel data, but retaining other info for transformation computation
        '''
        self.m_voxdata = None
        self.m_dvox_filled = None
        self.m_dvox_hollow = None
        self.m_dvox_original = None
    
    def _set_voxel_data(self, vox : tri.voxel.VoxelGrid):
        self.m_voxdata = vox
        self.m_shape = np.array(vox.shape)
    
    def update_cache(self):
        if self.m_voxdata is not None:
            self.m_dvox_original = self.m_voxdata.matrix
            self.m_dvox_filled = self.m_voxdata.copy().fill().matrix
            self.m_dvox_hollow = self.m_voxdata.copy().fill().hollow().matrix
        
    def get_dense_voxel(self) -> np.ndarray:
        ''' get dense voxel data
        '''
        # vmat = self.m_voxdata.matrix
        vmat = self.m_dvox_original
        if vmat is None and self.m_voxdata is not None:
            vmat = np.array(self.m_voxdata.matrix)
        return vmat
    
    def get_dense_voxel_filled(self) -> np.ndarray:
        ''' get the filled voxel data
        '''
        # out = self.m_voxdata.copy().fill().matrix
        out = self.m_dvox_filled
        if out is None and self.m_voxdata is not None:
            out = np.array(self.m_voxdata.copy().fill().matrix)
        return out
    
    def get_dense_voxel_hollow(self) -> np.ndarray:
        ''' get the hollow voxel data
        '''
        out = self.m_dvox_hollow
        if out is None and self.m_voxdata is not None:
            out = np.array(self.m_voxdata.copy().fill().hollow().matrix)
        return out
    
    def clone(self, copy_voxel_data : bool = True, 
              copy_cache : bool = True) -> "VoxelizedMesh":
        ''' clone the object
        
        parameters
        ------------
        copy_voxel_data: bool
            whether to deep copy the voxel data. If False, the voxel data will be shallow copied
        copy_cache : bool
            whether to deep copy the cache. If False, the cache will be shallow copied
        '''
        newobj = self.__class__()
        newobj.m_box_local = self.m_box_local.clone()
        newobj.m_unit = self.m_unit        
        newobj.m_method = self.m_method
        
        if self.m_shape is not None:
            newobj.m_shape = self.m_shape.copy()
            
        newobj.m_voxdata = self.m_voxdata   # make a shallow copy
        if copy_voxel_data and self.m_voxdata is not None:
            # requested deep copy of voxel data
            newobj.m_voxdata = self.m_voxdata.copy()
            
        if copy_cache:
            # deep copy of cache
            newobj.update_cache()
        else:
            # shallow copy of cache
            newobj.m_dvox_filled = self.m_dvox_filled
            newobj.m_dvox_hollow = self.m_dvox_hollow
            newobj.m_dvox_original = self.m_dvox_original
            
        return newobj
    
    def convert_points_local_to_voxel(self, pts : np.ndarray, to_int : bool = False) -> np.ndarray:
        ''' convert points from local coordinate to voxel coordinate
        
        parameters
        ------------
        pts: np.ndarray
            points in local coordinate
        to_int: bool
            whether to convert the voxel points to integer
            
        return
        ----------
        pts_voxel: np.ndarray
            points in voxel coordinate, optionally converted to integer
        '''
        tmat = self.transmat_local_to_voxel
        pts_voxel = sf.transform_points(pts, tmat)
        if to_int:
            pts_voxel = pts_voxel.astype(int)
        return pts_voxel
    
    def convert_points_voxel_to_local(self, pts : np.ndarray) -> np.ndarray:
        ''' convert points from voxel coordinate to local coordinate
        '''
        tmat = self.transmat_voxel_to_local
        return sf.transform_points(pts, tmat)
    
    def get_placement_transmat_other_to_this_voxel(self, other : "VoxelizedMesh", 
                                                   this_voxel_position : np.ndarray,
                                                   other_pivot : np.ndarray = (0,0,0)) -> np.ndarray:
        ''' get the transformation matrix that maps the coordinate for a voxel point in another voxel space to this voxel space,
        if the other voxel object is placed at this_voxel_position inside this voxel space. After placement, the pivot of the other
        voxel object will be at this_voxel_position.
        
        parameters
        ------------
        other: VoxelizedMesh
            the other voxel object to be placed inside this voxel space
        this_voxel_position: np.ndarray
            the voxel position of the other voxel object in this voxel space
        other_pivot
            the pivot of the other voxel object in its own voxel space, which defines the location of the object, default is (0,0,0)
            
        return
        ---------
        tmat: np.ndarray
            the transformation matrix which maps the coordinate for a voxel point in other voxel space to the coordinate in this voxel space
        '''
        tmat_other_to_this = np.eye(4)
        scale_factor = other.voxel_unit / self.voxel_unit   # use this unit to measure other unit
        tmat_other_to_this[:3,:3] = np.diag([scale_factor, scale_factor, scale_factor])
        
        # move other_pivot to this_voxel_position, suppose two voxels are aligned at origin
        tmat_other_to_this[-1,:3] = this_voxel_position - other_pivot * scale_factor    
        
        return tmat_other_to_this
    
    def get_placement_transmat_other_to_this_local(self, other : "VoxelizedMesh",
                                                   this_voxel_position : np.ndarray,
                                                   other_pivot : np.ndarray = (0,0,0)) -> np.ndarray:
        ''' get the transformation matrix that maps the coordinate for a point in the local coordinate of another voxel object to this voxel object,
        if the other voxel object is placed at this_voxel_position inside this voxel space. After placement, the pivot of the other
        voxel object will be at this_voxel_position.
        
        parameters
        ------------
        other: VoxelizedMesh
            the other voxel object to be placed inside this voxel space
        this_voxel_position: np.ndarray
            the voxel position of the other voxel object in this voxel space
        other_pivot
            the pivot of the other voxel object in its own voxel space, which defines the location of the object, default is (0,0,0)
            
        return
        ---------
        tmat: np.ndarray
            the transformation matrix which maps the coordinate for a local point in other voxel object to the coordinate in this voxel object
        '''
        tmat_other_to_this = self.get_placement_transmat_other_to_this_voxel(other, this_voxel_position, other_pivot)
        out = other.transmat_local_to_voxel @ tmat_other_to_this @ self.transmat_voxel_to_local
        return out



class MeshVoxelizer:
    def __init__(self) -> None:
        self.m_mesh : ModelData = None
        self.m_voxel_unit : float = None
        
        # 3d voxel array
        self.m_voxdata : tri.voxel.VoxelGrid = None
        
        # the selected voxelize method
        self.m_method : str = None
    
    @property
    def voxel_grid(self) -> tri.voxel.VoxelGrid:
        ''' get trimesh voxel grid
        '''
        return self.m_voxdata
    
    @property
    def voxel_size(self) -> float:
        ''' get voxel size
        '''
        return self.m_voxel_unit
    
    @property
    def method(self) -> str:
        ''' get the used voxelization method
        '''
        return self.m_method
    
    @property
    def mesh(self) -> ModelData:
        ''' get mesh
        '''
        return self.m_mesh
        
    @classmethod
    def from_model_data(cls, mesh : ModelData, voxel_unit : float) -> "MeshVoxelizer":
        ''' initialize voxelizer with model data
        
        parameters
        ---------------
        mesh : ModelData
            the model
        voxel_unit : float
            mesh unit per voxel
        '''
        out = cls()
        out.m_mesh = mesh
        out.m_voxel_unit = voxel_unit
        return out
    
    @classmethod
    def from_trimesh(cls, mesh : tri.Trimesh, voxel_unit : float) -> "MeshVoxelizer":
        ''' initialize voxelizer with trimesh
        
        parameters
        ---------------
        mesh : tri.Trimesh
            the mesh
        voxel_unit : float
            mesh unit per voxel
        '''
        out = cls()
        out.m_mesh = ModelData.from_trimesh(mesh)
        out.m_voxel_unit = voxel_unit
        return out
        
    def process(self, method : str = None):
        ''' run voxelization algorithm to voxelize the mesh
        '''
        
        tmesh = self.m_mesh.get_trimesh()
        
        if method is None:
            # do we have igfast?
            try:
                import igfast
                method = VoxelizeMethod.igfast
            except:
                method = VoxelizeMethod.trimesh
        self.m_method = method
        
        if(method == VoxelizeMethod.trimesh):
            self.m_voxdata = tmesh.voxelized(self.m_voxel_unit)
        elif(method == VoxelizeMethod.igfast):
            precision = self.m_voxel_unit / VoxelizeParams.igfast_precision_ratio
            vox_indices = igfast.voxelize_mesh_to_grid(tmesh.vertices, tmesh.faces, 
                                                       self.m_voxel_unit, precision)
            voxshape = vox_indices.max(axis=0) + 1
            tvox_encoding = tri.voxel.encoding.SparseBinaryEncoding(vox_indices, voxshape)
            tmat = np.diag([self.m_voxel_unit,self.m_voxel_unit,self.m_voxel_unit,1])
            tmat[:-1,-1] = tmesh.vertices.min(axis=0)
            tvox = tri.voxel.VoxelGrid(tvox_encoding, transform=tmat)
            self.m_voxdata = tvox
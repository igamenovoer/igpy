import numpy as np
import torch
from numpy.typing import NDArray
import scipy.ndimage as ndi
from collections.abc import Generator
from igpy.modeling.modeldata_v2 import ModelData
from igpy.geometry.voxelization import MeshVoxelizer, VoxelizedMesh
from igpy.util_torch.blocks import SumOfGaussian

class MeshAsGaussianSum:
    def __init__(self) -> None:
        self.m_model : ModelData = None
        self.m_voxmesh : VoxelizedMesh = None
        
        # sum of gaussian model that represents this mesh, the coordinate is in [0,1]^3 space,
        # which is voxel coordinates divided by voxel shape
        self.m_sog : SumOfGaussian = None   
        
        # optimization parameters
        self.m_w_interior : float = 0.5
        self.m_w_exterior : float = 0.5
        self.m_optim_lr : float = 0.02
        self.m_optimizer : torch.optim.Optimizer = None
        self.m_torch_devce : str = 'cpu'
        self.m_loss_function = torch.nn.MSELoss()
        
    @classmethod
    def from_mesh(cls, mesh : ModelData, voxel_unit : float, n_gauss_component : int = 100) -> "MeshAsGaussianSum":
        ''' create a ShapeAsGaussianSum from a mesh
        
        parameters
        -------------
        mesh : ModelData
            the mesh which will be voxelized and represented as a sum of gaussian
        voxel_unit : float
            the unit length of the voxel
        n_gauss_component : int
            the number of gaussian components to use
            
        return
        ---------
        out : ShapeAsGaussianSum
            the ShapeAsGaussianSum object
        '''
        out = cls()
        
        out.m_model = mesh
        out.m_voxmesh = VoxelizedMesh.from_model_data(mesh, vox_unit=voxel_unit, use_cache=True)
        out.m_sog = SumOfGaussian(n_component=n_gauss_component, n_dim=3)
        out._initialize_sog()
        
        return out
    
    @classmethod
    def from_voxelized_mesh(cls, mesh : ModelData, 
                            voxmesh : VoxelizedMesh, 
                            n_gauss_component : int = 100) -> "MeshAsGaussianSum":
        ''' create a ShapeAsGaussianSum from a voxelized mesh
        
        parameters
        -----------
        mesh : ModelData
            the mesh which will be represented as a sum of gaussian
        voxmesh : VoxelizedMesh
            the voxelized mesh
        n_gauss_component : int
            the number of gaussian components to use
            
        return
        -------
        out : ShapeAsGaussianSum
            the ShapeAsGaussianSum object
        '''
        out = cls()
        
        out.m_model = mesh
        out.m_voxmesh = voxmesh
        out.m_sog = SumOfGaussian(n_component=n_gauss_component, n_dim=3)
        out._initialize_sog()
        
        return out
    
    @property
    def transmat_voxel_to_sog(self) -> np.ndarray:
        ''' return the transformation matrix which transforms points from voxel to the sum of gaussian,
        which transforms points to [0,1]^3 space.
        '''
        scale = 1.0 / self.m_voxmesh.shape
        tmat = np.diag([scale[0], scale[1], scale[2], 1.0])
        return tmat
        
    @property
    def transmat_world_to_sog(self) -> np.ndarray:
        ''' return the transformation matrix which transforms points from world to the sum of gaussian,
        in right-multiplied format, i.e. x_sog = x_world @ transmat_world_to_sog
        '''
        tmat_local2vox = self.m_voxmesh.transmat_local_to_voxel
        tmat_world2local = np.linalg.inv(self.m_model.transmat)
        return tmat_world2local @ tmat_local2vox @ self.transmat_voxel_to_sog
    
    @property
    def transmat_local_to_sog(self) -> np.ndarray:
        ''' return the transformation matrix which transforms points from mesh local coordinate to the sum of gaussian,
        in right-multiplied format, i.e. x_sog = x_local @ transmat_local_to_sog
        '''
        return self.m_voxmesh.transmat_local_to_voxel @ self.transmat_voxel_to_sog
    
    @property
    def gaussian_model(self) -> SumOfGaussian:
        ''' return the sum of gaussian model in voxel space. 
        The coordinate is in [0,1]^3 space, which is voxel coordinates divided by voxel shape
        '''
        return self.m_sog
    
    def evaluate_points_in_world(self, pts:np.ndarray) -> np.ndarray:
        ''' evaluate the sum of gaussian model at points in world space
        
        parameters
        -----------
        pts : np.ndarray
            points in world space, in shape (n,3)
        
        return
        -------
        out : np.ndarray
            the value of the sum of gaussian model at the given points. 
            out[i,j] < 0 means the point is inside the mesh, and out[i,j] > 0 outside the mesh
        '''
        pts = np.atleast_2d(pts)
        
        # convert points to voxel space
        tmat = self.transmat_world_to_sog
        pts_sog = pts @ tmat[:3,:3].T + tmat[-1,:3][None,:]
        pts_sog_t = torch.tensor(pts_sog, dtype=torch.float, device=self.m_torch_devce)
        with torch.no_grad():
            p = self.m_sog(pts_sog_t).detach().cpu().numpy().flatten()
        return p
    
    def evaluate_points_in_voxel(self, pts:np.ndarray) -> np.ndarray:
        ''' evaluate the sum of gaussian model at points in voxel space
        
        parameters
        ----------
        pts : np.ndarray
            points in voxel space, in shape (n,3)
        
        return
        -------
        out : np.ndarray
            the value of the sum of gaussian model at the given points. 
            out[i,j] < 0 means the point is inside the mesh, and out[i,j] > 0 outside the mesh
        '''
        pts = np.atleast_2d(pts)
        
        # convert points to voxel space
        tmat = self.transmat_voxel_to_sog
        pts_sog = pts @ tmat[:3,:3].T + tmat[-1,:3][None,:]
        pts_sog_t = torch.tensor(pts_sog, dtype=torch.float, device=self.m_torch_devce)
        with torch.no_grad():
            p = self.m_sog(pts_sog_t).detach().cpu().numpy().flatten()
        return p
    
    def to_device(self, device : str) -> "MeshAsGaussianSum":
        ''' move the model to a device
        
        parameters
        -----------
        device : str
            device name, e.g. 'cuda' or 'cpu', following the convention of pytorch
        '''
        self.m_torch_devce = device
        
        if self.m_sog is not None:
            self.m_sog.to(device)
        return self
    
    def _initialize_sog(self):
        ''' initialize the parameters of the sum of gaussian
        '''
        
        vox = self.m_voxmesh.get_dense_voxel_filled()
        
        x,y,z = np.unravel_index(np.arange(vox.size), vox.shape)
        pos = np.column_stack((x,y,z)).astype(float)
        pos = pos @ self.transmat_voxel_to_sog[:3,:3] + self.transmat_voxel_to_sog[-1,:3][None,:]
        idx_centers = np.arange(0, vox.size, vox.size//self.m_sog.num_component)[:self.m_sog.num_component]
        self.m_sog.set_mean(pos[idx_centers])
        
    def _set_learning_rate(self, lr : float):
        ''' set the learning rate of the optimizer
        '''
        self.m_optim_lr = lr
        if self.m_optimizer is not None:
            self.m_optimizer.param_groups['lr'] = lr
        
    def fit(self, n_iter_each_run : int = None, use_data_parallel : bool = None) -> Generator[dict[str,torch.Tensor], None, None]:
        ''' fit the sum of gaussian to the voxelized mesh
        '''
        if n_iter_each_run is None:
            n_iter_each_run = 1
            
        if use_data_parallel is None:
            use_data_parallel = False
            
        if use_data_parallel:
            assert torch.cuda.is_available() and 'cuda' in self.m_torch_devce, 'data parallel does not work on cpu, use cuda instead'
            
        torch_device = self.m_torch_devce
        optim = torch.optim.Adam(self.m_sog.parameters(), lr=self.m_optim_lr)
        self.m_optimizer = optim
        
        # find interior and exterior points
        vox_interior = self.m_voxmesh.get_dense_voxel_filled()
        
        # signed distance field, negative inside, positive outside
        # in the world unit
        sdf = ndi.distance_transform_edt(~vox_interior) - ndi.distance_transform_edt(vox_interior)
        
        x,y,z = np.unravel_index(np.arange(vox_interior.size), vox_interior.shape)
        pts_all = np.column_stack([x,y,z]).astype(float)
        pts_all = pts_all @ self.transmat_voxel_to_sog[:3,:3] + self.transmat_voxel_to_sog[-1,:3][None,:]
        
        pts_in_t = torch.tensor(pts_all[vox_interior.flat], dtype=torch.float, device=torch_device)
        sdf_in_t = torch.tensor(sdf[vox_interior], dtype=torch.float, device=torch_device)
        pts_out_t = torch.tensor(pts_all[~vox_interior.flatten()], dtype=torch.float, device=torch_device)
        sdf_out_t = torch.tensor(sdf[~vox_interior], dtype=torch.float, device=torch_device)
        
        if use_data_parallel:
            sog = torch.nn.DataParallel(self.m_sog)
        else:
            sog = self.m_sog
        
        while True:
            for i in range(n_iter_each_run):
                # self.m_sog.clip_sigma_det()
                
                optim.zero_grad()
                sg_inside = sog(pts_in_t)
                sg_outside = sog(pts_out_t)
                cost_interior = self.m_loss_function(sg_inside, sdf_in_t) * self.m_w_interior
                cost_exterior = self.m_loss_function(sg_outside, sdf_out_t) * self.m_w_exterior
                # cost_det = torch.relu(torch.log(sigma_min_det) - torch.log(torch.det(self.m_sog.sigma))).mean()
                # cost_det = torch.relu(torch.det(self.sigma)
                cost : torch.Tensor = cost_interior + cost_exterior
                cost.backward()
                optim.step()
            yield {'cost':cost, 'cost_interior':cost_interior, 'cost_exterior':cost_exterior}
        
        
        
    
        
import numpy as np
import igpy.common.shortfunc as sf
from .geometry import decompose_transmat

from typing import Tuple

class Box_3:
    ''' oriented 3d box, defined by a 4x4 right-multiply transformation matrix and lengths along x,y,z local axis.
    The transformation matrix defines the location of the box center, and the box orientation.
    By default, the box is of unit length, with identity transform
    '''
    def __init__(self) -> None:
        # length along x,y,z local axis
        self.m_length_xyz : np.ndarray = np.ones(3)
        
        # right-multiply transformation matrix, let it be T, 
        # then use p.dot(T) to transform a point.
        # the local coorindate is at the center of the bounding box
        self.m_transmat : np.ndarray = np.eye(4)
        
    def __repr__(self) -> str:
        infos : list[str] = [
            'length = [{}, {}, {}]'.format(*self.m_length_xyz),
            'transmat = {}'.format(self.m_transmat)
        ]
        return '\n'.join(infos)
        
    @property
    def local_points(self) -> np.ndarray:
        ''' get all the corner points in local coordinate
        
        return
        -----------
        pts : np.ndarray
            8x3 array, 8 corners of the box
        '''
        corners = np.array([[0., 0., 0.],
                            [0., 0., 1.],
                            [0., 1., 0.],
                            [0., 1., 1.],
                            [1., 0., 0.],
                            [1., 0., 1.],
                            [1., 1., 0.],
                            [1., 1., 1.]])
        pts = corners * self.m_length_xyz - self.m_length_xyz/2
        return pts
    
    @property
    def transmat(self) -> np.ndarray:
        return self.m_transmat
    
    @property
    def scale(self) -> np.ndarray:
        ''' scale in x,y,z directions
        '''
        r,s,t = decompose_transmat(self.m_transmat)
        return s
    
    @property
    def orientation(self) -> np.ndarray:
        ''' 3x3 orientation matrix, each row is the direction along one axis
        '''
        r,s,t = decompose_transmat(self.m_transmat)
        return r
    
    @property
    def lengths(self) -> np.ndarray:
        ''' length along local x,y,z axis, with scaling factor applied
        '''
        s = self.scale
        return self.m_length_xyz * s
    
    @property
    def volume(self) -> float:
        ''' volume of the box
        '''
        return np.prod(self.lengths)
    
    @property
    def position(self) -> np.ndarray:
        ''' world position of the box center
        '''
        return self.m_transmat[-1,:-1]
    
    def set_transmat(self, transmat : np.ndarray):
        ''' set the 4x4 right-mul transformation matrix
        '''
        assert np.allclose(transmat.shape,(4,4)), 'transmat must be 4x4'
        self.m_transmat = transmat
        
    def apply_transform(self, transmat: np.ndarray):
        ''' apply 4x4 right-mul transformation to the box
        '''
        self.m_transmat = self.m_transmat @ transmat
        
    def set_lengths(self, dx, dy, dz):
        ''' set the length along x,y,z local axis
        '''
        self.m_length_xyz = np.array([dx, dy, dz])
        
    def set_position(self, x, y, z):
        self.m_transmat[-1,:-1] = (x,y,z)
        
    def get_edges(self) -> np.ndarray:
        ''' get box edges as a list of line segments
        
        return
        --------
        edge_index : Nx2 index array
            End points of line segments, let i,j = edge_index[k], then the k-th segment joins
            pts[i] and pts[j], where pts = get_points()
        '''
        edges = np.array([[0, 1],
                        [0, 2],
                        [0, 4],
                        [1, 3],
                        [1, 5],
                        [2, 3],
                        [2, 6],
                        [3, 7],
                        [4, 5],
                        [4, 6],
                        [5, 7],
                        [6, 7]])
        return edges
        
    
    def get_triangle_faces(self) -> np.ndarray:
        ''' get the index array that can organize the vertices into triangles
        
        return
        ---------
        faces
            Nx3 face array, each row contains the indices of the corners that resembles a triangle
        '''
        f = np.array([[0, 2, 6],
                        [6, 4, 0],
                        [0, 4, 5],
                        [5, 1, 0],
                        [4, 6, 5],
                        [5, 6, 7],
                        [3, 2, 0],
                        [0, 1, 3],
                        [3, 6, 2],
                        [7, 6, 3],
                        [1, 5, 3],
                        [3, 5, 7]])
        return f
        
    def get_points(self) -> np.ndarray:
        ''' get all the corner points in world coordinate
        
        return
        -----------
        pts : np.ndarray
            8x3 array, 8 corners of the box
        '''
        corners = self.local_points
        pts = sf.transform_points(corners, self.m_transmat)
        
        return pts
    
    def clone(self) -> "Box_3":
        ''' clone this box
        '''
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def create_as_bounding_box_of_points(pts3d : np.ndarray) -> "Box_3":
        ''' create axis-aligned bounding box to enclose list of points
        '''
        pts = np.atleast_2d(pts3d)
        minc = pts.min(axis=0)
        maxc = pts.max(axis=0)
        out = Box_3()
        out.m_length_xyz = maxc - minc
        out.m_transmat = np.eye(4)
        out.m_transmat[-1,:-1] = (maxc + minc)/2
        return out
        
    
        
        
        
import numpy as np
import igpy.common.shortfunc as sf
from .geometry import decompose_transmat

from typing import Tuple

class Rectangle_2:
    ''' oriented 2d box, defined by a 3x3 right-multiply transformation matrix and lengths along x,y in local axis.
    The transformation matrix defines the location of the box center and the box orientation.
    By default, the box is of unit length, with identity transform
    '''
    class PointIndex:
        ''' use these index to access specific point from returned points
        '''
        LowerLeft = 0
        UpperLeft = 3
        UpperRight = 2
        LowerRight = 1
    
    def __init__(self) -> None:
        # length along x,y,z local axis
        self.m_length_xy : np.ndarray = np.ones(2)
        
        # right-multiply transformation matrix, let it be T, 
        # then use p.dot(T) to transform a point
        self.m_transmat : np.ndarray = np.eye(3)
        
    def __repr__(self) -> str:
        infos : list[str] = [
            'length = [{}, {}]'.format(*self.m_length_xy),
            'transmat = {}'.format(self.m_transmat)
        ]
        return '\n'.join(infos)
        
    @property
    def local_points(self) -> np.ndarray:
        ''' get all the corner points in local coordinate
        
        return
        -----------
        pts : np.ndarray
            4x3 array, corners of the rectangle. Ordered by lower left, upper left, upper right, lower right
        '''
        corners = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
        pts = (corners-0.5) * self.m_length_xy
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
        ''' length along local x,y,z axis
        '''
        s = self.scale
        return self.m_length_xy * s
    
    @property
    def area(self) -> float:
        ''' volume of the box
        '''
        return np.prod(self.lengths)
    
    @property
    def position(self) -> np.ndarray:
        ''' world position of the box center
        '''
        return self.m_transmat[-1,:-1]
    
    @staticmethod
    def create_from_xywh(x,y,width,height) -> "Rectangle_2":
        ''' create axis aligned rectangle given top left corner (x,y) and size (width, height)
        '''
        obj = Rectangle_2()
        obj.m_length_xy = np.array([width, height], dtype=float)
        tmat = np.eye(3)
        tmat[-1,:-1] = (x+width/2.0, y+height/2.0)
        obj.m_transmat = tmat
        return obj
    
    @staticmethod
    def create_as_bounding_box_of_points(pts2d : np.ndarray) -> "Rectangle_2":
        ''' create rectangle to bound a list of 2d points
        '''
        pts2d = np.atleast_2d(pts2d)
        minc = pts2d.min(axis=0)
        maxc = pts2d.max(axis=0)
        x,y = minc
        w,h = maxc - minc
        return Rectangle_2.create_from_xywh(x,y,w,h)
    
    @staticmethod
    def create_from_center_vec(center:np.ndarray, 
                               x_length : float,
                               y_length : float,
                               x_dir : np.ndarray, 
                               y_dir : np.ndarray = None) -> "Rectangle_2":
        ''' create rectangle given the center, length and edge directions
        
        parameters
        ---------------
        center
            the center of the rectangle
        x_length, y_length
            rectangle length along x and y directions
        x_dir, y_dir
            x and y directions, must be perpendicular.
            If y_dir is not specified, it is automatically inferred
            
        return
        ----------
        rect
            the resulting rect
        '''
        out = Rectangle_2()
        out.set_lengths(x_length, y_length)
        
        x_dir = np.atleast_1d(x_dir) / np.linalg.norm(x_dir)
        if y_dir is None:
            y_dir = np.array([-x_dir[1], x_dir[0]])
        else:
            assert np.allclose(np.dot(x_dir,y_dir), 0), 'x_dir and y_dir must be perpendicular'
            y_dir = np.atleast_1d(y_dir) / np.linalg.norm(y_dir)
        transmat = np.eye(3)
        transmat[:,:2] = np.row_stack((x_dir, y_dir, center))
        out.m_transmat = transmat
        return out
        
    
    def to_xywh(self) -> np.ndarray:
        ''' get the axis-aligned rectangle that bounds this rectangle in (x,y,width,height) format,
        where (x,y) is the top-left corner        
        '''
        pts = self.get_points()
        minc = pts.min(axis=0)
        maxc = pts.max(axis=0)
        x,y = minc
        w,h = maxc - minc
        return np.array([x,y,w,h], dtype=float)
    
    def set_transmat(self, transmat : np.ndarray):
        ''' set the 4x4 right-mul transformation matrix
        '''
        assert np.allclose(transmat.shape,(3,3)), 'transmat must be 3x3'
        self.m_transmat = transmat
        
    def set_lengths(self, dx, dy):
        ''' set the length along x,y,z local axis
        '''
        self.m_length_xy = np.array([dx, dy], dtype=float)
        
    def get_edges(self) -> np.ndarray:
        ''' get box edges as a list of line segments
        
        return
        --------
        edge_index : Nx2 index array
            End points of line segments, let i,j = edge_index[k], then the k-th segment joins
            pts[i] and pts[j], where pts = get_points()
        '''
        edges = np.array([[0,1],[1,2],[2,3],[3,0]])
        return edges
        
    def get_points(self) -> np.ndarray:
        ''' get all the corner points in world coordinate
        
        return
        -----------
        pts : np.ndarray
            Nx2 corner points
        '''
        corners = self.local_points
        pts = sf.transform_points(corners, self.m_transmat)
        
        return pts
    
    def get_polyline(self, close : bool = True) -> np.ndarray:
        ''' get the edge as polyline
        '''
        pts = self.get_points()
        if close:
            pts = np.row_stack([pts, pts[0]])
        return pts
        

        
    
        
        
        
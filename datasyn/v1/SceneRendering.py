import numpy as np
from typing import Union
from igpy.common.calibration import CameraModel
from igpy.geometry.Rectangle_2 import Rectangle_2

class SceneRendering:
    ''' represent the rendering of a scene
    '''
    def __init__(self) -> None:        
        # camera in opencv convention
        self.m_camera : CameraModel = None
        
    @property
    def camera(self) -> CameraModel:
        ''' the camera (in opencv convention) with which this frame is rendered
        '''
        return self.m_camera
    
#######
class RenderedObject:
    ''' An object rendered by whatever engine
    '''
    def __init__(self) -> None:
        self.m_name : str = None   # unique for each scene object
        self.m_code : int = None   # object id in exr
        
        # linear pixel indices covered by this instance
        self.m_pixel_indices : np.ndarray = None
        self.m_image_size_hw: tuple = None  # image size in (height, width), of the image where this object is rendered
        
    @property
    def name(self) -> str:
        ''' the unique name of this object, suitable for hash map
        '''
        return self.m_name
    
    @property
    def code(self) -> int:
        ''' the unique ID code of this object, suitable for hash map
        '''
        return self.m_code
    
    @property
    def pixel_indices(self) -> np.ndarray:
        ''' 1d pixel indices occupied by this object
        '''
        return self.m_pixel_indices
    
    @property
    def image_size_hw(self)->tuple:
        ''' image size in (height, width), of the image where this object is rendered
        '''
        return self.m_image_size_hw
    
    @property
    def mask(self) -> np.ndarray:
        ''' get the binary mask of this object
        '''
        img = np.zeros(self.m_image_size_hw, dtype=bool)
        if self.m_pixel_indices is not None:
            img.flat[self.m_pixel_indices] = True
        return img
        
    def get_bbox2_visible(self) -> Rectangle_2:
        ''' get the image space bounding box that encloses the visible part of the object.
        
        return
        ----------
        box : Rectangle_2
            the axis aligned bounding box
        '''
        assert self.m_image_size_hw is not None
        pix_indices = self.m_pixel_indices
        if pix_indices is None or len(pix_indices) == 0:
            return None
        
        yy, xx = np.unravel_index(pix_indices, self.m_image_size_hw)
        minc = np.array([xx.min(), yy.min()])
        maxc = np.array([xx.max(), yy.max()])
        x, y = minc
        w, h = maxc - minc
        box_xywh = Rectangle_2.create_from_xywh(x,y,w,h)        
        return box_xywh
    

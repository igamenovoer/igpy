import numpy as np
from .SingleFrameData import SingleFrameData, SceneObject, CameraModel
from typing import Tuple
from igpy.geometry.Box_3 import Box_3

class BoxInfo:
    def __init__(self) -> None:
        self.m_sobj : SceneObject = None    # the scene object that generates this bbox
        self.m_semantic_name : str = None    # a non-unique name for this object 
        self.m_bbox_xywh_full = None     #2d bounding box in x,y,w,h format, no occlusion
        self.m_bbox_xywh_visible = None  #2d bounding box for the visible part of the object
        self.m_bbox_3 : Box_3 = None  #3d oriented bounding box
        
    @property
    def instance_code(self):
        if self.m_sobj:
            return self.m_sobj.instance_code
        else:
            return None
    
    @property
    def instance_name(self):
        if self.m_sobj:
            return self.m_sobj.instance_name
        else:
            return None

class BBoxLabelRenderer:
    ''' extract and render 2d an 3d bounding box
    '''
    def __init__(self) -> None:
        self.m_camera : CameraModel = None  # the camera view of this frame
        self.m_name2boxinfo : dict[str, BoxInfo] = {} # box info indexed by scene object's instance name
        self.m_semantic_group : dict[str, list[str]] = {}   # mapping semantic name to a list of instance names
        
    def init(self, cam : CameraModel):
        ''' initialize with camera and scene objects
        
        parameters
        ---------------
        cam
            camera information for this image
        '''
        self.m_camera = cam
        
    def set_camera(self, cam: CameraModel):
        self.m_camera = cam
        
    def add_scene_object(self, semantic_name : str,  obj : SceneObject):
        box = BoxInfo()
        box.m_sobj = obj
        box.m_semantic_name = semantic_name
        box.m_bbox_xywh_full = obj.get_bbox2_unoccluded(self.m_camera)
        box.m_bbox_xywh_visible = obj.get_bbox2_visible()
        self.m_name2boxinfo[semantic_name] = box
        
        
        
        
        
# load a single frame rendering with scene description and camera info
import numpy as np
import pandas as pd

import igpy.common.shortfunc as sf
from igpy.common.EnumBase import CustomEnumBase
from igpy.common.calibration import CameraModel, intrinsic_matrix_by_fovx
import igpy.datasyn.unreal.util_unreal_engine as ut_ue
from . import UE_SceneDescription as ue_desc
from . import UE_SceneRendering as ue_render
from . import GLTF_SceneGeometry as ue_geom
from . import UE_Snapshot as ue_snp

class UE_StaticFrameLoader:
    ''' loading and parsing sequence of unreal rendered exr files and camera data
    '''
    
    def __init__(self) -> None:
        self.m_fn_exr : str = None  # the exr file of a static frame
        self.m_fn_camera : str = None   # the camera information file
        self.m_fn_scene_desc : str = None  # the scene description file
        self.m_fn_scene_geom : str = None   # the scene geometry file (gltf format)
        
        self.m_camera : CameraModel = None  # camera in opencv convention
        
        # label data for the current frame
        self.m_snapshot : ue_snp.Snapshot = None

        # scene description and geometry
        self.m_scene_desc : ue_desc.UE_SceneDescription = None
        self.m_scene_geom : ue_geom.GLTF_SceneGeometry = None
        self.m_scene_render : ue_render.UE_SceneRendering = None
        
    def parse_snapshot(self) -> ue_snp.Snapshot:
        ''' parse the snapshot data with scene rendering, description and geometry (optional)
        '''
        assert self.m_scene_render is not None, 'you must load frame first'
        assert self.m_scene_desc is not None, 'you must load scene description first'
        
        snp = ue_snp.UE_Snapshot()
        self.m_snapshot = snp
        snp.init(self.m_scene_render, self.m_scene_desc, self.m_scene_geom)
        snp.parse()
        
        return snp
        
    def load_rendering(self, fn_exr : str) -> ue_render.UE_SceneRendering:
        ''' load a single image rendering of the scene
        
        parameters
        ------------
        fn_exr
            the exr file of a static frame
        '''
        self.m_fn_exr = fn_exr
        
        # load image and set camera
        frame_data = ue_render.UE_SceneRendering.create_from_exr_file(fn_exr, parse_now=True)
        self.m_scene_render = frame_data
        return frame_data
        
    def load_scene_description(self, fn_scene_desc : str) -> ue_desc.SceneDescription:
        ''' load scene structure from the json file
        
        parameters
        ------------
        fn_scene_desc
            the json file that saves the scene hierarchy and relevant class names
            
        return
        ----------
        scene_desc
            the scene description by interpreting scene structure
        '''
        desc = ue_desc.UE_SceneDescription.create_from_desc_file(fn_scene_desc)
        desc.parse()
        
        self.m_scene_desc = desc
        self.m_fn_scene_desc = fn_scene_desc
        
        return desc

    def load_scene_geometry(self, fn_gltf_scene : str) -> ue_geom.GLTF_SceneGeometry:
        ''' load scene geometry from gltf file
        
        parameters
        --------------
        fn_gltf_scene
            the gltf scene file
            
        return
        ---------
        scene_geometry : GLTF_SceneGeometry
            the geometry data of the scene
        '''
        
        obj = ue_geom.GLTF_SceneGeometry.create_from_gltf_file(fn_gltf_scene)
        obj.parse()
        self.m_scene_geom = obj
        self.m_fn_scene_geom = fn_gltf_scene
        
        return obj
    
    def load_camera(self, fn_camera : str):
        ''' load camera info from ue output json file
        '''
        import json
        self.m_fn_camera = fn_camera
        with open(fn_camera, 'r') as fid:
            json_data = json.load(fid)
            fov_x_deg = json_data['fov_x']
            
        # load an image to get size
        assert self.m_scene_render is not None, 'you must load frame first'
        height, width = self.m_scene_render.image_size_hw
        
        cam = CameraModel()
        intmat = intrinsic_matrix_by_fovx(fov_x_deg, width, height)
        cam.set_intrinsic(intmat)
        cam.set_image_size(width, height)
        cam.m_unit_cm = 100 # follows gltf standard, scene unit is in meter
        
        # set camera position and orientation
        ue_forward : np.ndarray = np.array(json_data['forward_vector'])
        ue_up : np.ndarray = np.array(json_data['up_vector'])
        ue_position : np.ndarray = np.array(json_data['position'])
        
        extmat = CameraModel.make_extrinsic_by_position_view_up(ue_position, view_dir=ue_forward, up_dir=ue_up)
        
        # z_forward = ue_forward / np.linalg.norm(ue_forward)
        # y_down = -ue_up / np.linalg.norm(ue_up)
        # x_right = np.cross(y_down, z_forward)
        # x_right /= np.linalg.norm(x_right)
        # rotmat = np.row_stack([x_right, y_down, z_forward])
        
        # # make extrinsic matrix
        # tmat = np.eye(4)
        # tmat[:3,:3] = rotmat
        # tmat[-1,:3] = ue_position
        # extmat = np.linalg.inv(tmat)
        cam.set_extrinsic(extmat)
        
        self.m_camera = cam
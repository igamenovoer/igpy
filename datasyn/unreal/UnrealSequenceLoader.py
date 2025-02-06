import numpy as np
import trimesh
import pandas as pd
import re

import igpy.common.util_trimesh as ut_trimesh
import igpy.common.shortfunc as sf
from igpy.common.calibration import CameraModel, intrinsic_matrix_by_fovx
from igpy.datasyn.ExrImage import ExrImage

from .SingleFrameData import SingleFrameData
from . import util_unreal_engine as ut_ue

class FrameInfoHeader:
    ''' headers of the per-frame info file
    '''
    
    # for old version
    QuatX = 'qx'
    QuatY = 'qy'
    QuatZ = 'qz'
    QuatW = 'qw'
    
    # for new version
    ViewX = 'view_x'
    ViewY = 'view_y'
    ViewZ = 'view_z'
    UpX = 'up_x'
    UpY = 'up_y'
    UpZ = 'up_z'
    
    # common
    PosX = 'tx'
    PosY = 'ty'
    PosZ = 'tz'
    Time = 'time'
    FrameNumber = 'frame'
    
    @classmethod
    def all_headers(cls):
        out = []
        for x in dir(cls):
            if x.startswith('__'):
                continue
            if callable(getattr(cls, x)):
                continue
            out.append(getattr(cls,x))
        return out
class RenderSequenceLoader:
    ''' loading and parsing sequence of unreal rendered exr files and camera data
    '''
    
    DefaultClassToNamePatternMatchMethod = 'search'
    
    def __init__(self) -> None:
        self.m_fnlist_exr : list[str] = None # image files without directory
        self.m_dir_exr : str = None # image directory
        
        self.m_fn_camera : str = None   #the camera information file
        self.m_camera : CameraModel = None  # camera in opencv convention
        
        # per frame info (pose data)
        self.m_motion_per_frame : pd.DataFrame = None  # per frame data
        self.m_motion_high_fq: pd.DataFrame = None  # high frequency data, more than frames
        
        # label data for the current frame
        self.m_cur_frame : SingleFrameData = None
        self.m_cur_index : int = None   # index of the current frame

        # scene
        self.m_fn_scene_structure : str = None  # scene structure filename
        self.m_scene_structure : dict = None
        self.m_fn_scene_meshes : str = None # the scene meshes filename
        self.m_scene_meshes : list[ut_trimesh.SceneMesh] = None   # the scene
        
        # class definition, mapping class name to object name pattern
        self.m_class2pattern : dict[str, str] = None
    
    @property
    def frame_data(self) -> SingleFrameData:
        return self.m_cur_frame
        
    def init_image_dir(self, dir_name : str):
        ''' scan image directory for image files
        '''
        files = sf.listdir(dir_name, file_extension='.exr', 
                   sort_by='natural', prepend_parent_path=False)
        self.m_fnlist_exr = files
        self.m_dir_exr = dir_name
        
    def load_motion_data(self, fn_motion_per_frame : str, fn_motion_high_fq : str = None):
        ''' load per frame info given the per frame info file
        '''
        info : pd.DataFrame = pd.read_csv(fn_motion_per_frame, delimiter=' ')
        self.m_motion_per_frame = info
        
        if fn_motion_high_fq:
            info : pd.DataFrame = pd.read_csv(fn_motion_high_fq, delimiter=' ')
            self.m_motion_high_fq = info
            
    def load_scene_structure(self, fn_scene_structure : str) -> dict:
        ''' load scene structure from the json file
        
        parameters
        ------------
        fn_scene_structure
            the json file that saves the scene hierarchy and relevant class names
            
        return
        ----------
        scene_struct : dict
            the scene structure
        '''
        import json
        self.m_fn_scene_structure = fn_scene_structure
        
        with open(fn_scene_structure, 'r') as fid:
            scene_struct = json.load(fid)
            self.m_scene_structure = scene_struct
            
    def load_scene_meshes(self, fn_gltf_scene : str,
                                exclude_geometry_regex : str = None,
                                include_geometry_regex : str = None) -> trimesh.Scene:
        ''' load scene meshes
        
        parameters
        --------------
        fn_gltf_scene
            the gltf scene file
        exclude_geometry_regex
            regex to designate what geometry should be ignored
        include_geometry_regex
            regex to designate what geometry should be included. If None, all geometries will be included
        '''
        
        import json
        scene : trimesh.Scene = trimesh.load_mesh(fn_gltf_scene)
        meshlist = ut_trimesh.flatten_trimesh_scene(scene, 
                                                    exclude_geometry_regex=exclude_geometry_regex,
                                                    include_geometry_regex=include_geometry_regex)
            
        self.m_scene_filename = fn_gltf_scene
        self.m_scene_meshes = meshlist
        return scene
        
    def load_camera(self, fn_camera : str):
        ''' load camera info from ue output json file, note that you must have image dir set first
        '''
        import json
        self.m_fn_camera = fn_camera
        with open(fn_camera, 'r') as fid:
            x = json.load(fid)
            fov_x_deg = x['lens']['fov_x']
            
        # load an image to get size
        assert self.m_fnlist_exr is not None, 'you must have image dir set first'
        fn_exr = '{}/{}'.format(self.m_dir_exr, self.m_fnlist_exr[0])
        img = ExrImage()
        img.open_file(fn_exr)
        height, width = img.image_size_hw
        img.close_file()
        
        cam = CameraModel()
        intmat = intrinsic_matrix_by_fovx(fov_x_deg, width, height)
        cam.set_intrinsic(intmat)
        cam.set_image_size(width, height)
        cam.m_unit_cm = 100 # follows gltf standard, scene unit is in meter
        self.m_camera = cam
        
    def get_camera_extrinsic3x4_at_frame(self, frame_index : int) -> np.ndarray:
        ''' get the standard opencv camera extrinsic at specific frame index
        
        parameters
        -------------
        frame_index
            the index of the target frame
            
        return
        ------------
        extrinsic_3x4
            3x4 opencv extrinsic matrix
        '''
        H = FrameInfoHeader
        
        pose_data = self.m_motion_per_frame.iloc[frame_index]
        ue_pos = pose_data[[H.PosX, H.PosY, H.PosZ]].to_numpy()
        ue_forward = pose_data[[H.ViewX, H.ViewY, H.ViewZ]].to_numpy()
        ue_up = pose_data[[H.UpX, H.UpY, H.UpZ]]
        
        # map to gltf coordinate
        ue2gltf = ut_ue.coordinate_ue_to_gltf()
        z_forward = ue_forward.dot(ue2gltf[:3,:3])
        z_forward /= np.linalg.norm(z_forward)
        
        y_down = -ue_up.dot(ue2gltf[:3,:3])
        y_down /= np.linalg.norm(y_down)
        
        x_right = np.cross(y_down, z_forward)
        x_right /= np.linalg.norm(x_right)
        
        # normalize and make rotation matrix
        rotmat = np.row_stack([x_right, y_down, z_forward])
        pos = ut_ue.ue_position_to_gltf_position(ue_pos, convert_unit=True)
        
        tmat = np.eye(4)
        tmat[:3,:3] = rotmat
        tmat[-1,:3] = pos
        extmat = np.linalg.inv(tmat).T[:-1,:]
        extmat = np.ascontiguousarray(extmat)
        # extmat = np.ascontiguousarray(np.row_stack([rotmat, pos]).T)
        
        return extmat
        
    def load_frame(self, frame_index : int):
        fn_exr_file ='{}/{}'.format(self.m_dir_exr, self.m_fnlist_exr[frame_index])
        
        # load image
        img : ExrImage = ExrImage()
        img.open_file(fn_exr_file)
        
        sdata = SingleFrameData()
        sdata.set_exr_image(img)
        self.m_cur_frame = sdata
        
        # set camera
        H = FrameInfoHeader
        cam = self.m_camera
        if cam and self.m_motion_per_frame is not None:
            extmat_3x4 = self.get_camera_extrinsic3x4_at_frame(frame_index)
            cam.set_extrinsic(extmat_3x4.T)
            sdata.set_camera(cam)
            
        # set scene
        if self.m_scene_meshes:
            sdata.set_scene_meshes(self.m_scene_meshes)
            
        # set scene structure
        if self.m_scene_structure:
            sdata.set_scene_structure(self.m_scene_structure)
        
        # parse labels
        sdata.set_exr_cryptomatte_max_rank(0)
        sdata.parse_scene()
        
        self.m_cur_index = frame_index
        
    @property
    def frame_count(self):
        if not self.m_fnlist_exr:
            return 0
        else:
            return len(self.m_fnlist_exr)
        
    @property
    def frame_index(self):
        return self.m_cur_index
    
    @property
    def frame_data(self) -> SingleFrameData:
        return self.m_cur_frame
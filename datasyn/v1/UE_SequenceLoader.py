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

class FrameInfoHeader(CustomEnumBase):
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
    
    class IMU(CustomEnumBase):
        # simulated imu data
        
        # gyroscope
        GyroX = 'gyro_x'
        GyroY = 'gyro_y'
        GyroZ = 'gyro_z'
        
        # accelerometer
        AccelX = 'accel_x'
        AccelY = 'accel_y'
        AccelZ = 'accel_z'
        
        # gravity
        GravityX = 'gravity_x'
        GravityY = 'gravity_y'
        GravityZ = 'gravity_z'
        
    # common
    PosX = 'tx'
    PosY = 'ty'
    PosZ = 'tz'
    Time = 'time'
    FrameNumber = 'frame'
    
class UE_SequenceLoader:
    ''' loading and parsing sequence of unreal rendered exr files and camera data
    '''
    
    def __init__(self) -> None:
        self.m_fnlist_exr : list[str] = None # image files without directory
        self.m_dir_exr : str = None # image directory
        
        self.m_fn_camera : str = None   #the camera information file
        self.m_camera : CameraModel = None  # camera in opencv convention
        
        # per frame info (pose data)
        
        # in UE coordinate
        self.m_motion_per_frame_ue : pd.DataFrame = None  # per frame data
        self.m_motion_high_fq_ue: pd.DataFrame = None  # high frequency data, more than frames
        
        # in GLTF coordinate
        self.m_motion_per_frame_gltf : pd.DataFrame = None  # per frame data
        self.m_motion_high_fq_gltf: pd.DataFrame = None  # high frequency data, more than frames
        
        # label data for the current frame
        self.m_snapshot : ue_snp.Snapshot = None
        self.m_frame_index : int = None   # index of the current frame

        # scene
        self.m_scene_desc : ue_desc.UE_SceneDescription = None
        self.m_scene_geom : ue_geom.GLTF_SceneGeometry = None
    
    @property
    def frame_data(self) -> ue_snp.UE_Snapshot:
        return self.m_snapshot
    
    @property
    def frame_count(self) -> int:
        if not self.m_fnlist_exr:
            return 0
        else:
            return len(self.m_fnlist_exr)
        
    @property
    def frame_index(self) -> int:
        return self.m_frame_index
    
    @property
    def motion_data_per_frame(self) -> pd.DataFrame:
        ''' per-frame motion data
        '''
        return self.m_motion_per_frame_gltf
    
    @property
    def motion_data_high_freq(self) -> pd.DataFrame:
        ''' high frequency motion data
        '''
        return self.m_motion_high_fq_gltf
        
    def init(self, image_dir : str):
        ''' initialize with image directory
        '''
        files = sf.listdir(image_dir, file_extension='.exr', 
                   sort_by='natural', prepend_parent_path=False)
        self.m_fnlist_exr = files
        self.m_dir_exr = image_dir
        
    def load_motion_data(self, fn_motion_per_frame : str, fn_motion_high_fq : str = None):
        ''' load per frame info given the per frame info file
        '''
        info : pd.DataFrame = pd.read_csv(fn_motion_per_frame, delimiter=' ')
        self.m_motion_per_frame_ue = info
        self.m_motion_per_frame_gltf = convert_motion_data_to_gltf_coordinate(info)
        
        if fn_motion_high_fq:
            info : pd.DataFrame = pd.read_csv(fn_motion_high_fq, delimiter=' ')
            self.m_motion_high_fq_ue = info
            self.m_motion_high_fq_gltf = convert_motion_data_to_gltf_coordinate(info)
            
    def load_scene_description(self, fn_scene_structure : str) -> ue_desc.SceneDescription:
        ''' load scene structure from the json file
        
        parameters
        ------------
        fn_scene_structure
            the json file that saves the scene hierarchy and relevant class names
            
        return
        ----------
        scene_desc
            the scene description by interpreting scene structure
        '''
        desc = ue_desc.UE_SceneDescription.create_from_desc_file(fn_scene_structure)
        desc.parse()
        
        self.m_scene_desc = desc
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
        return obj
        
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
        img = ue_render.ExrImage()
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
        
        pose_data = self.m_motion_per_frame_gltf.iloc[frame_index]
        pos = pose_data[[H.PosX, H.PosY, H.PosZ]].to_numpy()
        z_forward = pose_data[[H.ViewX, H.ViewY, H.ViewZ]].to_numpy()
        y_down = -pose_data[[H.UpX, H.UpY, H.UpZ]].to_numpy()
        x_right = np.cross(y_down, z_forward)
        x_right /= np.linalg.norm(x_right)
        rotmat = np.row_stack([x_right, y_down, z_forward])
        
        if False:
            # obsolete implementation, convert from ue to gltf line by line
            
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
        
        return extmat
        
    def parse_frame(self, frame_index : int) -> ue_snp.Snapshot:
        fn_exr_file ='{}/{}'.format(self.m_dir_exr, self.m_fnlist_exr[frame_index])
        
        # load image and set camera
        frame_data = ue_render.UE_SceneRendering.create_from_exr_file(fn_exr_file, parse_now=True)
        
        H = FrameInfoHeader
        cam = self.m_camera
        if cam and self.m_motion_per_frame_gltf is not None:
            extmat_3x4 = self.get_camera_extrinsic3x4_at_frame(frame_index)
            cam.set_extrinsic(extmat_3x4.T)
            frame_data.set_camera(cam)
        
        # make snapshot
        snp = ue_snp.UE_Snapshot()
        self.m_snapshot = snp
        snp.init(frame_data, self.m_scene_desc, self.m_scene_geom)
        snp.parse()
        
        self.m_frame_index = frame_index
        return snp
        
###
def convert_motion_data_to_gltf_coordinate(ue_motion_data : pd.DataFrame) -> pd.DataFrame:
    ''' convert motion data from ue coordinate to gltf coordinate
    
    parameters
    --------------
    ue_motion_data
        motion data in ue coordinate
        
    return
    ----------
    gltf_motion_data
        motion data in gltf coordinate
    '''
    H = FrameInfoHeader

    pose_data : pd.DataFrame = ue_motion_data
    ue_pos = pose_data[[H.PosX, H.PosY, H.PosZ]].to_numpy()
    ue_forward = pose_data[[H.ViewX, H.ViewY, H.ViewZ]].to_numpy()
    ue_up = pose_data[[H.UpX, H.UpY, H.UpZ]].to_numpy()

    # map to gltf coordinate
    ue2gltf = ut_ue.coordinate_ue_to_gltf()
    z_forward = ue_forward.dot(ue2gltf[:3,:3])
    z_forward /= np.linalg.norm(z_forward, axis=1).reshape((-1,1))

    y_down = -ue_up.dot(ue2gltf[:3,:3])
    y_down /= np.linalg.norm(y_down, axis=1).reshape((-1,1))

    x_right = np.cross(y_down, z_forward, axis=1)
    x_right /= np.linalg.norm(x_right, axis=1).reshape((-1,1))

    pos = ut_ue.ue_position_to_gltf_position(ue_pos, convert_unit=True)

    out = pose_data.copy()
    out[[H.PosX, H.PosY, H.PosZ]] = pos
    out[[H.ViewX, H.ViewY, H.ViewZ]] = z_forward
    out[[H.UpX, H.UpY, H.UpZ]] = -y_down
    
    return out
# utility functions related to unreal engine
import numpy as np
from typing import Tuple
import scipy.spatial

class UnrealDefaults:
    UnitInMeter = 0.01  # the unreal unit is cm

def make_opencv_camera_from_ue_info(ue_pos_cm : np.ndarray, 
                            ue_quat_xyzw : np.ndarray, 
                            height : int, width : int,
                            fov_x_rad : float, 
                            fov_y_rad : float = None):
    ''' create and set camera with data directly exported from UE
    
    parameters
    ------------
    ue_pos_cm
        position in ue coordinate (left-handed, unit=cm)
    ue_quat_xyzw
        quaternion in ue coordinate
    height, width
        image size
    fov_x_rad
        horizontal FOV in rad
    fov_y_rad
        vertical FOV in rad. If not set, will be computed automatically assuming single focal length
        
    return
    ----------
    camera : CameraModel
        the camera model following opencv convention, unit in m
    '''
    from igpy.common.calibration import CameraModel, intrinsic_matrix_by_fovx
    
    extmat_3x4 = ue_to_opencv_extrinsic_3x4_meter(ue_pos_cm, ue_quat_xyzw)  # now unit in meters
    intmat = ue_to_opencv_intrinsic(height, width, fov_x_rad, fov_y_rad)
    cam = CameraModel()
    cam.set_intrinsic(intmat.T.copy())
    cam.set_extrinsic(extmat_3x4.T.copy())
    cam.set_image_size(width, height)
    cam.m_unit_cm = 100.0   # unit is meter
    return cam

def camera_get_view_up_direction(rot3x3_ue : np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    ''' get view and up directions from the 3x3 rotation matrix of the UE camera.
    The UE camera uses left-handed coordinate, x+ view, z+ up, y+ right
    
    parameters
    ------------
    rot3x3_ue : np.ndarray
        3x3 rotation matrix got directly from unreal engine (note that UE uses left-handed coordinate)
        
    return
    ----------
    view_direction : np.ndarray
        the view direction 
    up_direction : np.ndarray
        the up direction
    '''
    view_dir = rot3x3_ue[:,0]
    up_dir = rot3x3_ue[:,2]
    return view_dir, up_dir

def coordinate_gltf_to_ue(gltf_transmat : np.ndarray = None) -> np.ndarray:
    ''' convert gltf coordinate into ue coordinate
    
    parameters
    ------------
    gltf_transmat : np.ndarray 4x4
        right-multiply transformation matrix in gltf coordinate conventions, that is, y+ up, z+ forward, x+ left,
        so R==transmat[:3,:3]==row_stack([x,y,z])
        
    return
    ----------
    d_transmat : np.ndarray 4x4
        conversion matrix, so that ue_transmat_rmul = gltf_transmat_rmul.dot(d_transmat)
    '''
    if gltf_transmat is None:
        gltf_transmat = np.eye(4)
        
    gltf_to_ue = np.eye(4)
    gltf_to_ue[1,:-1] = (0,0,1)
    gltf_to_ue[2,:-1] = (0,1,0)
    return gltf_transmat.dot(gltf_to_ue)

def coordinate_ue_to_gltf(ue_transmat : np.ndarray = None) -> np.ndarray:
    ''' convert ue coordinate into gltf coordinate
    
    parameters
    ------------
    ue_transmat : np.ndarray 4x4
        right-multiply transformation matrix in ue coordinate conventions, that is, x+ forward, z+ up, y+ right
        so R==transmat[:3,:3]==row_stack([x,y,z])
        
    return
    ----------
    d_transmat : np.ndarray 4x4
        conversion matrix, so that gltf_transmat = ue_transmat.dot(d_transmat)
    '''
    if ue_transmat is None:
        ue_transmat = np.eye(4)
        
    dT = coordinate_gltf_to_ue()
    dT[:3,:3] = dT[:3,:3].T #invert rotation
    return ue_transmat.dot(dT)

def ue_quaternion_to_gltf_orientation(x,y,z,w) -> np.ndarray:
    ''' convert quaternion of UE to gltf 3x3 orientation matrix (right-mul).
    UE convention is x+ forward, y+ right, z+ up, GLTF convention is  x+ left, y+ up, z+ forward
    
    parameter
    -------------
    x,y,z,w
        the ue quaternion
        
    return
    ---------
    gltf_orientation
        out = row_stack([left, up, forward]), in gltf convention
    '''
    
    # note that ue uses left-handed coordinate, but from_quat() uses right-handed coordinate
    rotmat = scipy.spatial.transform.Rotation.from_quat([x,y,z,w]).as_matrix().T    #.T converts it to right-mul format
    rotmat = rotmat[:,[0,2,1]]
    forward = rotmat[0]
    left = -rotmat[1]
    up = rotmat[2]
    out = np.row_stack([left, up, forward])
    # out = rotmat[:,[0,2,1]][[0,2,1],:]
    return out

def ue_position_to_gltf_position(ue_pts : np.ndarray, convert_unit: bool = True) -> np.ndarray:
    unit = 1.0/100 if convert_unit else 1.0
    return np.array(ue_pts).dot(coordinate_ue_to_gltf()[:3,:3]) * unit # note that UE unit is cm, GLTF unit is m

def ue_to_opencv_extrinsic_3x4_meter(ue_pos_cm : np.ndarray, 
                                     ue_quat_xyzw : np.ndarray) -> np.ndarray:
    ''' convert UE position and quaternion into opencv camera extrinsic matrix, unit in meters
    
    parameters
    -----------
    ue_pos_cm
        position in ue coordinate (cm unit)
    ue_quat_xyzw
        quaterion in ue coordinate
        
    return
    -----------
    extrinsic_matrix
        3x4 opencv extrinsic matrix
    '''
    ue_pos_cm = ue_position_to_gltf_position(ue_pos_cm, convert_unit=True)
    rotmat_gltf = ue_quaternion_to_gltf_orientation(*ue_quat_xyzw)
    
    left, up, forward = rotmat_gltf
    extmat_opencv = np.column_stack([-left, -up, forward, ue_pos_cm])
    return extmat_opencv

def ue_to_opencv_intrinsic(height : int, width : int, 
                           fov_x_rad : float, fov_y_rad : float = None):
    ''' convert UE camera intrinsic to opencv intrinsic
    
    return
    ---------
    intrinsic_matrix
        the 3x3 opencv intrinsic matrix
    '''
    fx = width/2 / np.tan(fov_x_rad/2)
    if fov_y_rad is None:
        fy = fx
    else:
        fy = height/2 / np.tan(fov_y_rad/2)
        
    intmat = np.eye(3)
    intmat[0,0] = fx
    intmat[1,1] = fy
    intmat[:-1,-1] = (width/2, height/2)
    return intmat
    
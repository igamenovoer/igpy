# functions and classes related to stereo reconstruction
import numpy as np
import cv2
from enum import Enum
from .calibration import CameraModel
from . import shortfunc as sf

def find_stereo_image_pair_in_dir(dir_left, dir_right, extention = 'jpg', 
                           sync_file_pair : list = None, max_time_difference_sec = 1.1, ignore_case = False):
    ''' find images in left and right image folder that make up a stereo pair.
    You must provide a pair of files in 
    We assume that the last pair of images in natural order is a stereo pair, and use that to
    sync the time of the two cameras.
    
    parameters
    --------------
    dir_left, dir_right
        the image folder of the left and right cameras
    sync_file_pair
        [left_filename, right_filename] a pair of images in left and right folders, such that
        they form a stereo pair. If not specified, the last image in natural order in both 
        folders are assumed to make a pair
    max_time_difference_sec
        if the time between two shots is larger than this, the two images are not considered a stereo pair
    extention
        the file extension of the images
    ignore_case
        if True, we treat everything as lowercase letters. Otherwise, preserve its original content.
        
    return
    ------------
    fnlist_left, fnlist_right
        image filenames including path in left and right folders that form pairs of stereo images
    '''
    import igpy.common.shortfunc as sf
    import igpy.common.stereo as sto
    
    get_files = lambda x: sf.listdir(x, name_pattern=r'.+\.{0}'.format(extention),
                                     include_dir=False,sort_by='string',
                                     ignore_case=ignore_case, prepend_parent_path=True)
    fnlist_left : list = get_files(dir_left)
    fnlist_right : list = get_files(dir_right)
    
    if sync_file_pair is not None:
        fn_left, fn_right = sync_file_pair
        idxref_left = np.nonzero([fn_left in x for x in fnlist_left])[0]
        idxref_right = np.nonzero([fn_right in x for x in fnlist_right])[0]
    else:
        idxref_left = len(fnlist_left)-1
        idxref_right = len(fnlist_right)-1
        
    idxleft, idxright = sto.find_stereo_image_pair(fnlist_left,fnlist_right,idxref_left, idxref_right, max_time_difference_sec=max_time_difference_sec)
    
    out_left = [fnlist_left[x] for x in idxleft]
    out_right = [fnlist_right[x] for x in idxright]
    
    return out_left, out_right

def find_stereo_image_pair(
    fnlist_left : list, fnlist_right : list,
    idxref_left : int,  idxref_right : int,
    max_time_difference_sec = 0.1):
    ''' find stereo image pairs from two sets of files.
    The stereo image dataset may have some orphan image, which only has a left or right view
    but not both, thus it will be incorrect to match the images by order.
    This function matches the images from left camera and right camera by timestamp,
    and outputs the image pairs.
    Exif info must exist for this function to work
    
    parameters
    ----------------
    fnlist_left, fnlist_right
        the filenames of the left and right images
    idxref_left, idxref_right
        index of the reference image in left and right collection. It means that
        fnlist_left[idxref_left] and fnlist_right[idxref_right] are known to be taken at the same time
        by left and right camera. This image pair is used as a pivot the synchronize the time of the
        two cameras.
    max_time_difference_sec
        the difference of the timestamps of two images must be within this threshold or otherwise
        they are not considered matched.
        
    return
    ---------------
    idxleft, idxright
        image pairs such that (fnlist_left[idxleft], fnlist_right[idxright]) is a stereo image pair
    '''
    
    from . import image_processing as ip
    import scipy.spatial as spatial
    
    # find shot time of all files
    times_left = []
    times_right = []
    for fn in fnlist_left:
        ex = ip.ExifInfo.init_from_image_file(fn)
        times_left.append(ex.datetime_original)
        
    for fn in fnlist_right:
        ex = ip.ExifInfo.init_from_image_file(fn)
        times_right.append(ex.datetime_original)
        
    # sync the time
    t_ref_left = times_left[idxref_left]
    t_ref_right = times_right[idxref_right]
    
    dt_left = np.array([(x - t_ref_left).total_seconds() for x in times_left])
    dt_right = np.array([(x - t_ref_right).total_seconds() for x in times_right])
    
    kd = spatial.cKDTree(dt_right.reshape((-1,1)))
    dist, idx = kd.query(dt_left.reshape((-1,1)))
    
    mask = dist.flatten() <= max_time_difference_sec
    idxleft = np.nonzero(mask)[0]
    idxright = idx.flatten()[mask]
    
    return idxleft, idxright
    

class StereoCamera:
    ''' two cameras making a stereo system that can reconstruct 3d from images
    '''
    class CameraType(Enum):
        # original camera
        LEFT_ORIGINAL =  1
        RIGHT_ORIGINAL  = 2
        
        # rectified camera
        LEFT_RECTIFIED = 3
        RIGHT_RECTIFIED  = 4
    
    def __init__(self):
        self.m_left_camera = CameraModel()
        self.m_right_camera = CameraModel()
        
        # ======= rectification info ============
        # to rectify, rotate the left camera by this right-mul  matrix
        self.m_left_rectify_rotation : np.ndarray = None
        
        # 4x3 projection matrix that projects the points to rectified image
        self.m_left_rectify_projection : np.ndarray = None
        
        self.m_right_rectify_rotation : np.ndarray = None
        self.m_right_rectify_projection : np.ndarray = None
        
        # the Q matrix returned by opencv for disparity map reprojection
        self.m_rectify_Q : np.ndarray = None
        
        # left and right rectification map
        self.m_left_rectify_map : list = None
        self.m_right_rectify_map : list = None
        
    @property
    def image_size_hw(self):
        ''' get image size in (height, width)
        '''
        camera = self.m_left_camera
        if camera is None:
            camera = self.m_right_camera
            
        if camera is None:
            return np.zeros(2, dtype=int)
        else:
            return np.array([camera.image_height, camera.image_width])
        
    def set_cameras(self, left_camera:CameraModel, right_camera:CameraModel):
        ''' set the left and right cameras
        '''
        self.m_left_camera = left_camera
        self.m_right_camera = right_camera
        assert left_camera.image_height == right_camera.image_height and left_camera.image_width == right_camera.image_width, 'camera resolution is different'
        
    def prepare_rectify(self):
        ''' compute rectification info, only possible when the image sizes are the same
        '''
        
        assert self.m_left_camera.image_height == self.m_right_camera.image_height
        assert self.m_left_camera.image_width == self.m_right_camera.image_width
        
        # transform both cameras so that the left camera extrinsic is identity
        left_extrinsic = self.m_left_camera.extrinsic_4x4
        right_extrinsic = self.m_right_camera.extrinsic_4x4
        
        # extrinsic after camera transformation:
        # inv(inv(extmat).dot(T)) = inv(T).dot(extmat)
        extrinsic = np.linalg.inv(left_extrinsic).dot(right_extrinsic)
        R = extrinsic[:3,:3].T
        t = extrinsic[-1,:3]
        
        imgsize_wh = (self.m_left_camera.image_width, self.m_left_camera.image_height)
        res = cv2.stereoRectify(
            self.m_left_camera.intrinsic_3x3.T, 
            self.m_left_camera.distortion,
            self.m_right_camera.intrinsic_3x3.T, 
            self.m_right_camera.distortion,
            imgsize_wh, R, t, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, 0
        )
        
        R1, R2, P1, P2, Q = res[:5]
        self.m_left_rectify_rotation = R1 #actually R1.T.T
        self.m_left_rectify_projection = P1.T
        
        self.m_right_rectify_rotation = R2  #it is acutally R2.T.T
        self.m_right_rectify_projection = P2.T
        
        self.m_rectify_Q = Q
        
        m1, m2 = cv2.initUndistortRectifyMap(
            self.m_left_camera.intrinsic_3x3.T, 
            self.m_left_camera.distortion,
            R1, P1[:3,:3], imgsize_wh, cv2.CV_32FC1)
        self.m_left_rectify_map = [m1, m2]
        
        m1, m2 = cv2.initUndistortRectifyMap(
            self.m_right_camera.intrinsic_3x3.T, 
            self.m_right_camera.distortion,
            R2, P2[:3,:3], imgsize_wh, cv2.CV_32FC1)
        self.m_right_rectify_map = [m1, m2]
        
    def rectify_left_points(self, pts : np.ndarray) -> np.ndarray:
        ''' rectify 2d points in left image
        
        parameters
        --------------
        pts
            Nx2 xy points in left image. You must make sure the points are all inside the image.
        '''
        pts = np.atleast_2d(pts)
        R = self.m_left_rectify_rotation
        P = self.m_left_rectify_projection.T
        _pts = pts.reshape((-1,1,2)).astype(np.float)
        res = cv2.undistortPoints(_pts, self.m_left_camera.intrinsic_3x3.T,self.m_left_camera.distortion,None, R, P)
        output = res.reshape((-1,2))
        return output
    
    def rectify_right_points(self, pts : np.ndarray) -> np.ndarray:
        ''' rectify 2d points in right image
        
        parameters
        --------------
        pts
            Nx2 xy points in right image. You must make sure the points are all inside the image.
        '''
        pts = np.atleast_2d(pts)
        R = self.m_right_rectify_rotation
        P = self.m_right_rectify_projection.T
        _pts = pts.reshape((-1,1,2)).astype(np.float)
        res = cv2.undistortPoints(_pts, self.m_right_camera.intrinsic_3x3.T,self.m_right_camera.distortion,None,R, P)
        output = res.reshape((-1,2))
        return output        
        
    def rectify_left_image(self, img : np.ndarray, interp_method = cv2.INTER_NEAREST) -> np.ndarray:
        ''' rectify left image using the rectification info.
        Definition of stereo rectification is that to transform the image so that the epipolar lines are horizontal and have the same y coordinate.
        
        parameters
        --------------
        img : np.ndarray
            the image to rectify, can be binary or color image
        interp_method : int, opencv interpolation method
            the interpolation method used to fill in the pixels in the rectified image, defined in opencv as cv2.INTER_XXX
            
        return
        ----------
        output : np.ndarray
            the rectified image using cv2.remap
        '''
        if img.dtype == np.bool:
            tmp = img.astype(np.uint8) * 255
            output = cv2.remap(tmp, self.m_left_rectify_map[0],  self.m_left_rectify_map[1], interp_method)
            output = output > 0
        else:
            output = cv2.remap(img, self.m_left_rectify_map[0],  self.m_left_rectify_map[1], interp_method)
            
        return output
    
    def rectify_right_image(self, img : np.ndarray, interp_method = cv2.INTER_NEAREST) -> np.ndarray:
        if img.dtype == np.bool:
            tmp = img.astype(np.uint8) * 255
            output = cv2.remap(tmp, self.m_right_rectify_map[0],  self.m_right_rectify_map[1], interp_method)
            output = output > 0
        else:
            output = cv2.remap(img, self.m_right_rectify_map[0],  self.m_right_rectify_map[1], interp_method)        
        
        return output
    
    def get_camera(self, camera_type, extrinsic_source_camera = None) -> CameraModel:
        ''' get a camera
        
        parameters
        ----------------
        camera_type
            the camera type to get
        extrinsic_source_camera
            the extrinsic of the retrieved camera converts points in extrinsic_source_camera's coordinate system
            into points in its own coordinate system
        '''
        
        if extrinsic_source_camera is None:
            extrinsic_source_camera = camera_type
        
        intmat = self.get_intrinsic(camera_type)
        extmat = self.get_extrinsic(extrinsic_source_camera, camera_type)
        dist = self.get_distortion(camera_type)
        
        camera = CameraModel()
        camera.set_distortion(dist)
        camera.set_extrinsic(extmat)
        camera.set_intrinsic(intmat)
        camera.set_image_size(self.m_left_camera.image_width, self.m_left_camera.image_height)
        camera.m_unit_cm = self.m_left_camera.physical_unit_cm
        
        return camera
    
    def get_distortion(self, camera_type):
        if camera_type == StereoCamera.CameraType.LEFT_ORIGINAL:
            return self.m_left_camera.distortion
        elif camera_type == StereoCamera.CameraType.RIGHT_ORIGINAL:
            return self.m_right_camera.distortion
        else:
            return np.zeros(5)
        
    def get_intrinsic(self, camera_type):
        ''' get the right-multiply camera intrinsic matrix
        '''
        if camera_type == StereoCamera.CameraType.LEFT_ORIGINAL:
            return np.array(self.m_left_camera.intrinsic_3x3)
        elif camera_type == StereoCamera.CameraType.LEFT_RECTIFIED:
            return np.array(self.m_left_rectify_projection[:3,:3])
        elif camera_type == StereoCamera.CameraType.RIGHT_ORIGINAL:
            return np.array(self.m_right_camera.intrinsic_3x3)
        elif camera_type == StereoCamera.CameraType.RIGHT_RECTIFIED:
            return np.array(self.m_right_rectify_projection[:3,:3])
        
    def get_extrinsic(self, source_camera, target_camera = None):
        ''' get an 4x4 extrinsic matrix that converts coordinates of source camera to target camera. Specifically, let M be the extrinsic matrix, and P be a point in the coordinate system of source camera, then P.dot(M) is the point in target camera coordinate system.
        '''
        if target_camera is None:
            target_camera = source_camera
            
        if source_camera == target_camera:
            return np.eye(4)
        
        # use the right camera as world coordinate, express all other camera frames
        tmat_right = self.m_right_camera.world_transform
        tmat_left = self.m_left_camera.world_transform
        
        tmat_right_rectify_wrt_right = np.eye(4)
        tmat_right_rectify_wrt_right[:3,:3] = self.m_right_rectify_rotation
        tmat_right_rectify_wrt_world = tmat_right_rectify_wrt_right.dot(tmat_right)
        
        tmat_left_rectify_wrt_right_rectify = np.eye(4)
        K = self.m_right_rectify_projection[:3,:3]
        p = self.m_right_rectify_projection[-1,:3]
        pos = p.dot(np.linalg.inv(K))
        
        tmat_left_rectify_wrt_right_rectify[-1,:3] = pos
        tmat_left_rectify_wrt_world = tmat_left_rectify_wrt_right_rectify.dot(tmat_right_rectify_wrt_world)
        
        type2frame = {
            StereoCamera.CameraType.LEFT_ORIGINAL: tmat_left,
            StereoCamera.CameraType.LEFT_RECTIFIED: tmat_left_rectify_wrt_world,
            StereoCamera.CameraType.RIGHT_ORIGINAL: tmat_right,
            StereoCamera.CameraType.RIGHT_RECTIFIED: tmat_right_rectify_wrt_world
        }
        
        output = type2frame[source_camera].dot(np.linalg.inv(type2frame[target_camera]))
        return output
        
    
    def triangulate_points(self, pts_xy_left, pts_xy_right, 
                           left_camera_type, 
                           right_camera_type, 
                           output_camera_type):
        ''' triangulate points given correspondences in left and right images
        
        parameters
        ----------------
        pts_xy_left, pts_xy_right
            xy points left and right images
        left_camera_type, right_camera_type
            the type of camera that captures the image
        output_camera_type
            the output 3d points are in the coordinate system of which camera
            
        return
        ---------
        pts_3d
            3d points of the correspondences
        '''
        pts_xy_left = pts_xy_left.astype(np.float32)
        pts_xy_right = pts_xy_right.astype(np.float32)
        
        left_intrinsic = self.get_intrinsic(left_camera_type)
        left_extrinsic = self.get_extrinsic(output_camera_type, left_camera_type)
        left_distortion = self.get_distortion(left_camera_type)
        
        right_intrinsic =  self.get_intrinsic(right_camera_type)
        right_extrinsic = self.get_extrinsic(output_camera_type, right_camera_type)
        right_distortion = self.get_distortion(right_camera_type)
        
        left_proj = left_extrinsic[:,:-1].dot(left_intrinsic)
        right_proj = right_extrinsic[:,:-1].dot(right_intrinsic)
        
        # undistort points before triangulation, otherwise points are not
        # on the camera ray
        if left_distortion is not None and left_distortion.any() and left_camera_type != self.CameraType.LEFT_RECTIFIED:
            pts_xy_left = cv2.undistortPoints(pts_xy_left, left_intrinsic.T, left_distortion, None, None, left_intrinsic.T).reshape((-1,2))
            
        if right_distortion is not None and right_distortion.any() and right_camera_type != self.CameraType.RIGHT_RECTIFIED:
            pts_xy_right = cv2.undistortPoints(pts_xy_right, right_intrinsic.T, right_distortion, None, None, right_intrinsic.T).reshape((-1,2))
        
        pts = cv2.triangulatePoints(left_proj.T, right_proj.T, pts_xy_left.T, pts_xy_right.T)
        pts = (pts[:3,:]/pts[-1]).T
        return pts
    
    @property
    def physical_unit_cm(self):
        return self.m_left_camera.physical_unit_cm
    
def planar_image_project(img_src, camera_src : CameraModel, camera_dst: CameraModel, p0:np.ndarray, normal:np.ndarray):
    ''' assuming the image is on a plane perpendicular to the source camera, compute the image on the dst camera
    
    return
    -----------
    img
        the reprojected image
    '''
    p0 = np.array(p0)
    normal /= np.linalg.norm(normal)
    homography_matrix = compute_homography_of_plane_projection(camera_src, camera_dst, p0, normal)
    
    # undistort the image
    img_undist = camera_src.undistort_image(img_src)
    
    # project
    img_warp = cv2.warpPerspective(img_undist, np.ascontiguousarray(homography_matrix.T), (camera_dst.image_width, camera_dst.image_height))
    
    # distort
    img_dist = camera_src.distort_image(img_warp)
    
    return img_dist
    
def planar_image_project_by_distance(img_src, camera_src : CameraModel, camera_dst: CameraModel, distance_to_src):
    ''' assuming the image is on a plane perpendicular to the source camera, compute the image on the dst camera.
    The plane is assumed to be facing the src camera at a specified distance
    
    return
    -----------
    img
        the reprojected image
    '''
    p0 = np.array([0,0,distance_to_src], dtype=float)
    normal = np.array([0,0,-1], dtype=float)
    homography_matrix = compute_homography_of_plane_projection(camera_src, camera_dst, p0, normal)
    
    # undistort the image
    img_undist = camera_src.undistort_image(img_src)
    
    # project
    img_warp = cv2.warpPerspective(img_undist, np.ascontiguousarray(homography_matrix.T), (camera_dst.image_width, camera_dst.image_height))
    
    # distort
    img_dist = camera_dst.distort_image(img_warp)
    
    return img_dist
        
    
def compute_homography_of_plane_projection(cam_left:CameraModel, cam_right:CameraModel, p0, normal):
    ''' Given a known 3d plane, its image on two cameras are related by a homography transform (3x3 matrix). This function computes that transformation matrix.
    
    parameters
    ----------------
    cam_left, cam_right
        the left and right camera
    p0, normal
        a point and a normal of the plane
        
    return
    ------------
    homography_matrix
        3x3 matrix in right-mul format
    '''
    
    p0 = np.array(p0).flatten()
    normal = np.array(normal)
    normal /= np.linalg.norm(normal)
    
    # create a frame over the plane
    plane_axis = sf.make_frame(normal)
    plane_frame = np.eye(4)
    plane_frame[:3,:3] = plane_axis
    plane_frame[-1,:3] = p0
    
    # plane frame wrt left camera
    E1 = plane_frame.dot(cam_left.extrinsic_4x4)
    E1 = E1[[0,1,3],:-1]
    K1 = cam_left.intrinsic_3x3
    
    # plane frame wrt right camera
    E2 = plane_frame.dot(cam_right.extrinsic_4x4)
    E2 = E2[[0,1,3],:-1]    # when using the plane frame as coordinate frame, z=0, so 3rd row of extrinsic is not used
    K2 = cam_right.intrinsic_3x3
    
    homography_matrix = np.linalg.inv(K1).dot(np.linalg.inv(E1)).dot(E2).dot(K2)
    return homography_matrix
    
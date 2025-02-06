import numpy as np
import cv2
import copy

class CalibPatternType:
    CHESSBOARD = 1,
    SYMMETRIC_CIRCLE = 2,
    ASYMMETRIC_CIRCLE = 3

class CameraModel:
    ''' representing a camera with intrinsic, extrinsic and distortion.
    Note that the intrinsic and extrinsic matrices are in right-multiply,
    that is, let P be a point and M be the intrinsic, then P.dot(M) projects
    the point to camera. This is transpose of the opencv camera matrices.
    Extrinsic matrix is in 4x4 format.
    '''
    def __init__(self):
        self.m_intrinsic : np.ndarray = np.eye(3)
        self.m_extrinsic : np.ndarray = np.eye(4)
        self.m_distortion : np.ndarray = np.zeros(5)
        self.m_imgsize_hw : np.ndarray = np.zeros(2, dtype = int)
        self.m_unit_cm = None   #unit of the extrinsic
        
        # (xmap, ymap) to map the undistorted image to distorted image
        self.m_distort_map : list = None
        
    @property
    def extrinsic_4x4(self):
        return np.array(self.m_extrinsic)
    
    @property
    def extrinsic_4x3(self):
        return np.array(self.m_extrinsic[:,:-1])
    
    @property
    def world_transform(self) -> np.ndarray:
        return np.linalg.inv(self.m_extrinsic)
    
    @property
    def position(self) -> np.ndarray:
        ''' world-space position
        '''
        return self.world_transform[-1,:-1]
    
    @property
    def forward_vector(self) -> np.ndarray:
        world_rot = self.m_extrinsic[:-1,:-1].T # ==inv
        z = world_rot[-1]
        return z
    
    @property
    def up_vector(self) -> np.ndarray:
        world_rot = self.m_extrinsic[:-1,:-1].T # ==inv
        y = world_rot[1]
        return -y
    
    @property
    def intrinsic_3x3(self):
        return np.array(self.m_intrinsic)
    
    @property
    def intrinsic_4x4(self):
        x = np.eye(4)
        x[:3,:3] = self.m_intrinsic
        return x
    
    @property
    def image_width(self):
        return self.m_imgsize_hw[1]
    
    @property
    def image_height(self):
        return self.m_imgsize_hw[0]
    
    @property
    def distortion(self):
        return np.array(self.m_distortion)
    
    @property
    def physical_unit_cm(self):
        return self.m_unit_cm
    
    @property
    def fov_y(self) -> float:
        fy = self.m_intrinsic[1,1]
        cy = self.m_intrinsic[-1,1]
        h = self.image_height
        
        # we DO NOT assume the focal point is at center
        a = np.sqrt(fy**2 + cy**2)
        b = np.sqrt(fy**2 + (h-cy)**2)
        c = h
        cos_theta = np.clip((a**2+b**2-c**2)/(2*a*b),-1,1)
        theta = np.arccos(cos_theta)
        return theta
    
    @property
    def fov_x(self) -> float:
        fx = self.m_intrinsic[0,0]
        cx = self.m_intrinsic[-1,0]
        w = self.image_width
        
        # we DO NOT assume the focal point is at center
        a = np.sqrt(fx**2 + cx**2)
        b = np.sqrt(fx**2 + (w-cx)**2)
        c = w
        cos_theta = np.clip((a**2+b**2-c**2)/(2*a*b),-1,1)
        theta = np.arccos(cos_theta)
        return theta
    
    def set_image_size(self, width, height):
        self.m_imgsize_hw = np.array([height, width])
        
    def project_points(self, pts) -> np.ndarray:
        ''' project 3d points to the image of this camera
        '''
        assert self.m_imgsize_hw[0]>0, 'image size not set'
        
        rvec = cv2.Rodrigues(self.m_extrinsic[:3,:3].T)[0]
        tvec = self.m_extrinsic[-1,:3]
        pts = pts.astype(np.float)
        res = cv2.projectPoints(pts, rvec, tvec, self.m_intrinsic.T, self.m_distortion)
        return res[0].reshape((-1,2))
    
    @staticmethod
    def make_extrinsic_by_position_view_up(pos: np.ndarray, 
                                          view_dir : np.ndarray, 
                                          up_dir : np.ndarray) -> np.ndarray:
        ''' get 4x3 right-multiply extrinsic matrix by position, view and up vector, in OpenCV convention
        '''
        y = -np.atleast_1d(up_dir).flatten()
        z = np.atleast_1d(view_dir).flatten()
        x = np.cross(y,z)
        
        # normalization
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)
        
        # construct extrinsic
        pos = np.atleast_1d(pos).flatten()
        extmat = np.row_stack((x,y,z,pos))
        return extmat
        
    def set_extrinsic(self, extrinsic):
        ''' set the right-multiply extrinsic, which can be 4x3 or 4x4 matrix.
        '''
        if np.allclose(extrinsic.shape, (4,4)):
            self.m_extrinsic = np.array(extrinsic)
        elif np.allclose(extrinsic.shape, (4,3)):
            x = np.eye(4)
            x[:,:-1] = extrinsic
            self.m_extrinsic = x
        else:
            assert False, 'incorrect shape of input extrinsic'
            
    def set_intrinsic(self, intrinsic):
        ''' set the right-multiply intrinsic, which can be 4x4 or 3x3
        '''
        if np.allclose(intrinsic.shape, (4,4)):
            self.m_intrinsic = np.array(intrinsic[:3,:3])
        elif np.allclose(intrinsic.shape, (3,3)):
            self.m_intrinsic = np.array(intrinsic)
        else:
            assert False, 'incorrect shape of input intrinsic'
            
    def set_distortion(self, distortion):
        ''' set the distortion coefficients, following opencv convention, which can
        be 4,5,8,12,14 elements, or None to clear distortion info
        '''
        if distortion is None:
            self.m_distortion = np.zeros(5)
        
        distortion = np.array(distortion).flatten()
        n = len(distortion)
        assert n==4 or n==5 or n==8 or n==12 or n==14, 'number of elements in distortion vector is wrong'
        self.m_distortion = np.array(distortion)
        self.m_distort_map = None
        
    def build_distort_map(self):
        ''' build a distort map used for applying camera distortion to an image
        '''
        xs = np.arange(self.image_width)
        ys = np.arange(self.image_height)
        xx, yy = np.meshgrid(xs, ys)
        
        if self.m_distortion is None or not self.m_distortion.any():
            self.m_distort_map = [xx.astype(np.float), yy.astype(np.float)]
            return
        
        # undistort all the points
        cam_matrix = np.ascontiguousarray(self.intrinsic_3x3.T)
        pts = np.column_stack((xx.flat, yy.flat)).astype(np.float)
        pts_undist = cv2.undistortPoints(pts, cam_matrix, self.m_distortion, None, None, cam_matrix).reshape((-1,2))
        
        x_undist = pts_undist[:,0].reshape((self.image_height, self.image_width))
        y_undist = pts_undist[:,1].reshape((self.image_height, self.image_width))
        self.m_distort_map = [x_undist.astype(np.float32), y_undist.astype(np.float32)]
        
    def distort_image(self, img) -> np.ndarray:
        ''' apply camera distortion to an image
        '''
        if self.m_distortion is None or not self.m_distortion.any():
            return img
        
        if self.m_distort_map is None:
            self.build_distort_map()
            
        xmap, ymap = self.m_distort_map
        output = cv2.remap(img, xmap, ymap, cv2.INTER_NEAREST)
        return output
        
    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        ''' undistort the image using distortion coefficients
        '''
        if self.m_distortion is None or len(self.m_distortion) == 0:
            return img
        else:
            output = cv2.undistort(img, self.intrinsic_3x3.T, self.distortion)
            return output
        
    def get_undistort_map(self) -> np.ndarray:
        ''' get a backward mapping (xmap, ymap), such that undistort_image[i,j] = image[ymap[i,j], xmap[i,j]], you can use the mapping in opencv remap() to get the undistorted image
        
        return
        -------------
        xmap, ymap
            coordinate maps such that undistort_image[i,j] = image[ymap[i,j], xmap[i,j]]
        '''
        h, w = self.m_imgsize_hw
        if self.m_distortion is None or len(self.m_distortion) == 0:
            xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        else:
            camera_matrix = self.intrinsic_3x3.T
            res = cv2.initUndistortRectifyMap(camera_matrix, self.distortion, np.eye(3), camera_matrix, (w,h), cv2.CV_32FC2)
            xmap = res[:,:,0]
            ymap = res[:,:,1]
        return xmap, ymap
    
    def undistort_points(self, pts:np.ndarray) -> np.ndarray:
        ''' undistort image xy points
        
        parameters
        --------------
        pts
            a list of xy points in image space
        
        return
        -----------
        pts_undistort
            the undistorted points
        '''
        if self.m_distortion is None or len(self.m_distortion) == 0:
            return pts
        else:
            output = cv2.undistortPoints(pts.reshape((-1,1,2)).astype(np.float32), self.intrinsic_3x3.T, self.distortion, None, None, self.intrinsic_3x3.T)
            output = output.reshape((-1,2))
        return output
    
    def convert_pts_world_to_camera(self, pts_world: np.ndarray) -> np.ndarray:
        ''' convert points from world coordinate to camera coordinate
        '''
        pts_world = np.atleast_2d(pts_world)
        R = self.m_extrinsic[:3,:3]
        t = self.m_extrinsic[-1,:3]
        pts_cam = pts_world.dot(R) + t
        return pts_cam
    
    def convert_pts_camera_to_world(self, pts_camera : np.ndarray) -> np.ndarray:
        ''' convert points from camera space to world space
        '''
        pts_camera = np.atleast_2d(pts_camera)
        tmat = self.world_transform
        R = tmat[:3,:3]
        t = tmat[-1,:3]
        pts_world = pts_camera.dot(R)+t
        return pts_world
    
    def convert_pts_world_to_image(self, pts_world : np.ndarray, 
                                   near_clip_distance : float = None)->np.ndarray:
        ''' convert points from world space to image space
        
        parameters
        --------------
        pts_world
            Nx3 points in world space
        near_clip_distance
            If the distance from the point to image plane is smaller than this, it will be clipped
            to this value. Note that if not set, points that are behind the camera may cause problem.
            
        return
        ---------
        pts_image
            image points in xy coordinate
        '''
        pts_camera = self.convert_pts_world_to_camera(pts_world)
        pts_image = self.convert_pts_camera_to_image(pts_camera, near_clip_distance=near_clip_distance)
        return pts_image
    
    def convert_pts_camera_to_image(self, pts_camera : np.ndarray, 
                                    near_clip_distance : float = None) -> np.ndarray:
        ''' convert points from camera space to image space
        
        parameters
        --------------
        pts_camera
            Nx3 points in camera space
        near_clip_distance
            If the distance from the point to image plane is smaller than this, it will be clipped
            to this value. Note that if not set, points that are behind the camera may cause problem.
            
        return
        ---------
        pts_image
            image points in xy coordinate
        '''
        pts_camera = np.atleast_2d(pts_camera).copy()
        if near_clip_distance is not None:
            z = pts_camera[:,-1]
            z = np.clip(z, near_clip_distance)
            pts_camera[:,-1] = z
        imgpts = pts_camera.dot(self.m_intrinsic)
        imgpts /= imgpts[:,-1:]
        output = np.ascontiguousarray(imgpts[:,:-1])
        return output
        
    
    def clone(self):
        import copy
        output = CameraModel()
        for key, val in vars(self).items():
            setattr(output, key, copy.deepcopy(val))
        return output

class StereoCameraCalibrator:
    ''' calibrate two cameras given a series of chessboard image files
    '''
    class Options:
        def __init__(self):
            self.use_subpixel_refinement = True
            self.subpixel_refinement_window = (30,30) 
            
    def __init__(self):
        self.m_left_image_filenames : list = None
        self.m_left_corner_per_image : list = None
        self.m_left_camera = CameraModel()
        
        self.m_right_image_filenames : list = None
        self.m_right_corner_per_image : list = None
        
        # the extrinsic matrix is relative to the left camera
        # that is, it converts coordinate from left to right
        self.m_right_camera = CameraModel()
        
        self.m_reprojection_error = 0
        
        # nx2 matrix where error[i]=(left_error, right_error), error of view i in left and right
        self.m_reprojection_error_per_view = None
        
        # the coordinate frame of left camera relative to right camera
        # self.m_left_transmat : np.ndarray = None
        
        # fundamental and essential matrices
        self.m_essential_matrix : np.ndarray = None
        self.m_fundamental_matrix : np.ndarray = None
        self.m_pattern_size = None  # in (n_objects_per_row, n_objects_per_column)
        self.m_options = StereoCameraCalibrator.Options()
        
    @property
    def left_camera(self):
        return self.m_left_camera
    
    @property
    def right_camera(self):
        return self.m_right_camera
        
    def set_left_camera(self, camera : CameraModel):
        ''' set the left camera
        '''
        self.m_left_camera = copy.deepcopy(camera)
        
    def set_right_camera(self, camera : CameraModel):
        ''' set the right camera
        '''
        self.m_right_camera = copy.deepcopy(camera)
        
    def set_image_files(self, imgfiles_left, imgfiles_right, pattern_size):
        ''' initialize with the image files for left and right camera
        
        parameters
        ---------------
        imgfiles_left, imgfiles_right
            chessboard or circle pattern image files for left and right camera, imgfiles_left[i] and imgfiles_right[i] are images captured at the same instant.
        pattern_size
            board size (counting the cross corners) in (width, height)
        '''
        assert len(imgfiles_left) ==  len(imgfiles_right), 'must have same number of input images'
        
        self.m_left_image_filenames = list(imgfiles_left)
        self.m_right_image_filenames = list(imgfiles_right)
        self.m_pattern_size = tuple(pattern_size)
        
    def set_pattern_size(self, pattern_size_wh):
        ''' set the pattern size in (width, height)
        '''
        self.m_pattern_size = tuple(pattern_size_wh)
        
    def calibrate(self, verbose = False):
        ''' find the relative transformation between the cameras
        '''
        assert self.m_left_corner_per_image, 'left image keypoints not found yet'
        assert self.m_right_corner_per_image, 'right image keypoints not found yet'
        
        w, h = self.m_pattern_size
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        object_points = np.column_stack((xx.flat, yy.flat, np.zeros(xx.size))).astype(np.float32)
        
        if verbose:
            print('perform stereo calibration ...')
            
        flags = cv2.CALIB_FIX_INTRINSIC
        left_valid = [x is not None for x in self.m_left_corner_per_image]
        right_valid = [x is not None for x in self.m_right_corner_per_image]
        pts_left = [x for i,x in enumerate(self.m_left_corner_per_image) if left_valid[i] and right_valid[i]]
        pts_right = [x for i,x in enumerate(self.m_right_corner_per_image) if left_valid[i] and right_valid[i]]
        
        left_camera_matrix = self.m_left_camera.intrinsic_3x3
        left_distortion = self.m_left_camera.distortion
        right_camera_matrix = self.m_right_camera.intrinsic_3x3
        right_distortion = self.m_right_camera.distortion
        
        res = cv2.stereoCalibrateExtended(
            [object_points] * len(pts_left), pts_left, pts_right,
            np.ascontiguousarray(left_camera_matrix.T), left_distortion,
            np.ascontiguousarray(right_camera_matrix.T), right_distortion,
            (0,0), None, None, None, None, None, flags
        )
        
        self.m_reprojection_error = res[0]
        self.m_reprojection_error_per_view = res[-1]
        R,t = res[5:7]
        
        # the coordinate frame of left camera relative to the right camera
        extmat = np.eye(4)
        extmat[:3,:3] = R.T
        extmat[-1,:3] = t.flatten()
        self.m_right_camera.set_extrinsic(extmat)
        self.m_left_camera.set_extrinsic(np.eye(4)) #clear left camera extrinsic
        
        # fundamental and essential matrices
        self.m_essential_matrix = res[7]
        self.m_fundamental_matrix = res[8]
        
        if verbose:
            print('calibration done')
            
    def set_points_correspondence(self, ptslist_left, ptslist_right):
        ''' directly set left/right corresponding points
        
        parameters
        ----------------
        ptslist_left
            list of Nx2 points of left camera
        ptslist_right
            list of Nx2 points of right camera, ptslist_left[i][k] matches to ptslist_right[i][k]
        '''
        assert len(ptslist_left) == len(ptslist_right), 'number of point sets should match'
        self.m_left_corner_per_image = ptslist_left
        self.m_right_corner_per_image = ptslist_right
        
        
    def find_circle_centers(self, is_symmetric = True, is_left = True,
                            min_area = None, max_area = None, 
                            verbose = False, show = False,
                            func_preproc = None,
                            func_post_transform = None):
        ''' find centers of circle pattern of left and right images
        
        parameters
        ---------------
        is_left
            are we computing for left or right camera?
        is_symmetric
            is the circular pattern symmetric
        min_area, max_area
            min and max area of the circles
        verbose
            print info during seeking
        show
            should we show the calibration grid in the progress?
        func_preproc
            image preprocessing function, in the form of newimg = func(img)
        func_post_transform
            a function to transform the image and keypoints after they are detected,
            in the form of (img_new, centers_new)=func(img, centers)
            
        return
        ---------
        centers
            list of circle centers
        '''
        from . import inout
        if is_left:
            image_files = self.m_left_image_filenames
            camera = self.m_left_camera
        else:
            image_files = self.m_right_image_filenames
            camera = self.m_right_camera
            
        output = [None] * len(image_files)
            
        for i, imgfile in enumerate(image_files):
            if verbose:
                print('[{0}]:{1}'.format(i, imgfile))
            img : np.ndarray = inout.imread(imgfile)
            
            # preprocessing
            if func_preproc:
                img = func_preproc(img)
            
            corners = find_circle_centers(
                img, self.m_pattern_size, 
                min_area = min_area, max_area = max_area)
            
            if corners is not None:
                if func_post_transform:
                    img, corners = func_post_transform(img, corners)
                    
                imgsize_hw = (camera.image_height, camera.image_width)
                assert np.allclose(img.shape[:2], imgsize_hw), 'input image size is not the same as single-camera calibration image size'
                
                if show:
                    import igpy.myplot.implot as cvp
                    canvas = cvp.draw_calibration_grid(img, corners, self.m_pattern_size)
                    cvp.imshow(canvas)
            
            output[i] = corners
            
        # set
        if is_left:
            self.m_left_corner_per_image = output
        else:
            self.m_right_corner_per_image = output
        return output
        
    def find_chessboard_corners(self, verbose = False, is_left = True, func_preproc = None, show= False):
        ''' find chessboard corners in the images
        
        parameters
        ------------------
        verbose
            should the function print progress
        is_left
            are we finding corners for left or right camera
        func_preproc
            image preprocessing function, in the form of newimg = func(img)
        
        return
        --------------
        corners
            chessboard corners found in left or right camera, for each image
        '''
        from . import inout
        
        if is_left:
            image_files = self.m_left_image_filenames
            camera = self.m_left_camera
        else:
            image_files = self.m_right_image_filenames
            camera = self.m_right_camera        
            
        output = [None] * len(image_files)
            
        for i, imgfile in enumerate(image_files):
            if verbose:
                print('[{0}]:{1}'.format(i, imgfile))
            img = inout.imread(imgfile)
            
            if func_preproc:
                img = func_preproc(img)
            
            imgsize_hw = (camera.image_height, camera.image_width)
            assert np.allclose(img.shape[:2], imgsize_hw), 'input image size is not the same as single-camera calibration image size'
            
            corners = find_chessboard_corner(
                img, self.m_pattern_size, 
                use_subpix_refine=self.m_options.use_subpixel_refinement,
                refine_window_size=self.m_options.subpixel_refinement_window)
            
            if show:
                import igpy.myplot.implot as cvp
                canvas = cvp.draw_calibration_grid(img, corners, self.m_pattern_size)
                cvp.imshow(canvas)
            output[i] = corners
        
        if is_left:
            self.m_left_corner_per_image = output
        else:
            self.m_right_corner_per_image = output
        
        return output

class SingleCameraCalibrator:
    ''' calibrate a single camera given a series of chessboard image files
    '''
    
    class Options:
        def __init__(self):
            self.use_subpixel_refinement = True
            self.subpixel_refinement_window = (30,30)
            
            # calibrate the distortion coefficients
            self.use_distortion = True
            
            # force the x and y focal length to be equal
            self.use_equal_focal_length = False
        
    def __init__(self):
        self.m_image_filenames:list = None
        self.m_corner_per_image:list = None
        self.m_reprojection_error = 0
         
        self.m_options = SingleCameraCalibrator.Options()
        self.m_pattern_size = None
        self.m_camera = CameraModel()
        
    def init_with_image_files(self, filenamelist, pattern_size):
        ''' set the paths of the chessboard images
        
        parameters
        ----------------
        filenamelist
            the file names of the images
        pattern_size
            (width,height) of the chessboard pattern
        '''
        self.m_image_filenames = filenamelist
        self.m_pattern_size = pattern_size
        self.m_corner_per_image = None
        
    def get_options(self):
        return self.m_options
    
    def find_circle_centers_symmetric(self, verbose = False, 
                                      min_area = None, max_area = None,
                                      func_image_preproc = None,
                                      show = False,
                                      func_post_transform = None):
        ''' find circle centers in each image, the circles are arranged in symmetric pattern
        
        return
        -----------
        centers
            centers[i] is Nx2 matrix, which is xy coordinates of the centers in image[i]. If no corner is found, centers[i]=None
        min_area, max_area
            min and max area of each individual circle
        func_image_preproc
            preprocessing function over the image, has the form func(img)->new image.
        func_post_transform
            a function to transform the image and the found centers after they are detected, 
            in the form of (img_new, centers_new) = func(img, centers)
        '''         
        cornerlist = []
        imgsize_hw = None
        for i, fn in enumerate(self.m_image_filenames):
            if verbose:
                print('processing image {0} : {1}'.format(i, fn))
            
            img = cv2.imread(fn, cv2.IMREAD_ANYCOLOR)
            if func_image_preproc is not None:
                img = func_image_preproc(img)
                
            centers = find_circle_centers(
                img, self.m_pattern_size, min_area = min_area, max_area = max_area
            )
            
            if centers is None:
                cornerlist.append(None)
                continue
            
            if func_post_transform:
                img, centers = func_post_transform(img, centers)            
            
            if show:
                import igpy.myplot.implot as cvp
                canvas = cvp.draw_calibration_grid(img, centers, self.m_pattern_size)
                cvp.imshow(canvas)
             
            assert centers is not None, 'failed to detect circle grid in {0}'.format(fn)
            cornerlist.append(centers)
            imgsize_hw = img.shape[:2]
            
        self.m_corner_per_image = list(cornerlist)
        self.m_camera.set_image_size(imgsize_hw[1], imgsize_hw[0])
        
        return cornerlist                            
        
    
    def find_chessboard_corners(self, verbose = False):
        ''' find chessboard corners in each image.
        
        return
        -----------
        corners
            corners[i] is Nx2 matrix, which is xy coordinates of the corners in image[i]. If no corner is found, corners[i]=None
        '''
        cornerlist = []
        imgsize_hw = None
        for i, fn in enumerate(self.m_image_filenames):
            if verbose:
                print('processing image {0} : {1}'.format(i, fn))
            
            img = cv2.imread(fn, cv2.IMREAD_ANYCOLOR)
            corners = find_chessboard_corner(
                img, self.m_pattern_size, 
                use_subpix_refine=self.m_options.use_subpixel_refinement,
                refine_window_size=self.m_options.subpixel_refinement_window)
            
            cornerlist.append(corners)
            imgsize_hw = img.shape[:2]
            
        self.m_corner_per_image = list(cornerlist)
        self.m_camera.set_image_size(imgsize_hw[1], imgsize_hw[0])
        
        return cornerlist                    
    
    def calibrate(self, verbose = False):
        ''' run calibration. 
        If the chessboard corners are not found yet, it will automatically find them first.
        '''
        assert self.m_corner_per_image is not None, 'keypoints are missing, please run keypoint detection first'
        
        corners = [x for x in self.m_corner_per_image if x is not None]
            
        w, h = self.m_pattern_size
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        object_points = np.column_stack((xx.flat, yy.flat, np.zeros(xx.size))).astype(np.float32)
        flags = 0
        
        if not self.m_options.use_distortion:
            flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_FIX_S1_S2_S3_S4 + cv2.CALIB_FIX_TANGENT_DIST
            
        if self.m_options.use_equal_focal_length:
            flags += cv2.CALIB_FIX_ASPECT_RATIO
            
        imgsize_wh = (self.m_camera.image_width, self.m_camera.image_height)
        init_camera_matrix = np.eye(3, dtype=np.float32)
        init_camera_matrix[0,-1] = imgsize_wh[0]/2
        init_camera_matrix[1,-1] = imgsize_wh[1]/2
        
        res = cv2.calibrateCamera(
            [object_points] * len(corners),
            corners, imgsize_wh,
            init_camera_matrix, None, None, None, flags
        )
        
        self.m_reprojection_error = res[0]
        self.m_camera.set_intrinsic(res[1].T)
        self.m_camera.set_distortion(res[2])
        
def find_circle_centers(img, pattern_size_wh, is_symmetric = True, min_area = None, max_area = None, show=False):
    ''' find centers of the circles in circle pattern grid
    
    parameters
    ----------------
    img
        the image, where dark circles are searched over bright background
    pattern_size_wh
        (width, height) of the circle pattern
    min_area, max_area
        minimum and maximum area of the circle, in pixels
    show
        if True, show the detected pattern
        
    return
    ------------
    centers
        Nx2 list of circle centers, in xy coordinates. Or None if no circle is detected.
    '''    
    
    p = cv2.SimpleBlobDetector_Params()
    if min_area is not None:
        p.minArea = min_area
    if max_area is not None:
        p.maxArea = max_area
        
    det = cv2.SimpleBlobDetector_create(p)
    ptn = tuple(pattern_size_wh)
    
    if is_symmetric:
        sym_tag = cv2.CALIB_CB_SYMMETRIC_GRID
    else:
        sym_tag = cv2.CALIB_CB_ASYMMETRIC_GRID
    
    res = cv2.findCirclesGrid(img, ptn, sym_tag + cv2.CALIB_CB_CLUSTERING, blobDetector = det)
    
    if show and res[0]:
        import igpy.myplot.implot as cvp
        canvas = cvp.draw_calibration_grid(img, res[1], pattern_size_wh)
        cvp.imshow(canvas)
            
    if res[0]:
        return res[1]
    else:
        return None
        

def find_chessboard_corner(img, pattern_size_wh, use_subpix_refine = True, refine_window_size=(11,11)):
    ''' find chessboard corners in an image
    
    parameters
    -------------------
    img
        gray or RGB image in uint8
    pattern_size_wh
        (width, height) of the chessboard, counting the inner corners
    use_subpix_refine
        whether to use subpixel refinement
    refine_window_size
        in subpixel refinement, the window size to look at
        
    return
    ----------------
    corners
        Nx2 matrix, each row is the xy coordinate of a conrner. If not found, return None
    '''
    pattern_size_wh = tuple(pattern_size_wh)
    refine_window_size = tuple(refine_window_size)
    
    found, corners = cv2.findChessboardCorners(img, pattern_size_wh)
    if not found:
        return None
    
    if use_subpix_refine:
        if np.ndim(img) == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[-1] ==4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                
        winsize = (refine_window_size[0]//2, refine_window_size[1]//2)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
        corners = cv2.cornerSubPix(img, corners, winsize, (-1,1), criteria)
        
    output = corners.reshape((-1, 2))
    return output

def pad_image_for_integral_scale(img : np.ndarray, target_size_hw):
    ''' pad the image, so that it can be scaled to target size by an integer scale factor.
    
    parameters
    -------------
    img
        the image to be padded and scaled
    target_size_hw
        (height, width) of the target image size
        
    return
    ------------
    imgnew
        the padded image
    padinfo
        (top, bottom, width, height) padding size
    '''
    target_height, target_width = target_size_hw
    src_height, src_width = img.shape[:2]
    
    h_scale = target_height / src_height
    w_scale = target_width / src_width
    
    if h_scale == w_scale:
        padinfo = np.zeros(4, dtype=int)
        return img, padinfo
    elif w_scale < h_scale:
        # respect w_scale, pad height
        scale_int = np.floor(w_scale)
    else:
        scale_int = np.floor(h_scale)
        
    # compute how to pad
    w_pad = target_width - scale_int * src_width
    w_pad_left = int(w_pad/2)
    w_pad_right = w_pad - w_pad_left
    
    h_pad = target_height - scale_int * src_height
    h_pad_top = int(h_pad/2)
    h_pad_bottom = h_pad - h_pad_top
    
    # pad it
    #imgsize_new = (h_pad + src_height, w_pad + src_width)
    imgpad = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, None, cv2.BORDER_CONSTANT, 0)
    
    return imgpad, (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)    
    
def intrinsic_matrix_by_fovx(fovx_deg, image_width, image_height):
    ''' compute intrinsic matrix given horizontal fov and image size
    
    return
    ------------
    intrinsic_matrix
        3x3 intrinsic matrix in right-mul format (last row is image center)
    '''
    intmat = np.eye(3)
    f = 0.5 * image_width/np.tan(np.deg2rad(fovx_deg/2))
    cx = image_width/2
    cy = image_height/2
    intmat = np.array([
        [f, 0, 0],
        [0, f, 0],
        [cx, cy, 1]
    ], dtype=float)
    
    return intmat
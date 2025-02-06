import numpy as np
from ..geometry import geometry as geom

class GyroAccelSensor:
    ''' IMU sensor with gyroscope and accelerometer
    '''
    def __init__(self, imu_buffer_size = 1000):
        self.m_buf_index = -1    # latest sample is at this buffer location
        
        # gyroscope data
        self.m_gyro = np.zeros((imu_buffer_size,3))
        
        # accelerometer data
        self.m_accel = np.zeros((imu_buffer_size,3))
        
        # timestamp in seconds for each sample
        self.m_time = np.zeros((imu_buffer_size,3))
        
        # object 4x4 transformations for each sampel
        self.m_end_transforms : list = [None] * imu_buffer_size
        
        # accumulated velocity in world space
        self.m_velocity_world = np.zeros((imu_buffer_size,3))
        
        # accumulated angle-axis orientation
        self.m_gyro_cum = np.zeros((imu_buffer_size,3))
        
        # mask of externally fixed transformations
        self.m_end_transform_fixmask = np.zeros(imu_buffer_size, dtype=bool)
        
        # the accelerometer reading when the object is idle
        self.m_accel_idle : np.ndarray = None
        
        # idle gyroscope reading, use for eliminating noise
        self.m_gyro_idle : np.ndarray = np.zeros(3)
        
    def reset(self, idle_accel : np.ndarray = None, idle_gyro : np.ndarray = None):
        ''' clear tracking state
        '''
        if idle_accel is not None:
            self.m_accel_idle = idle_accel
            
        if idle_gyro is not None:
            self.m_gyro_idle = idle_gyro
            
        self.m_gyro.flat[:] = 0
        self.m_accel.flat[:] = 0
        self.m_time.flat[:] = 0
        self.m_end_transforms = [None] * self.buffer_size
        self.m_velocity_world.flat[:] = 0
        self.m_gyro_cum.flat[:] = 0
        self.m_end_transform_fixmask.flat[:] = False
        self.m_buf_index = -1
        
    @property
    def buffer_size(self):
        return len(self.m_time)
        
    def init(self, idle_accel : np.ndarray, idle_gyro : np.ndarray = None):
        assert idle_accel.size() == 3, 'accelerometer reading must be 3 dimensional'
        self.reset(idle_accel, idle_gyro)
        
    def set_idle_accel(self, accel : np.ndarray):
        ''' set the accelerometer reading when the object is idle
        '''
        assert accel.size == 3, 'accelerometer reading must be 3 dimensional'
        self.m_accel_idle = accel
        
    def set_idle_gyro(self, gyro:np.ndarray):
        ''' set the idle gyro scope noise reading
        '''
        assert gyro.size == 3, 'gyroscope reading must be 3 dimensional'
        self.m_gyro_idle = gyro
        
    def push_sample(self, gyro:np.ndarray, accel:np.ndarray, timestamp:float):
        ''' add a new sample with gyroscope and accelerometer reading, as well as the timestamp in ms.
        
        parameters
        ---------------
        gyro
            1x3 vector, the gyroscope reading in rad/s
        accel
            1x3 vector, the accelerometer reading in m/s**2
        timestamp
            timestamp reading in seconds
            
        '''
        idx_write = (self.m_buf_index + 1) % self.buffer_size
        self.m_gyro[idx_write] = gyro - self.m_gyro_idle
        self.m_accel[idx_write] = accel
        self.m_time[idx_write] = timestamp
        self.m_buf_index = idx_write
        
        self._update_transform(idx_write)
        return idx_write
        
    def get_transform(self, index : int = None):
        if index is None:
            index = self.m_buf_index
        return self.m_end_transforms[index]
        
    def _update_transform(self, idx):
        ''' update the transformation at a specific sample
        '''
        idx_prev = (idx - 1 + self.buffer_size) % self.buffer_size
        
        # previous transformation and imu readings
        tmat_prev : np.ndarray = self.m_end_transforms[idx_prev]
        if tmat_prev is None or self.m_accel_idle is None:
            self.m_end_transforms[idx] = np.eye(4)
            self.m_velocity_world[idx] = [0,0,0]
            self.m_gyro_cum[idx] = [0,0,0]
            self.m_end_transform_fixmask[idx] = True
            return
        
        accel_prev : np.ndarray = self.m_accel[idx_prev]
        gyro_prev : np.ndarray = self.m_gyro[idx_prev]
        dt_sec = self.m_time[idx] - self.m_time[idx_prev]
        
        # gravity in previous coordinate frame
        accel_idle_prev = self.m_accel_idle.dot(tmat_prev[:3,:3].T)
        
        # previous coordinate velocity and acceleration
        accel_coord_prev = accel_prev - accel_idle_prev
        accel_world_prev = accel_coord_prev.dot(tmat_prev[:3,:3]) # in world coordinate
        
        # previous velocity in world coordinate
        vel_world_prev = self.m_velocity_world[idx_prev]
        
        # new position in world
        pos_new = vel_world_prev * dt_sec + 0.5 * accel_world_prev * dt_sec**2 + tmat_prev[-1,:3]
        #pos_new = vel_world_prev * dt_sec + tmat_prev[-1,:3]
        
        # compute orientation
        gyro_cum_now = self.m_gyro_cum[idx_prev] + gyro_prev * dt_sec        
        
        tmat_new = np.eye(4)
        if False:
            # use accumulated gyroscope data
            rot_angle = np.linalg.norm(gyro_cum_now)
            rot_axis = gyro_cum_now / rot_angle
            rotmat = geom.rotation_matrix_by_angle_axis(rot_angle, rot_axis)
            tmat_new[:3,:3] = rotmat
        else:
            # use accumulated rotation matrix
            rot_angle_rate = np.linalg.norm(gyro_prev)
            rot_axis = gyro_prev/rot_angle_rate
            rot_angle = rot_angle_rate * dt_sec
            d_rotmat = geom.rotation_matrix_by_angle_axis(rot_angle, rot_axis)
            tmat_new[:3,:3] = d_rotmat.dot(tmat_prev[:3,:3]) # R_prev * inv(R_prev) * R_new * R_prev
            
        tmat_new[-1,:3] = pos_new
        self.m_end_transforms[idx] = tmat_new
        
        # update velocity
        vel_world_new = vel_world_prev + accel_world_prev * dt_sec
        self.m_velocity_world[idx] = vel_world_new
        
        # update gyroscope accumulation
        self.m_gyro_cum[idx] = gyro_cum_now
        
    def fix_transform(self, transmat : np.ndarray):
        ''' fix the latest transformation
        '''
        idx = self.m_buf_index
        assert idx >= 0
        
        # fix accumulated velocity
        
        # find previously fixed transformation
        idx_prev = None
        for i in range(1, self.buffer_size):
            idx_prev = (idx - i) % self.buffer_size
            if self.m_end_transform_fixmask[idx_prev]:
                break
        
        # have previous transformation, update velocity
        if idx_prev is not None:
            prev_tmat = self.m_end_transforms[idx_prev]
            dpos = transmat[-1,:-1] - prev_tmat[-1,:-1]
            dt = self.m_time[idx] - self.m_time[idx_prev]
            self.m_velocity_world[idx] = dpos / dt
        
        # replace transformation
        self.m_end_transforms[idx] = np.array(transmat)
        
        
        
        
        
        
        
        
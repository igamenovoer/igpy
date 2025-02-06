# functions to deal with lie algebra
import numpy as np

class SO3:
    ''' lie group SO3 object. We follow the convention in most textbooks,
    using LEFT-multiply matrix to transform vectors.
    '''
    def __init__(self) -> None:
        self.m_SO3_matrix : np.ndarray = None   # SO3 in matrix form
        self.m_so3_vec : np.ndarray = None  # lie algebra so3 in vector form
        
    def init_with_rotation_matrix(self, rotmat_left : np.ndarray):
        ''' initialize with left-multiply rotation matrix
        '''
        # check
        assert np.allclose(rotmat_left.shape, (3,3)), 'R should be 3x3 matrix'
        assert np.allclose(np.linalg.det(rotmat_left), 1.0), 'det(R) should be 1'
        self.m_SO3_matrix = rotmat_left
        self.m_so3_vec = __class__.rotation_matrix_to_so3vec(rotmat_left)
        
    @staticmethod
    def rotation_matrix_to_so3vec(rotmat_left : np.ndarray) -> np.ndarray:
        so3mat = __class__.rotation_matrix_to_so3mat(rotmat_left)
        so3vec = __class__.so3_mat_to_vec(so3mat)
        return so3vec
    
    @staticmethod
    def rotation_matrix_to_so3mat(rotmat_left : np.ndarray) -> np.ndarray:
        R = rotmat_left
        cos_theta = np.clip((R.trace()-1)/2,-1,1)
        angle = np.arccos(cos_theta)
        u_cross = angle/(2*np.sin(angle)) * (R-R.T) # so3 in matrix form
        return u_cross
    
    @staticmethod
    def so3_vec_to_mat(so3vec : np.ndarray) -> np.ndarray:
        x,y,z = so3vec
        so3mat = np.array([
            [0,z,-y],
            [z,0,x],
            [-y,x,0]
        ])
        return so3mat
    
    @staticmethod
    def so3_mat_to_vec(so3mat : np.ndarray) -> np.ndarray:
        x,y,z = so3mat[-1,1],so3mat[-1,0],so3mat[1,0]
        return np.array([x,y,z])
    
    @staticmethod
    def so3mat_to_rotation_matrix(so3mat: np.ndarray) -> np.ndarray:
        I = np.eye(3, dtype=float)
        so3vec = __class__.so3_mat_to_vec(so3mat)
        angle = np.linalg.norm(so3vec)
        u_cross = so3mat / angle
        R = I+np.sin(angle) * u_cross + (1-np.cos(angle))*u_cross**2
        return R
    
    @staticmethod
    def so3vec_to_rotation_matrix(so3vec: np.ndarray) -> np.ndarray:
        so3mat = __class__.so3_vec_to_mat(so3vec)
        R = __class__.so3mat_to_rotation_matrix(so3mat)
        return R
        
        
        
        
    
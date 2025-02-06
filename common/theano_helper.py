import numpy as np
import theano
import theano.tensor as tht

def rotation_matrix_by_X(angle_rad):
    ''' rotation matrix that rotates an object around X by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a theano scalar
    '''
    x = angle_rad  
    cos = tht.cos(x)
    sin = tht.sin(x)
    rs = [1,0,0,
          0,cos,-sin,
          0,sin,cos]
    rotmat = tht.reshape(rs, (3,3)).T
    return rotmat

def rotation_matrix_by_Y(angle_rad):
    ''' rotation matrix that rotates an object around Y by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a theano scalar
    '''
    x = angle_rad
    cos = tht.cos(x)
    sin = tht.sin(x)
    rs = [cos,0,sin,
          0,1,0,
          -sin,0,cos]
    rotmat = tht.reshape(rs, (3,3)).T
    return rotmat

def rotation_matrix_by_Z(angle_rad):
    ''' rotation matrix that rotates an object around Z by angle_rad.
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a theano scalar
    '''
    x = angle_rad
    cos = tht.cos(x)
    sin = tht.sin(x)
    rs = [cos,-sin,0,
          sin,cos,0,
          0,0,1]
    rotmat = tht.reshape(rs,(3,3)).T
    return rotmat

def rotation_matrix_by_xyz_vec(xyz):
    ''' create rotation matrix from rotating around x by dx, y by dy and z by z
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    xyz
        tensors of angles, represented in radian
        
    Return
    --------------
    R
        rotation matrix where point.dot(R) gives a rotated point
    '''
    rx = rotation_matrix_by_X(xyz[0])
    ry = rotation_matrix_by_Y(xyz[1])
    rz = rotation_matrix_by_Z(xyz[2])
    return rx.dot(ry).dot(rz)

def make_variables(name2val):
    ''' create theano variables
    
    Parameters
    ----------------
    name2val
        a dict, such that name2val[name]==val iff variable of a given name
        is initialized to be val
        
    Return
    ---------------
    x_dict
        a dict, where x_dict[name] is the variable of name2val[name], and they
        are of the same shape
    x_raw
        all variables concatenated into a vector
    x0
        the initial value of x_raw
    name2idx
        a dict, where name2idx[name]=idx iff the variable of a given name
        is in x_raw[name2idx[name]]
    '''
    import theano
    import theano.tensor as T
    
    if False:
        name2val = {}
    
    # collect current values of all fields
    varnames = list(name2val.keys())
    varvalues = [name2val[x] for x in varnames]
    varsize = []
    varshape = []
    for v in varvalues:
        if np.isscalar(v):
            varsize.append(1)
            varshape.append(())
        else:
            varsize.append(v.size)
            varshape.append(v.shape)
    varsize = np.array(varsize)
    total_size = np.sum(varsize)
    
    x_raw = T.dvector()
    field2idx = {}
    
    # distribute the raw variable to relevant fields
    idxread = 0
    x_dict = {}
    for i in range(len(varnames)):
        name = varnames[i]
        v = varvalues[i]
        size = varsize[i]
        shape = varshape[i]
        if len(shape) == 0: #scalar
            x_sub = x_raw[idxread]
            idxend = idxread + 1
        else:
            n_var = size
            x_sub = x_raw[idxread:idxread + n_var]
            x_sub = T.reshape(x_sub, shape)
            idxend = idxread + n_var
            
        field2idx[name] = np.arange(idxread, idxend)
        idxread = idxend
        x_dict[name] = x_sub
    
    vvlist = [np.atleast_1d(v).flatten() for v in varvalues]
    x0 = np.concatenate(vvlist)
    #done
    return x_dict, x_raw, x0, field2idx
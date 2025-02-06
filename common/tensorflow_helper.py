import numpy as np
import tensorflow as tf
import igpy.common.shortfunc as sf

def list_all_tfrecord_files(dirname, return_full_path = False):
    ''' list all tfrecord files in a directory
    
    parameters
    --------------
    return_full_path
        if True, the returned paths are full paths, otherwise they are
        just filenames
        
    return
    ----------
    pathlist
        a list of paths to all .tfrecord files
    '''
    fnlist = sf.listdir(dirname, r'.*\.tfrecord', prepend_parent_path=return_full_path)
    return fnlist

def sparse_matrix_to_tensor(spmat):
    ''' convert scipy.sparse matrix into tensorflow's tensor
    '''
    import scipy.sparse as sparse
    spmat = spmat.tocoo()
    
    inds = np.column_stack([spmat.row, spmat.col]).astype(int)
    tfval = tf.SparseTensor(inds, spmat.data, spmat.shape)
    return tfval

def initialize_uninitalized_vars(sess):
    ''' initialize global variables that are not already initialized
    '''
    varlist = tf.global_variables()
    is_init_op = [tf.is_variable_initialized(x) for x in varlist]
    mask_init = sess.run(is_init_op)
    var_need_init = [x for i,x in enumerate(varlist) if mask_init[i]==False]
    sess.run(tf.variables_initializer(var_need_init))
    
def combine_vars_by_scatter(varlist, idxlist):
    ''' create a variable x such that varlist[i][k] is at x[idxlist[i][k]], 
    that is, idxlist[i][k] is the index position of varlist[i][k] in x
    
    Parameters
    ------------------------
    varlist
        tensorflow variables, must be 2d and the same second dimension
    idxlist
        a list of indices. Let n=max(idxlist[:]), then we must have union(idxlist[:]) == range(n+1)
    '''
    idx = np.concatenate(idxlist)
    u, inds = np.unique(idx, return_index=True)
    assert len(u) == len(idx), 'idxlist contains overlapping indices'
    assert np.all(u==np.arange(np.max(u)+1)), 'idxlist has gap, some element in 1:n is missing'
    
    v = tf.concat(varlist, axis=0)
    vnew = tf.gather(v, inds.tolist())
    return vnew

def rotation_matrix_by_X(angle_rad):
    ''' rotation matrix that rotates an object around X by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a tensor
    '''
    x = angle_rad
    if len(x.shape)==0:
        x = [x]    
    rs = [[1],[0],[0],
          [0],tf.cos(x),-tf.sin(x),
          [0],tf.sin(x),tf.cos(x)]
    rotmat = tf.reshape(rs, shape=[3,3])
    return tf.transpose(rotmat) #right-multiplication format

def rotation_matrix_by_Y(angle_rad):
    ''' rotation matrix that rotates an object around Y by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a tensor
    '''
    x = angle_rad
    if len(x.shape)==0:
        x = [x]    
    rs = [tf.cos(x),[0],tf.sin(x),
          [0],[1],[0],
          -tf.sin(x),[0],tf.cos(x)]
    rotmat = tf.reshape(rs, shape=[3,3])
    return tf.transpose(rotmat) #right-multiplication format

def rotation_matrix_by_Z(angle_rad):
    ''' rotation matrix that rotates an object around Z by angle_rad.
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a tensor
    '''
    x = angle_rad
    if len(x.shape)==0:
        x = [x]    
    rs = [tf.cos(x), -tf.sin(x), [0],
          tf.sin(x),tf.cos(x),[0],
          [0],[0],[1]]
    rotmat = tf.reshape(rs, shape=[3,3])    
    return tf.transpose(rotmat) #right-multiplication format

def rotation_matrix_by_xyz(dx,dy,dz):
    ''' create rotation matrix from rotating around x by dx, y by dy and z by z
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    dx,dy,dz
        tensors of angles, represented in radian
        
    Return
    --------------
    R
        rotation matrix where point.dot(R) gives a rotated point
    '''
    rx = rotation_matrix_by_X(dx)
    ry = rotation_matrix_by_Y(dy)
    rz = rotation_matrix_by_Z(dz)
    return tf.matmul(tf.matmul(rx,ry),rz)

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
    xyz = tf.convert_to_tensor(xyz)
    rx = rotation_matrix_by_X(xyz[0])
    ry = rotation_matrix_by_Y(xyz[1])
    rz = rotation_matrix_by_Z(xyz[2])
    return tf.matmul(tf.matmul(rx,ry),rz)

def rotation_matrix_by_Rodrigues(rotvec):
    ''' create 3x3 rotation matrix from angle-axis vector in Rodrigues format.
    Note that the rotation matrix is in right-multiply format
    
    Parameters
    ------------------
    rotvec
        The rotation vector, where |rotvec| is the rotation angle and rotvec/|rotvec|
        is the rotation axis.
    '''
    rot_angle = tf.norm(rotvec)
    dtype = rot_angle.dtype
    rot_axis = rotvec / rot_angle
    x = rot_axis[0]
    y = rot_axis[1]
    z = rot_axis[2]
    r = [0.,-z,y,
         z,0.,-x,
         -y,x,0.]
    K = tf.reshape(r,shape=(3,3))
    R = tf.eye(3, dtype=dtype) + tf.sin(rot_angle)*K + (1-tf.cos(rot_angle))*tf.matmul(K,K)
    return tf.transpose(R) #convert to right-multiply format

def rotation_RT_2d(angle, pivot = None):
    ''' create 2d rotation matrix and translation vector.
    
    parameters
    -----------------
    angle
        in rad, the angle of rotation
    pivot
        2d point, the pivot of the rotation
        
    return
    -------------
    R
        the 2x2 rotation matrix, in right-multiplication format
    t
        the translation vector after rotation
    '''
    R = tf.stack([
        [tf.cos(angle), -tf.sin(angle)],
        [tf.sin(angle), tf.cos(angle)]
    ])
    R = tf.transpose(R)
    dtype = R.dtype
    if pivot is None:
        t = tf.constant(np.zeros(2), dtype=dtype)
    else:
        pivot = tf.cast(tf.reshape(pivot, (1,2)), dtype=dtype)
        t = pivot - tf.matmul(pivot,R)
        
    return R, t
    
    
def make_scipy_optimization_func(sess, loss, varlist, feed_dict = None):
    ''' create scipy optimization functions which minimize tensorflow 
    loss function over a list of variables.
    
    Parameters
    --------------------------
    sess
        tensorflow session used to evaluate the loss function
    loss
        tensorflow loss function, a tensor
    varlist
        list of tensorflow variables to be optimized
    feed_dict
        feed into placeholders
    '''
    
    import tensorflow as tf
    
    def objective_funcion(x):
        # assign x into the variables and evaluate loss
        # x is broken down into pieces and assigned to each variable
        k = 0
        assign_ops = []
        for i in range(len(varlist)):
            tf_var = varlist[i]
            varsize= np.array(tf_var.shape.as_list())
            num_elem = varsize.prod()
            val = x[k:num_elem]
            k += num_elem
            op = tf.assign(tf_var, val.reshape(varsize))
            assign_ops.append(op)
        sess.run(assign_ops)
        val = sess.run(loss)
        return val
    
    def jacobian_function(x):
        # jacobian function 
        tf.gradients
    
def convert_sparse_matrix_to_sparse_tensor(X):
    '''convert a scipy's sparse matrix into tensorflow SparseTensor
    '''
    import tensorflow as tf
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
            
def to_bytes_feature(value):
    ''' convert byte array to tensorflow bytes feature
    '''
    if type(value) is not list:
        value = [value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))

def to_bytes_feature_from_ndarray(value):
    ''' convert ndarray to tensorflow bytes feature '''
    return to_bytes_feature(value.tostring())

def to_float_feature(value):
    value = np.atleast_2d(value).flatten().tolist()
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def to_int64_feature(value):
    value = np.atleast_2d(value).flatten().tolist()    
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def get_tfrecord_data(fn_record, n_example = None):
    ''' read some examples from a tfrecord file
    
    parameters
    ----------------
    fn_record
        the tfrecord file
    n_example
        the number of examples to read.
        by default, only reads one example
        
    return
    ------------
    list_of_dict
        a list of dictionary, each dict contains the data of a single example
    '''
    if n_example is None:
        n_example = 1
        
    _, feature_format = get_tfrecord_info(fn_record)
    rec_iter = tf.python_io.tf_record_iterator(fn_record)
    n = 0
    output = []
    
    ss = None
    if not tf.executing_eagerly():
        ss = tf.Session()
        
    for rec_string in rec_iter:
        exp = tf.train.Example.FromString(rec_string)
        data = tf.parse_single_example(rec_string, feature_format)
        for key in data.keys():
            if not tf.executing_eagerly():
                data[key] = ss.run(data[key])
            
            if type(feature_format[key]) is tf.VarLenFeature:
                data[key] = data[key].values
                
        # expand data in eager mode
        if tf.executing_eagerly():
            for key in data.keys():
                data[key] = data[key].numpy()
            
        output.append(data)
        n += 1
        if n >= n_example:
            break
        
    if ss is not None:
        ss.close()
        
    return output

def get_tfrecord_info(fn_record):
    ''' list the keys and data types in tf record. It assumes the Examples contained in tfrecord all have the same keys and data types.
    
    parameters
    --------------------
    fn_record
        the tf record filename
        
    return
    -----------------
    dict
        a dict containing the keys and data types of each example
    feature_format
        a dict containing the feature format that can be passed into 
        tf.parse_single_example() to get feature data
    '''
    rec_iter = tf.python_io.tf_record_iterator(fn_record)
    for rec_string in rec_iter:
        exp = tf.train.Example.FromString(rec_string)
        break
    
    f = exp.features.feature
    keylist = list(f.keys())
    output = {}
    feature_format = {}
    
    for key in keylist:
        content = f[key]
        datatype = None
        feature_desc = None
        if content.int64_list.ByteSize() > 0:
            datatype = tf.int64
            feature_desc = tf.VarLenFeature(tf.int64)
        elif content.float_list.ByteSize() > 0:
            datatype = tf.float32
            feature_desc = tf.VarLenFeature(tf.float32)
        elif content.bytes_list.ByteSize() > 0:
            datatype = tf.string
            feature_desc = tf.FixedLenFeature([], tf.string)
            
        output[key] = datatype
        feature_format[key] = feature_desc
    return output, feature_format

def get_placeholders(tfgraph, return_dict = False):
    ''' retrieve the tensors representing placeholders in a tensorflow graph
    
    parameters
    -----------------
    tfgraph
        a tensorflow graph object
    return_dict
        if True, return the placeholders as a dict, otherwise return them as a list
    
    return
    ----------------
    list_or_dict
        If return_dict is True, return a dict where the keys are operations names and the values are placeholder tensors. Otherwise, return a list of placeholder tensors
    '''
    oplist = [x for x in tfgraph.get_operations() if x.type=='Placeholder']
    if return_dict:
        output = {x.name: x.values()[0] for x in oplist}
    else:
        output = [x.values()[0] for x in oplist]
    return output

def make_tfrecord_example(features):
    ''' create an Example to be saved into a tfrecord file
    
    parameters
    ---------------
    features
        a dict, where the values should be created by to_int64_feature, to_bytes_feature or 
        to_float_feature
        
    return
    ------------
    example
        a tfrecord example
    '''
    f = tf.train.Features(feature = features)
    exp = tf.train.Example(features = f)
    return exp

def write_tfrecord_feature(tf_writer : tf.python_io.TFRecordWriter, feature_dict : dict):
    ''' create an Example to be saved into a tfrecord file
    
    parameters
    ---------------
    tf_writer
        an opened tf.python_io.TFRecordWriter for writing
    feature_dict
        a dict, where the values should be created by to_int64_feature, to_bytes_feature or 
        to_float_feature    
    '''
    f = tf.train.Features(feature = feature_dict)
    example = tf.train.Example(features = f)
    tf_writer.write(example.SerializeToString())

def scale_images(img, scale):
    ''' scale an image or a batch or images
    
    parameters
    -------------------
    img
        3d tensor [height, width, channels] or 4d tensor [batch, height, width, channels]
    scale
        scale factor
        
    return
    --------------
    img
        scaled image tensor
    '''
    if len(img.shape) < 4:
        shape = tf.shape(img)[:2]
    else:
        shape = tf.shape(img)[1:3]
    sizehw = tf.to_int32(tf.to_float(shape) * scale)
    return tf.image.resize_images(img, sizehw)

def get_size_hw(img, return_const = False):
    ''' get the height and width of the tensor.
    
    parameters
    --------------
    img
        the tensor in [batch, height ,width, channel] or [height, width, channel] for [height, width] format
    return_const
        if True, the size will be returned as numpy constants, not tf tensors.
        This is only possible when img has a fixed size
        
    return
    -------------
    sizehw
        return the size in (height, width)
    '''
    if return_const:
        s = img.shape
        ndim = len(s)
    else:
        s = tf.shape(img)
        ndim = int(s.shape[0])
        
    if ndim == 4:
        imgsize = s[1:3]
    elif ndim == 3:
        imgsize = s[:2]
    elif ndim == 2:
        imgsize = s
        
    if return_const:
        imgsize = np.array([int(x) for x in imgsize])
    return imgsize

class CheckpointSaver:
    def __init__(self):
        # the tensorflow session
        self.m_tf_session = None
        
        # output directory
        self.m_output_dir = None
        
        # number of seconds per save
        self.m_num_secs_per_save = 60 * 10
        
        # keep previous n save
        self.m_num_save_keep = 2
        
        # ============= internals ============
        self.m_prev_time = None
        self.m_ith_slot = 0
        self.m_tf_saver = tf.train.Saver()
        
    def reset(self):
        self.m_prev_time = None
        self.m_ith_slot = 0
        self.m_tf_saver = tf.train.Saver()
        self.m_tf_session = None
        
    def set_tf_session(self, ss):
        self.m_tf_session = ss
        
    def set_output_dir(self, output_dir):
        self.m_output_dir = output_dir
        
        import igpy.common.shortfunc as sf
        sf.mkdir(output_dir)
        
    def set_interval_per_save(self, n_secs):
        self.m_num_secs_per_save = n_secs
        
    def set_num_save_to_keep(self, n):
        self.m_num_save_keep = n
        
    @staticmethod
    def init_with_session(tf_session, output_dir):
        obj = CheckpointSaver()
        obj.m_tf_session = tf_session
        obj.set_output_dir(output_dir)
        return obj
    
    def save_now(self):
        ss = self.m_tf_session
        fn_output = '{0}/model_{1}'.format(self.m_output_dir, self.m_ith_slot)
        self.m_tf_saver.save(ss, fn_output)
        self.m_ith_slot = (self.m_ith_slot + 1) % self.m_num_save_keep        
        
    def check_and_save(self):
        ''' check if it is the time for saving. If yes, the model will be written to disk
        
        return
        --------------
        true_or_false
            if the model is saved, return True, otherwise return False
        '''
        import time
        
        assert self.m_output_dir is not None, 'output dir is not set'
        assert self.m_tf_session is not None, 'tensorflow session is not set'
        if self.m_prev_time is None:
            self.m_prev_time = time.clock()
            
        ss = self.m_tf_session
        model_saved = False
        import time
        dt = time.clock() - self.m_prev_time
        if dt >= self.m_num_secs_per_save:
            fn_output = '{0}/model_{1}'.format(self.m_output_dir, self.m_ith_slot)
            self.m_tf_saver.save(ss, fn_output)
            
            self.m_prev_time = time.clock()
            self.m_ith_slot = (self.m_ith_slot + 1) % self.m_num_save_keep
            
            model_saved = True
        
        return model_saved
            
def save_graph(graph, outdir):
    ''' save a tensorflow graph to a directory
    
    parameters
    --------------
    graph
        the tensorflow graph
    outdir
        the output directory, should be an empty directory
    '''
    wt = tf.summary.FileWriter(outdir, graph=graph)
    wt.close()

def save_variables_as_numpy(outfilename, 
                            sess : tf.Session, 
                            varlist : list = None, 
                            graph : tf.Graph = None):
    ''' save variables in session into a dict of nu
    mpy arrays
    
    parameters
    -------------
    outfilename
        the output filename
    sess
        the tensorflow session
    varlist
        a list of variables to save. If None, save all variables
    graph
        the graph containing the variable definitions. If None, use default graph
    '''
    import dill
    
    if graph is None:
        graph = tf.get_default_graph()
    
    if varlist is None:
        varlist = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        
    od = {}
    for v in varlist:
        v : tf.Variable = v
        od[v.name] = sess.run(v)
        
    with open(outfilename, 'wb+') as fid:
        dill.dump(od, fid)
        
def import_graph_from_checkpoint(fn_ckpt, sess = None):
    ''' import graph and data from a checkpoint file
    
    parameters
    ----------------
    fn_ckpt
        the checkpoint file, without extension. For example, if a checkpoint folder as a file
        named model.meta, model.data-000-of-001, model.index, then you only need to specify
        them as '<parent_dir>/model'
    sess
        the session into which the variables are loaded. If None, data will not be loaded.
    '''
    # the meta graph
    fn_meta = fn_ckpt + '.meta'
    saver = tf.train.import_meta_graph(fn_meta)
    
    # import data
    if sess is not None:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, fn_ckpt)
        
def freeze_graph_simple(graph : tf.Graph, output_node : tf.Tensor):
    with graph.as_default() as g:
        g : tf.Graph = g
        
        with tf.Session().as_default() as ss:
            ss.run(tf.global_variables_initializer())
            ss.run(tf.local_variables_initializer())
            gdef = g.as_graph_def()
            gdef = tf.graph_util.convert_variables_to_constants(ss, gdef, [output_node.name.split(':')[0]])
            
    return gdef
    
def profile_flops(graph_def):
    ''' compute the flops for a graph. The graph must have known shape for all nodes, including placeholder.
    '''    
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def)
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        run_meta = tf.RunMetadata()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        
    return flops.total_float_ops

def profile_count_parameter(graph : tf.Graph):
    ''' compute the number of parameters for a graph. The graph must have known shape for all nodes, including placeholder.
    '''
    with graph.as_default():
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        run_meta = tf.RunMetadata()
        res = tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)
        
    return res.total_parameters    
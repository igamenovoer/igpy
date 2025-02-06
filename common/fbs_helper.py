# flatbuffer helpers
import numpy as np
import flatbuffers as fbs

def get_matrix_class_by_dtype(dtype):
    ''' get the flatbuffer matrix class by dtype
    
    parameters
    --------------
    dtype
        the data type
    
    returns
    --------------
    module
        the module containing the matrix class
    class
        the matrix class
    '''
    import fbdef
    
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return fbdef.Matrix_d, fbdef.Matrix_d.Matrix_d
    elif dtype == np.float32:
        return fbdef.Matrix_f, fbdef.Matrix_f.Matrix_f
    elif dtype == np.int32:
        return fbdef.Matrix_i32, fbdef.Matrix_i32.Matrix_i32
    elif dtype == np.int64:
        return fbdef.Matrix_i64, fbdef.Matrix_i64.Matrix_i64
    elif dtype == np.uint8:
        return fbdef.Matrix_u8, fbdef.Matrix_u8.Matrix_u8
    
class MatrixMaker:
    def __init__(self):
        self.matrix_class = None
        self.matrix_module = None
        
        self.matrix_get_root_as_func = None
        self.matrix_start_func = None
        self.matrix_end_func = None
        self.matrix_add_content_func = None
        self.matrix_add_shape_func = None
        self.content_start_func = None
        self.shape_start_func = None
        self.shape_dtype = None
        
    @staticmethod
    def init_by_dtype(dtype):
        ''' get matrix construction functions based on data type
        ''' 
        obj = MatrixMaker()
        mat_module, mat_class = get_matrix_class_by_dtype(dtype)
        clname = mat_class.__name__
        obj.matrix_class = mat_class
        obj.matrix_module = mat_module
        
        obj.matrix_get_root_as_func = getattr(mat_class, 'GetRootAs' + clname)
        obj.matrix_start_func = getattr(mat_module, clname+'Start')
        obj.matrix_end_func = getattr(mat_module, clname+'End')
        obj.content_start_func = getattr(mat_module, clname + 'StartContentVector')
        obj.shape_start_func = getattr(mat_module, clname + 'StartShapeVector')
        obj.matrix_add_content_func = getattr(mat_module, clname + 'AddContent')
        obj.matrix_add_shape_func = getattr(mat_module, clname + 'AddShape')
        obj.shape_dtype = np.int32
        return obj

def get_prepend_func_by_dtype(dtype, fb_builder : fbs.Builder = None):
    ''' get the builder prepend function for a particular data type
    
    parameters
    ----------------
    dtype
        the data type
    fb_builder
        the flatbuffer builder. If set, the returned the member function of the builder.
        If None, return a global function of the builder, which should be called with the 
        builder object as input.
        
    return
    --------------
    prepend_func
        the prepend function
    '''
    obj : fbs.Builder = fb_builder if fb_builder is not None else fbs.Builder
    dtype = np.dtype(dtype)
    
    if dtype == np.uint8:
        return obj.PrependByte
    elif dtype == np.bool:
        return obj.PrependBool
    elif dtype == np.int16:
        return obj.PrependInt16
    elif dtype == np.int32:
        return obj.PrependInt32
    elif dtype == np.int64:
        return obj.PrependInt64
    elif dtype == np.float32:
        return obj.PrependFloat32
    elif dtype == np.float64:
        return obj.PrependFloat64
    else:
        assert False,'dtype not recognizable'

def write_vector(mat : np.ndarray, fb_builder : fbs.Builder, fb_func_start, 
                 fb_dtype = None, auto_reverse = True):
    ''' write a numpy array into flatbuffer
    
    parameters
    ----------------
    mat
        numpy array containing data
    fb_builder
        the flatbuffer builder
    fb_func_start
        flatbuffer start function for data
    fb_dtype
        data type for flatbuffer content. If None, it will follow mat.dtype
    auto_reverse
        reverse the data before writing into flatbuffer. As flatbuffer always prepend data into
        the stream, the data is reversed when you read the buffer forward. Reversing the data
        before writing will allow you to access the buffer in the same order as the original data.
        
    return
    ----------------
    vec
        the flatbuffer vector
    '''
    if type(mat) != np.ndarray:
        mat = np.array(mat)
        
    if fb_dtype is None:
        fb_dtype = mat.dtype
        
    fb_func_start(fb_builder, mat.size)
    prepend_func = get_prepend_func_by_dtype(fb_dtype, fb_builder)
       
    if auto_reverse:
        for x in mat.flat[::-1]:
            prepend_func(x)            
    else:
        for x in mat.flat:
            prepend_func(x)    
            
    obj = fb_builder.EndVector(mat.size)
    return obj

def write_matrix(mat : np.ndarray, fb_builder : fbs.Builder, 
                 fb_dtype = None, auto_reverse = True):
    ''' write a matrix into flatbuffer
    
    parameters
    -----------------
    mat
        numpy matrix to be written
    fb_builder
        the flatbuffer builder
    fb_dtype
        the data type of the matrix. If not set, use mat.dtype
    auto_reverse
        reverse the data before writing into flatbuffer. As flatbuffer always prepend data into
        the stream, the data is reversed when you read the buffer forward. Reversing the data
        before writing will allow you to access the buffer in the same order as the original data.
        
    return
    -----------
    fb_mat
        flatbuffer matrix
    '''
    if fb_dtype is None:
        fb_dtype = mat.dtype
    else:
        fb_dtype = np.dtype(fb_dtype)
        
    maker = MatrixMaker.init_by_dtype(fb_dtype)
    
    # create content and shape
    vec_content = write_vector(mat, fb_builder, maker.content_start_func, fb_dtype = fb_dtype, auto_reverse=auto_reverse)
    vec_shape = write_vector(mat.shape, fb_builder, maker.shape_start_func, fb_dtype = maker.shape_dtype, auto_reverse=auto_reverse)
    
    # create matrix
    maker.matrix_start_func(fb_builder)
    maker.matrix_add_content_func(fb_builder, vec_content)
    maker.matrix_add_shape_func(fb_builder, vec_shape)
    fb_mat = maker.matrix_end_func(fb_builder)
    
    return fb_mat
    
        

    
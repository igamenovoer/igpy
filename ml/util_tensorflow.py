# tensorflow 2.0 utilities

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.wrap_function import WrappedFunction

# def read_image_into_tensor(fn_image: str, 
#                            normalized_neg1_to_1:bool = True) -> tuple[np.ndarray, tf.Tensor]:
#     from igpy.common.inout 

class ModelLoader_SinglePb_v1:
    ''' load single v1 pb file
    '''
    def __init__(self) -> None:
        self.m_pb_filename : str = None
        self.m_graph_def : tf.compat.v1.GraphDef = None

class ModelLoader_SavedModelDir:
    ''' to load v2 saved model
    '''
    def __init__(self) -> None:
        # folder of the saved model, must have a saved_model.pb inside
        self.m_saved_model_dir : str = None
        self.m_model : WrappedFunction = None
        
    @property
    def model(self) -> WrappedFunction:
        return self.m_model
        
    def init(self, dirname : str):
        self.m_saved_model_dir = dirname
        
    def load_model(self, signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY) -> WrappedFunction:
        model_file = tf.saved_model.load(self.m_saved_model_dir)
        self.m_model = model_file.signatures[signature]
        return self.m_model
    
    def export_to_tensorboard(self, logdir : str, clear_dir : bool = False) -> str:
        ''' export the model for tensorboard visualization
        '''
        assert self.m_model is not None, 'load model first'
        from tensorflow.python.client import session
        
        if clear_dir:
            import shutil
            shutil.rmtree(logdir, ignore_errors=True)
        
        with session.Session(graph = self.m_model.graph) as ss:
            writer = tf.compat.v1.summary.FileWriter(logdir=logdir)
            writer.add_graph(ss.graph)
            writer.flush()
            writer.close()
        
        # generate command    
        import os
        p = os.path.abspath(logdir).replace('\\','/')
        cmdstr = 'tensorboard --logdir {}'.format(p)

        return cmdstr
        
    
        
        
        
        
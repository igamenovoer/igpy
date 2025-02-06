import numpy as np
import tensorflow as tf
import igpy.ml.util_tensorflow as ut_tf
from tensorflow.python.client import session

class SingleImageModel_tfv2:
    ''' tensorflow v2 model that only has 1 input which takes a batch of images
    '''

    def __init__(self) -> None:
        self.m_saved_model : ut_tf.WrappedFunction = None
        self.m_preprocess : tf.keras.Sequential = None
        
    @property
    def input_size_hwc(self) -> tuple[int,int,int]:
        ''' get input image size in (height, width, channel)
        '''
        raise NotImplementedError()
    
    @property
    def model(self) -> ut_tf.WrappedFunction:
        return self.m_saved_model
    
    def init_by_saved_model(self, model_dir : str):
        ''' initialize by loading tensorflow v2 model
        '''
        loader = ut_tf.ModelLoader_SavedModelDir()
        loader.init(model_dir)
        loader.load_model()
        self.m_saved_model = loader.model
        self._build_preprocess_pipeline()
    
    def process_image(self, img : np.ndarray):
        if not self.m_saved_model:
            assert False, 'no model is loaded'
            
        # check image size
        if np.ndim(img) == 4:
            h,w = img.shape[1:3]
            _img = img
        else:
            h,w = img.shape[:2]
            _img = img[np.newaxis,:,:,:]
            
        assert np.allclose((h,w), self.input_size_hwc[:2]), 'wrong input size {}, expect {}'.format((h,w), self.input_size_hwc[:2])
            
        x = self._pre_process(_img)
        res = self._process_image(x)
        out = self._post_process(res, original_input=_img, model_input=x)
        return out
            
    def _pre_process(self, img : np.ndarray) -> tf.Tensor:
        x = self.m_preprocess(img)
        return x
    
    def _process_image(self, img : tf.Tensor) -> dict:
        return self.m_saved_model(img)
    
    def _post_process(self, result : dict, 
                      original_input : np.ndarray,
                      model_input : tf.Tensor) -> dict:
        ''' do post process over the result, and return a new result.
        
        parameters
        ---------------
        result
            the result returned by the model
        original_input
            a batch of multi channel images, in [N,H,W,C] format
        model_input
            image batched converted to tensor
            
        return
        ----------
        new_result
            processed result
        '''
        return result
    
    def _build_preprocess_pipeline(self):
        ''' convert image from rgb uint8 format to model expected format
        '''
        h, w, _ = self.input_size_hwc
        
        kly = tf.keras.layers
        proc = tf.keras.Sequential(
            [
                kly.Lambda(lambda x: tf.cast(x, dtype=tf.float32)),
                kly.Rescaling(1/255.0*2, -1)    # convert image to (-1,1) range
            ]
        )
        self.m_preprocess = proc
    
    
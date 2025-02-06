# code to deal with mediapipe's hand landmark model
import numpy as np
import tensorflow as tf
from igpy.ml.tf_model.SingleImageModel import SingleImageModel_tfv2

class HandLandmarkModel(SingleImageModel_tfv2):
    
    class ResultKey:
        Landmark = '1d_21_2d'
        Handedness = 'output_handflag'  # left or right hand, left=0, right=1
    
    def __init__(self) -> None:
        super().__init__()
        
    @property
    def input_size_hwc(self) -> tuple[int, int, int]:
        return (256,256,3)
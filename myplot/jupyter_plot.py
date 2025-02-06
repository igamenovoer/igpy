# plotting in jupyter notebook
import numpy as np
import PIL.Image
from IPython.display import display
import igpy.common.shortfunc as sf
from typing import Union

def imshow(img : Union[np.ndarray, PIL.Image.Image]):
    if isinstance(img, PIL.Image.Image):
        display(img)
    elif isinstance(img, np.ndarray):
        imgshow = sf.to_uint8(img)
        display(PIL.Image.fromarray(imgshow))
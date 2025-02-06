import pickle
import base64
import numpy as np
import numpy.typing as npt

def encode_to_base64(x : object) -> str:
    ''' Encode an object to base64.
    '''
    return base64.b64encode(pickle.dumps(x)).decode('utf-8')

def decode_from_base64(x : str) -> object:
    ''' Decode an object from base64.
    '''
    return pickle.loads(base64.b64decode(x.encode('utf-8')))

def encode_to_numpy_uint8(x : object) -> npt.NDArray[np.uint8]:
    ''' Encode an object to numpy uint8 array.'''
    return np.frombuffer(pickle.dumps(x), dtype=np.uint8)

def decode_from_numpy_uint8(x : np.ndarray) -> object:
    ''' Decode an object from numpy uint8 array.'''
    assert x.dtype == np.uint8, f'expect uint8, got {x.dtype}'
    return pickle.loads(x.data.tobytes())

# test encoding/decoding
# import attrs
# import torch
# from rich import print
# @attrs.define
# class MyTest:
#     x : int = 10
#     y : float = 20.0
#     z : torch.Tensor = attrs.field(factory=lambda: torch.rand(3, 4))
    
# obj_send = MyTest()
# x = encode_to_numpy_uint8(obj_send)
# obj_recv : MyTest = decode_from_numpy_uint8(x)

# print(obj_send)
# print(obj_recv)
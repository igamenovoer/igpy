import numpy as np

def attrs_check_nd_array(x : np.ndarray, ndim : int):
    if len(x.shape) != ndim:
        raise ValueError(f"required ndim={ndim}, got {x.ndim}")

attrs_check_1d_array = lambda obj, attribute, x : attrs_check_nd_array(x, 1)
attrs_check_2d_array = lambda obj, attribute, x: attrs_check_nd_array(x, 2)
attrs_check_3d_array = lambda obj, attribute, x: attrs_check_nd_array(x, 3)
attrs_check_4d_array = lambda obj, attribute, x: attrs_check_nd_array(x, 4)

def attrs_check_all_nd_array(x : list[np.ndarray], ndim : int):
    for _x in x:
        attrs_check_nd_array(_x, ndim)
        
attrs_check_all_1d_array = lambda obj, attribute, x : attrs_check_all_nd_array(x, 1)
attrs_check_all_2d_array = lambda obj, attribute, x : attrs_check_all_nd_array(x, 2)
attrs_check_all_3d_array = lambda obj, attribute, x : attrs_check_all_nd_array(x, 3)
attrs_check_all_4d_array = lambda obj, attribute, x : attrs_check_all_nd_array(x, 4)

def pydantic_check_nd_array(x : np.ndarray, ndim : int) -> np.ndarray:
    if len(x.shape) != ndim:
        raise ValueError(f"required ndim={ndim}, got {x.ndim}")
    return x

pydantic_check_1d_array = lambda x : pydantic_check_nd_array(x, 1)
pydantic_check_2d_array = lambda x : pydantic_check_nd_array(x, 2)
pydantic_check_3d_array = lambda x : pydantic_check_nd_array(x, 3)
pydantic_check_4d_array = lambda x : pydantic_check_nd_array(x, 4)

def pydantic_check_all_nd_array(x : list[np.ndarray], ndim : int) -> list[np.ndarray]:
    for _x in x:
        pydantic_check_nd_array(_x, ndim)
    return x

pydantic_check_all_1d_array = lambda x : pydantic_check_all_nd_array(x, 1)
pydantic_check_all_2d_array = lambda x : pydantic_check_all_nd_array(x, 2)
pydantic_check_all_3d_array = lambda x : pydantic_check_all_nd_array(x, 3)
pydantic_check_all_4d_array = lambda x : pydantic_check_all_nd_array(x, 4)

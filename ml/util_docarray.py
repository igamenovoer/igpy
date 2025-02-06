# miscellaneous functions
import numpy as np
import torch
from docarray import BaseDoc, DocVec
from typing import Any

# a type that can be hosted on different devices
DeviceHostableType = BaseDoc | DocVec | torch.Tensor


def to_device_and_dtype(
    doc: DeviceHostableType | list[DeviceHostableType] | dict[Any, DeviceHostableType],
    device: torch.device | None = None,
    dtype_mapping: dict[torch.dtype, torch.dtype] | None = None,
) -> DeviceHostableType | list[DeviceHostableType] | dict[Any, DeviceHostableType]:
    """
    Convert the doc or tensor to the specified device and dtype.

    Parameters
    -----------
    doc: DeviceHostableType | list[DeviceHostableType] | dict[Any, DeviceHostableType]
        The document, tensor, or collection to convert.
        DeviceHostableType includes BaseDoc, DocVec, and torch.Tensor.
        If it is not one of these types or their collections, it will be returned as is.
    device: torch.device | None
        The device to convert to. If None, keep the original device.
    dtype_mapping: dict[torch.dtype, torch.dtype] | None
        A mapping from the original dtype to the target dtype.

    Returns
    --------
    DeviceHostableType | list[DeviceHostableType] | dict[Any, DeviceHostableType]
        The converted document, tensor, or collection, same type as the input.
    """
    # non-collection types

    # if it is a tensor, convert it directly
    if isinstance(doc, torch.Tensor):
        if dtype_mapping is not None and doc.dtype in dtype_mapping:
            doc = doc.to(dtype=dtype_mapping[doc.dtype])
        return doc.to(device=device)

    # if it is a DocVec, convert it using the pre-defined method
    if isinstance(doc, DocVec):
        # convert with default method
        doc.to(device=device, dtype_mapping=dtype_mapping)

        # then, convert all fields because list[tensor] will be not automatically converted
        for d in doc:
            d = to_device_and_dtype(d, device=device, dtype_mapping=dtype_mapping)

        return doc

    # if it is a BaseDoc, convert its fields recursively
    if isinstance(doc, BaseDoc):
        # for each field, if it is a tensor or doc, convert it
        # if it is a list, convert each element in the list
        # if it is a dict, convert each value in the dict
        for field_name, field_info in doc.model_fields.items():
            field_value = getattr(doc, field_name)
            if field_value is None:
                # sometimes you have default None but not assignable
                continue
            converted = to_device_and_dtype(
                field_value, device=device, dtype_mapping=dtype_mapping
            )
            if isinstance(converted, torch.Tensor):
                # tensors are converted inplace, no need to re-assign
                # in inference mode, it may cause problem
                continue
            setattr(doc, field_name, converted)

    # collection types
    # if it is a list, convert each element in the list
    if isinstance(doc, list):
        for i, d in enumerate(doc):
            doc[i] = to_device_and_dtype(d, device=device, dtype_mapping=dtype_mapping)
        return doc

    # if it is a dict, convert each value in the dict
    if isinstance(doc, dict) and all(isinstance(v, BaseDoc) for v in doc.values()):
        for k, v in doc.items():
            doc[k] = to_device_and_dtype(v, device=device, dtype_mapping=dtype_mapping)
        return doc

    # unknown type
    return doc

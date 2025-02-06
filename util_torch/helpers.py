import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable, TypeVar
import re

TorchDevice = Union[str, torch.device]
TensorContainer = TypeVar("TensorContainer", list, dict, tuple)


def clone_tensors(
    tensors: TensorContainer | torch.Tensor | None, clone_method: str
) -> TensorContainer | torch.Tensor | None:
    """clone a list/dict/tuple of tensors

    parameters
    ----------------
    tensors : list/dict/tuple of tensors or None
        a list/dict/tuple of tensors, or None
    clone_method : str, {'clone', 'recreate'}
        'clone' or 'recreate', where 'clone' will call .clone(),
        'recreate' will create a new tensor with the same shape and type, and copy the data from the original tensor to the new tensor,
        using zeros_like() and copy_()

    return
    ---------------
    cloned_tensors : list/dict/tuple of cloned tensors or None
        a list/dict/tuple of cloned tensors, or None
    """
    if tensors is None:
        return None

    # leaf case
    if isinstance(tensors, torch.Tensor):
        if clone_method == "clone":
            return tensors.clone()
        elif clone_method == "recreate":
            out = torch.zeros_like(tensors)
            out.copy_(tensors)
            return out
        else:
            raise ValueError(f"invalid clone method: {clone_method}")

    # recursive case
    if isinstance(tensors, list):
        return [clone_tensors(t, clone_method) for t in tensors]
    elif isinstance(tensors, dict):
        return {k: clone_tensors(v, clone_method) for k, v in tensors.items()}
    elif isinstance(tensors, tuple):
        return tuple([clone_tensors(t, clone_method) for t in tensors])
    else:
        raise ValueError("input must be a list/dict/tuple of tensors")


def list_all_cuda_tensors(func_filter: Callable[[torch.Tensor], bool] = None):
    """list all cuda tensors in the current python process

    parameters
    ----------------
    func_filter
        a function to filter the tensors to be listed. if not None,
        only the tensors that pass the filter (returns True) will be listed
    """
    import torch
    import gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if func_filter is not None:
                    if not func_filter(obj):
                        continue

                if "cuda" in str(obj.device):
                    if func_filter is not None:
                        if not func_filter(obj):
                            continue
                    print(type(obj), obj.size(), obj.device, obj.__class__.__name__)
        except:
            pass


def get_all_torch_devices(gpu_only: bool = False) -> list[torch.device]:
    """get all available torch devices"""
    devlist: list[torch.device] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devlist.append(torch.device(f"cuda:{i}"))

    if not gpu_only:
        devlist.append(torch.device("cpu"))

    return devlist


def get_canonical_torch_device(device: Union[str, torch.device]) -> torch.device:
    """get the canonical torch device, given a device name or torch.device object"""
    dev_out: torch.device = None
    if isinstance(device, str):
        if device == "cuda":
            dev_out = torch.device("cuda:0")
        else:
            dev_out = torch.device(device)
    elif isinstance(device, torch.device):
        if str(device) == "cuda":
            dev_out = torch.device("cuda:0")
        else:
            dev_out = device
    return dev_out


def exec_sequential_model_and_print(model: nn.Sequential, x: torch.Tensor):
    y = x
    for name, block in model.named_children():
        print("LAYER = {0}".format(name))
        z = block(y)
        print("IN = {0}, OUT = {1}".format(tuple(y.shape), tuple(z.shape)))
        y = z
    return y


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class SequenceModule(nn.Module):
    """a module that contains a nn.Sequential object as the model, and execute the model manually
    layer by layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m_model: nn.Sequential = self.init_model(*args, **kwargs)
        self.m_print_while_execution = False

    def init_model(self, *args, **kwargs) -> nn.Sequential:
        pass

    def set_print_while_execution(self, is_print: bool, recursive=True):
        """print the layer info while execution?"""
        self.m_print_while_execution = is_print

        if recursive and self.m_model is not None:
            for block in self.m_model.children():
                if isinstance(block, SequenceModule):
                    block.set_print_while_execution(is_print, recursive)

    @property
    def print_while_execution(self):
        return self.m_print_while_execution

    def forward_with_print(self, x):
        y = x
        for name, block in self.m_model.named_children():
            print("LAYER = {0}".format(name))
            z = block(y)
            print("IN = {0}, OUT = {1}".format(tuple(y.shape), tuple(z.shape)))
            y = z
        return y

    def forward(self, x):
        if self.m_print_while_execution:
            return self.forward_with_print(x)
        else:
            return self.m_model(x)


class LayerParams:
    def __init__(self):
        self.bn_momentum = 0.1
        self.bn_eps = 1e-5
        self.leaky_relu_slope = 1e-2


def get_padding_same(kernel_size):
    """get the padding for a layer with a specific kernel size so that the conv result has the same size as input"""
    output: np.ndarray = np.atleast_1d(kernel_size) // 2
    if len(output) == 1:
        output = int(output[0])
    else:
        output = tuple(output)

    return output


def layer_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding="same",
    use_bn=False,
    activation_func="relu",
    layer_params: LayerParams = None,
    add_to_module: nn.Module = None,
):
    """create a 2d conv layer, optinally with batch normalization and activation

    parameters
    ----------------
    in_channels, out_channels
        number of input and output channels
    kernel_size
        integer or tuple, the kernel size
    padding
        'same','valid', integer or tuple to indicate how many pixels to pad for each border
    use_bn
        whether to use batch normalization
    add_to_module
        if set, the layers will be added to this module

    return
    ---------------
    conv2d_layer
        the nn.Conv2d() layer
    bn_layer
        the nn.BatchNorm2d() layer, or None
    act_layer
        the activation layer, or None
    """
    if layer_params is None:
        layer_params = LayerParams()

    bn_momentum = layer_params.bn_momentum
    bn_eps = layer_params.bn_eps
    act_leaky_slope = layer_params.leaky_relu_slope

    if isinstance(padding, str):
        padding = padding.lower()

    if isinstance(activation_func, str):
        activation_func = activation_func.lower()

    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            _padding = get_padding_same(kernel_size)
        elif padding == "valid" or padding is None:
            _padding = 0
    else:
        _padding = padding

    # if batch norm is enabled, bias in conv layer is not needed anymore
    use_bias = not use_bn

    # create convolution layer
    ly_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=_padding,
        bias=use_bias,
    )

    # create batch normalization
    if not use_bn:
        ly_bn = None
    else:
        ly_bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)

    # create activation
    if activation_func is None:
        ly_act = None
    elif activation_func == "relu":
        ly_act = nn.ReLU()
    elif activation_func == "leaky":
        ly_act = nn.LeakyReLU(act_leaky_slope)

    if add_to_module:
        add_to_module.add_module("Conv2d", ly_conv)
        if ly_bn is not None:
            add_to_module.add_module("BatchNorm2d", ly_bn)
        if ly_act is not None:
            add_to_module.add_module("activation", ly_act)

    # done
    return ly_conv, ly_bn, ly_act


class Conv2d(nn.Module):
    """2d convolution layer with batch norm and activation"""

    def __init__(self):
        super().__init__()
        self.m_conv: nn.Conv2d = None
        self.m_bn: nn.BatchNorm2d = None
        self.m_activation = None

    def init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        use_bn=False,
        activation_func=None,
        layer_params: LayerParams = None,
    ):
        """
        initialize this layer, optinally with batch normalization and activation

        parameters
        ----------------
        in_channels, out_channels
            number of input and output channels
        stride
            the stride of the convolution
        kernel_size
            integer or tuple, the kernel size
        padding
            'same','valid', integer or tuple to indicate how many pixels to pad for each border
        use_bn
            whether to use batch normalization
        activation_func
            name of the activation function, or None if no activation function is required.
            can be 'relu' or 'leaky'
        """
        ly_conv, ly_bn, ly_act = layer_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bn=use_bn,
            activation_func=activation_func,
            layer_params=layer_params,
        )
        self.m_conv = ly_conv
        self.m_bn = ly_bn
        self.m_activation = ly_act

    def forward(self, x):
        y = self.m_conv(x)
        if self.m_bn is not None:
            y = self.m_bn(y)

        if self.m_activation is not None:
            y = self.m_activation(y)

        return y


class VOCDetectionLabelTransform:
    """transform VOC label into class and bounding box"""

    OBJECT_LABELS = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(self):
        # the images will be resized to this size
        # if not set, will not be resized
        # if dimension is -1 (e.g., (256, -1)), then aspect ratio is kept
        self.m_target_image_size_hw: np.ndarray = None
        self.m_label2id = {lb: i for i, lb in enumerate(self.OBJECT_LABELS)}
        self.m_id2label = self.OBJECT_LABELS

    def set_target_image_size(self, height, width):
        """set target image size

        parameters
        ---------------
        height, width:
            the target dimension, if -1, then that dimension is determined by keeping the aspect ratio
        """
        self.m_target_image_size_hw = np.array([height, width])

    def transform(self, label) -> dict:
        """transform the voc label into numbers and vectors

        return as dict
        ----------------
        object_index : np.ndarray
            1xN, the indices of objects, defined by self.m_label2id
        bbox : np.ndarray
            Nx4 matrix, object bounding boxes, in (x,y,w,h) format.
        imgsize : np.ndarray
            1x2, image size in (height, width)
        """
        jdata_objs = label["annotation"]["object"]
        n_obj = len(jdata_objs)

        obj_index = np.zeros(n_obj, dtype=int)
        obj_bbox = np.zeros((n_obj, 4))

        j_size = label["annotation"]["size"]

        for i, obj in enumerate(jdata_objs):
            key = obj["name"]
            idx = self.m_label2id[key]
            obj_index[i] = idx

            j_box = obj["bndbox"]
            xmin, ymin, xmax, ymax = (
                int(j_box["xmin"]),
                int(j_box["ymin"]),
                int(j_box["xmax"]),
                int(j_box["ymax"]),
            )
            w = xmax - xmin
            h = ymax - ymin
            obj_bbox[i] = (xmin, ymin, w, h)

        src_width, src_height = int(j_size["width"]), int(j_size["height"])
        imgsize = np.array([src_height, src_width])

        if self.m_target_image_size_hw is not None and np.any(
            self.m_target_image_size_hw > 0
        ):
            dst_width, dst_height = self.m_target_image_size_hw[::-1]
            x_scale = dst_width / src_width
            y_scale = dst_height / src_height

            # if -1 is specified as target dimension, do uniform scaling
            if x_scale < 0:
                x_scale = y_scale
            if y_scale < 0:
                y_scale = x_scale

            obj_bbox[:, [0, 2]] = obj_bbox[:, [0, 2]] * x_scale
            obj_bbox[:, [1, 3]] = obj_bbox[:, [1, 3]] * y_scale

            imgsize = np.array(self.m_target_image_size_hw)
            if imgsize[0] < 0:
                imgsize[0] = src_height * y_scale
            if imgsize[1] < 0:
                imgsize[1] = src_width * x_scale
            imgsize = np.round(imgsize).astype(int)

        output = {"obj_index": obj_index, "obj_bbox": obj_bbox, "imgsize": imgsize}

        return output

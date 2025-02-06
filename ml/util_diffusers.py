# use with diffusers library
import torch
import numpy as np
from attrs import define, field
from typing import Any

from igpy.common.EnumBase import CustomEnumBase

# must have peft and diffusers
import peft
from diffusers.models import UNet2DConditionModel

@define
class SafetensorData:
    fn_tensor : str | None = field(default=None)    # the file name of the safetensor
    data : dict[str, torch.Tensor | np.ndarray] | None = field(factory=dict)    # the data of the safetensor
    metadata : dict[str, Any] | None = field(factory=dict)    # the metadata of the safetensor

def load_safetensor_as_torch_tensor(fn_tensor : str) -> SafetensorData:
    ''' load safetensor from file into a dict of torch tensor
    
    parameters
    ------------
    fn_tensor : str
        file name of safetensor
        
    return
    --------
    out : SafetensorData
        the loaded data
    '''
    import safetensors
    
    data_dict = {}
    metadata : dict[str, str] = {}
    with safetensors.safe_open(fn_tensor, framework='pt') as f:
        for k in f.keys():
            data_dict[k] = f.get_tensor(k)
        metadata = f.metadata()
    out = SafetensorData(
        fn_tensor=fn_tensor,
        data=data_dict,
        metadata=metadata
    )
    return out

class AdapterType(CustomEnumBase):
    lora = 'lora'
    ip_adapter = 'ip_adapter'

@define(kw_only=True, eq=False)
class AdapterInfo:
    name : str = field()    # the name of the adapter
    type : str = field()    # the kind of the adapter, default to lora
    weight : float = field(default=1.0)    # the weight of the adapter
    is_enabled : bool = field(default=True)    # whether the adapter is enabled

@define(kw_only=True, eq=False)
class StableDiffusionAdapterHelper:
    ''' demonstrate how to load adapters into SD 1.5 unet model
    '''
    unet : UNet2DConditionModel = field()    # the unet model
    device : str | None = field(default=None)    # the device to use
    torch_dtype : torch.dtype | None = field(default=None)    # the dtype to use
    adapters : dict[str, AdapterInfo] = field(factory=dict)    # the adapters
    
    def load_lora_by_state_dict(self, name:str, state_dict : dict[str,torch.Tensor], weight:float=1.0):
        ''' load a lora into unet by state_dict, read from safetensor.
        Note that the loras are not activated by default, you need use update_adapters() to activate them.
        
        parameters
        -------------
        name : str
            the name of the adapter, must be unique among all adapters
        state_dict : dict[str,torch.Tensor], directly loaded from safetensor
            the state_dict of the lora
        weight : float
            the weight of the adapter, default to 1.0
        '''
        
        from diffusers.loaders.lora import LoraLoaderMixin
        from peft.tuners.lora.config import LoraConfig
        import copy
        
        # check if the given name is already in the adapters
        # adapters : dict[str, Any] = self.unet.peft_config
        # assert name not in adapters, f'adapter name {name} already exists'
        
        # create a new adapter by loading lora into unet
        _state_dict = copy.copy(state_dict) # prevent modifying the original state_dict
        lora_dict, network_alphas = LoraLoaderMixin.lora_state_dict(copy.copy(_state_dict), unet_config=self.unet.config)
        LoraLoaderMixin.load_lora_into_unet(lora_dict, network_alphas, self.unet, adapter_name=name)
        
        self.adapters[name] = AdapterInfo(name=name, type=AdapterType.lora, weight=weight)
        
    def update_adapters(self):
        ''' set the adapters used by the unet, based on adapter info.
        The weights are updated if they are not zero, otherwise the adapter is disabled.
        '''
        ap_names : list[str] = []
        ap_weights : list[float] = []
        
        for k,v in self.adapters.items():
            if v.weight > 0 and v.is_enabled:
                ap_names.append(k)
                ap_weights.append(v.weight)
        
        # FIXME: do we need to call disable_adapters () first?
        self.unet.set_adapters(ap_names, ap_weights)
        
        # make sure the adapter is loaded into the device and dtype
        if self.device is not None or self.torch_dtype is not None:
            self.unet.to(device=self.device, dtype=self.torch_dtype)
    
    def remove_adapter(self, name : str):
        ''' unload a specific adapter from the unet
        '''
        assert name in self.adapters, f'adapter name {name} does not exist'
        self.unet.delete_adapters([name])
        self.adapters.pop(name)
        
    # def disable_adapter(self, name:str):
    #     ''' disable specific adapter. You need to call update_adapters() to make it effective
    #     '''
    #     ap = self.adapters.get(name)
    #     assert ap is not None, f'adapter name {name} does not exist'
    #     ap.is_enable = False
        
    # def enable_adapter(self, name:str):
    #     ''' enable specific adapter. You need to call update_adapters() to make it effective
    #     '''
    #     ap = self.adapters.get(name)
    #     assert ap is not None, f'adapter name {name} does not exist'
    #     ap.is_enable = True
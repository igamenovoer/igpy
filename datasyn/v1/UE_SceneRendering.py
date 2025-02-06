import numpy as np
from typing import Union
from .SceneRendering import SceneRendering, RenderedObject, CameraModel
from igpy.datasyn.ExrImage import ExrImage, CryptomatteIDMap, CryptomatteLayer

class UE_RenderedObject(RenderedObject):
    ''' object rendered by unreal engine
    '''
    def __init__(self) -> None:
        super().__init__()
        self.m_owner : UE_SceneRendering = None     # the rendering that contains this object
        
class UE_SceneRendering(SceneRendering):
    ''' exr image rendered by unreal engine
    '''
    COLOR_CHANNEL_DEFAULT_PREFIX = ['PathTracer', None]
    COLOR_CHANNEL_DEFAULT_POSTFIX = ['R','G','B','A']
    
    # search for these names to find depth channel, the first one found is used
    DEPTH_CHANNEL_NAMES = [
        'FinalImageMovieRenderQueue_WorldDepth.R',
        'FinalImageSceneDepth.R'
    ]
    
    def __init__(self) -> None:
        super().__init__()
        
        self.m_exr_image : ExrImage = None # current exr image
        self.m_cryptomatte_max_rank = 0 # [0,1,2], the 
        
        # label the objects
        self.m_name2rdobj : dict[str, UE_RenderedObject] = {}
        self.m_code2rdobj : dict[int, UE_RenderedObject] = {}   #indexed by code
        
        # color channel        
        self.m_chn_rgba : list[str] = None  # rgba channel names
        self.m_chn_rgba_postfix : list[str] = list(self.COLOR_CHANNEL_DEFAULT_POSTFIX) # rgba post fixes
        
        # self.m_name2class: dict[str, SemanticClass] = {}
        self.m_idcode_background : int = CryptomatteIDMap.IDCODE_BACKGROUND # background id, representing empty space in label map
        
    @property
    def rendered_objects_by_name(self) -> dict[str, UE_RenderedObject]:
        ''' scene objects indexed by instance names
        '''
        return self.m_name2rdobj
    
    @property
    def rendered_objects_by_instance_code(self) -> dict[int, UE_RenderedObject]:
        ''' scene objects indexed by instance code. 
        '''
        return self.m_code2rdobj
    
    @property
    def image_size_hw(self) -> np.ndarray:
        ''' image size in (height, width)
        '''
        if self.m_exr_image:
            return self.m_exr_image.image_size_hw
        else:
            return None
    
    def init(self, exr_image : ExrImage):
        self.m_exr_image = exr_image
        self.m_name2rdobj = {}
        
        if not self.m_chn_rgba:
            chn_names = self._find_rgba_channel(self.m_exr_image)
            self.m_chn_rgba = chn_names
        
    @staticmethod
    def create_from_exr_file(fn_exr_image : ExrImage, parse_now : bool = False) -> "UE_SceneRendering":
        out = UE_SceneRendering()
        exr_image = ExrImage()
        exr_image.open_file(fn_exr_image)
        out.init(exr_image)
        
        if parse_now:
            out.parse()
            
        return out
    
    def get_rendered_object_by_name(self, name : str) -> UE_RenderedObject:
        '''  get a rendering by name. If not exists, return None
        '''
        if self.m_name2rdobj is None:
            return None
        return self.m_name2rdobj.get(name)
    
    def get_rendered_object_by_code(self, code : int) -> UE_RenderedObject:
        '''  get a rendering by code in rendered object. If not exists, return None
        '''
        if self.m_code2rdobj is None:
            return None
        return self.m_code2rdobj.get(code)
    
    def parse(self):
        ''' parse the objects and labels. If meshes exist, also relate them to scene objects.
        '''
        
        assert self.m_exr_image is not None, 'load exr image first'
        
        # create scene objects
        name2obj : dict[str, UE_RenderedObject] = {}
        self.m_name2rdobj = name2obj
        
        # parse label from exr image, and assign instance id code
        # the exr instance id code is consistent across frames
        self.m_exr_image.parse_cryptomatte(max_rank=self.m_cryptomatte_max_rank)
        exr_name2code = self.m_exr_image.crypto_name2code   # actor_name vs. idcode
        
        # no scene structure? use exr image to initilaize scene objects
        for name, code in exr_name2code.items():
            sc = UE_RenderedObject()
            sc.m_owner = self
            sc.m_name = name
            sc.m_code = code
            name2obj[name] = sc
        
        # extract pixel indices and assign them to objects
        import scipy.ndimage as ndimg
        from igpy.common.shortfunc import get_components_from_labelmap
        
        idmap_by_rank = self.m_exr_image.crypto_idmaps
        all_labels = list(exr_name2code.values())
        label2pix_final : dict[int, np.ndarray] = {}
        for idxrank, idmap in enumerate(idmap_by_rank):
            lbmap = idmap.label_map
            label2pix_thisrank = get_components_from_labelmap(lbmap, all_labels)
            
            # concatenate the pix indices to existing ones
            for lb, idxpix in label2pix_thisrank.items():
                idxpix_prev = label2pix_final.get(lb)
                if idxpix_prev is None:
                    label2pix_final[lb] = idxpix
                else:
                    label2pix_final[lb] = np.concatenate([idxpix_prev, idxpix])
                    
        for name, obj in name2obj.items():
            idcode = obj.code
            idxpix = label2pix_final.get(idcode)
            obj.m_pixel_indices = idxpix
            obj.m_image_size_hw = self.m_exr_image.image_size_hw
            
        # index by code
        self.m_code2rdobj = {x.code:x for x in self.m_name2rdobj.values()}
            
    def set_exr_cryptomatte_max_rank(self, max_rank : int):
        self.m_cryptomatte_max_rank = max_rank
        
    def set_exr_image(self, img : ExrImage):
        assert isinstance(img ,ExrImage)
        self.m_exr_image = img
        
    def set_camera(self, cam : CameraModel):
        ''' set camera in opencv convention
        '''
        self.m_camera = cam
        
    def get_color_image(self, dtype=np.uint8, gamma : float = None, rgb_only:bool = False):
        ''' get RGBA image from exr file
        '''
        if not self.m_chn_rgba:
            return None
        
        chn_names = list(self.m_chn_rgba)
        if rgb_only:
            chn_names[-1] = None
            
        output = self.m_exr_image.get_color_image(
            r_chn=chn_names[0],
            g_chn=chn_names[1],
            b_chn=chn_names[2],
            a_chn=chn_names[3],
            dtype = dtype,
            gamma=gamma
        )  
        return output
    
    def get_depth_image(self, dtype=None, channel_name : str = None) -> np.ndarray:
        ''' get depth image from exr file, in floating point format
        
        parameters
        ----------
        dtype
            if not None, convert the output to specified dtype
        channel_name
            if not None, use this channel name as depth channel name
        '''
        if channel_name is None:
            all_names = set(self.m_exr_image.channel_names)
            for cand in self.DEPTH_CHANNEL_NAMES:
                if cand in all_names:
                    channel_name = cand
                    break
        
        # depth not found
        if channel_name is None:
            return None
        
        output = self.m_exr_image.get_channel(channel_name)
        if dtype is not None:
            output = output.astype(dtype)
        return output
    
    def _find_rgba_channel(self, img : ExrImage) -> list[str]:
        ''' find default RGBA channel names
        '''
        
        output = []
        for prefix in self.COLOR_CHANNEL_DEFAULT_PREFIX:
            if prefix is None:  # no prefix, the postfix is the channel name
                expect_chn_names = self.m_chn_rgba_postfix
            else:
                expect_chn_names = ['{}.{}'.format(prefix, p) for p in self.m_chn_rgba_postfix]
                
            # find channels
            found_channel = 0
            for chn_name in img.channel_names:
                if chn_name in expect_chn_names:
                    found_channel += 1
                    
            if found_channel == len(expect_chn_names):
                output = expect_chn_names
                break
            
        return output
    
    def get_instance_label_map(self, return_all_ranks : bool = False) -> Union[np.ndarray, list[np.ndarray]]:
        ''' get instance label map of this frame
        
        parameters
        -------------
        return_all_ranks
            if True, return a list of label maps, each corresponding to a rank of the exr image
            
        return
        -----------
        lbmap_or_list : np.ndarray | list
            a label map of the highest rank in exr if return_all_ranks==False.
            Otherwise, return a list of label maps, sorted by rank, 
            output[0] is of the highest rank
        '''
        if not self.m_exr_image:
            return None
        
        output = [x.label_map for x in self.m_exr_image.crypto_idmaps]
        if return_all_ranks:
            return output
        else:
            return output[0]
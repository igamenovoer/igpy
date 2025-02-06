import numpy as np
import OpenEXR as exr
import Imath
from typing import Union, Tuple
from igpy.common import shortfunc as sf
import re
import struct


def _name_to_float32(name):
    import mmh3
    
    """Convert a string to an 8-digit hexadecimal Cryptomatte ID."""
    hash_32 = mmh3.hash(name, signed=False)
    exp = hash_32 >> 23 & 255
    if (exp == 0) or (exp == 255):
        hash_32 ^= 1 << 23
    packed = struct.pack("<L", hash_32 & 0xffffffff)
    return struct.unpack("<f", packed)[0]

def _float32_to_int32(value):
    packed = struct.pack("<f", value)
    # return struct.unpack("<L", packed)[0]
    return struct.unpack("<i", packed)[0]

class CryptomatteIDMap:
    IDCODE_BACKGROUND = 0   # background id code
    
    def __init__(self) -> None:
        self.m_id_label_map : np.ndarray = None # int32 idmap
        self.m_coverage : np.ndarray = None
        self.m_rank : int = None
        
        self.m_chn_name_label_map : str = None
        self.m_chn_name_coverage : str = None
        
    @property
    def label_map(self) -> np.ndarray:
        ''' per-pixel label map, output[i,j]=k iff pixel (i,j) is labeled as k
        '''
        return self.m_id_label_map
    
    @property
    def coverage_map(self) -> np.ndarray:
        ''' per-pixel weight of label map, output[i,j]=w iff pixel (i,j)'s weight is w
        '''
        return self.m_coverage
class CryptomatteLayer:
    def __init__(self) -> None:
        self.m_idmaps : list[CryptomatteIDMap] = None # id map per rank
        self.m_name2code : dict[str, int] = None  # mapping object name to id code
        
        # cryptomatte/<layer_id>/name/<chn_base_name>
        self.m_layer_id : str = None
        self.m_chn_base_name : str = None 

class ExrImage:
    ''' Data reader for OpenEXR image.
    Note that to use cryptomatte-related functions, you need to call parse_cryptomatte() first.
    '''
    ExrTypeToNumpyType = {
        Imath.PixelType.FLOAT : np.float32,
        Imath.PixelType.HALF : np.float16,
        Imath.PixelType.UINT : np.uint8
    }
    
    def __init__(self) -> None:
        self.m_filename : str = None
        self.m_exr_filedata : exr.InputFile = None
        self.m_imgsize_hw : np.ndarray = None
        self.m_exr_header : dict = None
        
        # cryptomatte info
        self.m_crypto_layers : dict[str, CryptomatteLayer] = None
        self.m_working_crypto_layer : CryptomatteLayer = None
        
        # self.m_crypto_idmaps : list[CryptomatteIDMap] = None  # id map per rank
        # self.m_crypto_name2code : dict[str, int] = None # mapping object name to id code
        
    @property
    def image_size_hw(self):
        ''' image size in (height, width)
        '''
        return self.m_imgsize_hw
    
    @property
    def header(self):
        ''' exr header
        '''
        return self.m_exr_header
    
    @property
    def channel_names(self) -> list[str]:
        ''' get all channel names
        '''
        if self.m_exr_header:
            return self.m_exr_header['channels']
        else:
            return None
    
    @property
    def crypto_layers(self):
        return self.m_crypto_layers
    
    @property
    def working_crypto_layer(self):
        return self.m_working_crypto_layer
    
    @property
    def crypto_name2code(self):
        ''' cryptomatte codes for current layer
        '''
        if self.m_working_crypto_layer is not None:
            return self.m_working_crypto_layer.m_name2code
        else:
            return None
            
    @property
    def crypto_idmaps(self):
        ''' cryptomatte id maps for current layer
        '''
        if self.m_working_crypto_layer is not None:
            return self.m_working_crypto_layer.m_idmaps
        else:
            return None
        
    def open_file(self, exr_filename_or_bytes : str | bytes):
        ''' open exr file and parse it
        '''
        import io
        
        if isinstance(exr_filename_or_bytes, str):  # as filename
            self.m_filename = exr_filename_or_bytes
            self.m_exr_filedata = exr.InputFile(exr_filename_or_bytes)
        elif isinstance(exr_filename_or_bytes, bytes): # as byte stream
            self.m_exr_filedata = exr.InputFile(io.BytesIO(exr_filename_or_bytes))
        else:
            raise ValueError('exr_filename_or_bytes should be either a filename or a byte stream')
        
        header = self.m_exr_filedata.header()
        self.m_exr_header = header
        
        # get image size
        key_imgsize = 'dataWindow'
        pt_max = header[key_imgsize].max
        pt_min = header[key_imgsize].min
        width = pt_max.x - pt_min.x + 1
        height = pt_max.y - pt_min.y + 1
        imgsize_hw = np.array([height ,width])
        self.m_imgsize_hw = imgsize_hw
    
    def set_working_cryptomatte_layer(self, layer_id : str):
        self.m_working_crypto_layer = self.m_crypto_layers[layer_id]
        
    def get_cryptomatte_layer_ids(self) ->list :
        return list(self.m_crypto_layers.keys())
    
    def parse_cryptomatte(self, max_rank : int = None):
        ''' parse cryptomatte-related data in file
        
        parameters
        --------------
        max_rank : None, or 0,1,2
            the max id rank to parse. Cryptomatte has 3 ranks, the higher the less important.
            By default, parse all ranks
        '''
        # find cryptomatte info
        self.m_crypto_layers = {}
        
        header = self.m_exr_header
        ptn_cryptomatte = re.compile('cryptomatte/(?P<layer_id>.+)/manifest')
        
        # convert rgba key word to ranking int, used to sort cryptomatte
        rgba2rank = {}
        for idx, x in enumerate('rgba'):
            rgba2rank[x] = idx
        for idx, x in enumerate(['red','green','blue','alpha']):
            rgba2rank[x] = idx
            
        for key, val in header.items():
            m = ptn_cryptomatte.match(key)
            if m is None:
                continue
            
            # parse layer id
            layer = CryptomatteLayer()
            layer.m_layer_id = m.groupdict()['layer_id']
            self.m_crypto_layers[layer.m_layer_id] = layer
            
            # parse base name
            key_basename = 'cryptomatte/{}/name'.format(layer.m_layer_id)
            layer.m_chn_base_name = header[key_basename].decode()
            
            # read channels
            # example : ActorHitProxyMask00.R
            rank2idmap = {}
            ptn_idmask_lower = re.compile(r'{}(?P<rank>\d+)\.(?P<rgba>([rgba]|red|blue|green|alpha))$'.format(layer.m_chn_base_name.lower()))
            channels : dict = header['channels']
            for key, val in channels.items():
                m = ptn_idmask_lower.match(key.lower())
                if m is None:
                    continue
                
                # find rank and mask type
                token_rgba : str = m.groupdict()['rgba']
                rgba_index = rgba2rank[token_rgba]
                rank_index = int(m.groupdict()['rank'])
                final_rank = rank_index * 2 + rgba_index // 2
                mask_type = 'id' if rgba_index % 2 == 0 else 'weight'
                
                # skip if this rank is not required to parse
                if max_rank is not None:
                    if final_rank > max_rank:
                        continue
                
                # print('channel={}, rank={}, masktype={}'.format(key, final_rank, mask_type))
                
                # create or get idmap
                idmap : CryptomatteIDMap = rank2idmap.get(final_rank, CryptomatteIDMap())
                rank2idmap[final_rank] = idmap
                    
                # fill in info
                idmap.m_rank = final_rank
                if mask_type == 'id':
                    idmap.m_chn_name_label_map = key
                    
                    # TODO: determine if the view should be int32 or int64
                    idmap.m_id_label_map = self.get_channel(key).view(np.int32).copy()
                else:
                    idmap.m_chn_name_coverage = key
                    idmap.m_coverage = self.get_channel(key)
                
                # end of parsing channels
            layer.m_idmaps = list(sorted(rank2idmap.values(), key=lambda x:x.m_rank))
            
            # parse manifest
            key_manifest = 'cryptomatte/{}/manifest'.format(layer.m_layer_id)
            manifest : dict = None
            manifest = eval(header[key_manifest].decode())
            layer.m_name2code = {}
            for name, idhex in manifest.items():
                idcode = _float32_to_int32(_name_to_float32(name))
                layer.m_name2code[name] = int(idcode)
                
        if self.m_crypto_layers:
            self.m_working_crypto_layer = list(self.m_crypto_layers.values())[0]
    
    def get_channel(self, channel_name : str) -> np.ndarray:
        ''' get the image data of a single channel
        '''
        chn_data : bytes = self.m_exr_filedata.channel(channel_name)
        exr_dtype = self.m_exr_header['channels'][channel_name].type.v
        
        img = np.frombuffer(chn_data, dtype = self.ExrTypeToNumpyType[exr_dtype]).reshape(self.m_imgsize_hw)
        return img
            
    def get_color_image(self, r_chn:str, g_chn:str = None, 
                          b_chn:str = None, a_chn:str = None, 
                          dtype = np.float32, gamma : float = None) -> np.ndarray:
        ''' parse exr file content to get a grayscale or color image
        
        parameters
        ---------------
        r_chn, g_chn, b_chn, a_chn
            Channel names for r,g,b,a channels. 
            If only r_chn is provided, parse for a grayscale image.
            Otherwise, parse for rgb or rgba image
        dtype
            the output data type
        gamma : float
            apply gamma correction to the original data. If None, no gamma correction is apply
            
        return
        -----------
        image
            the output image in dtype format
        '''
        r_img = self.get_channel(r_chn)
        g_img = self.get_channel(g_chn) if g_chn is not None else None
        b_img = self.get_channel(b_chn) if b_chn is not None else None
        a_img = self.get_channel(a_chn) if a_chn is not None else None
        
        out_image : np.ndarray = None
        if g_img is None:   #we only have r image
            out_image = r_img
        elif a_img is None:   #we have r,g,b
            out_image = np.dstack((r_img, g_img, b_img))
        else: # we have rgba
            out_image = np.dstack((r_img, g_img, b_img, a_img))
            
        if gamma:
            if len(out_image.shape) == 4:
                # do not apply gamma to alpha channel
                out_image[:,:,:3] = np.power(out_image[:,:,:3].clip(0, np.Inf), 1.0/gamma)
            else:
                out_image = np.power(out_image.clip(0, np.Inf), 1.0/gamma)
            
        # convert data type
        if out_image.dtype == dtype:
            return out_image
        elif out_image.dtype in (np.float16, np.float32) and dtype in (np.float64, np.float32):
            return out_image.astype(dtype=dtype)
        elif dtype == np.uint8:
            return (out_image * 255).clip(0,255).astype(np.uint8)
    
    def get_cryptomatte_remap_code(self, name_pattern_to_code : dict = None, 
                                   case_sensitive : bool = True,
                                   regex_method = 'match',
                                   unmatched_code : int = CryptomatteIDMap.IDCODE_BACKGROUND) -> dict:
        ''' assign object name with new id code, based on name patterns
        
        parameters
        ---------------
        name_pattern_to_code : dict
            each entry is (pattern, id_code), where pattern is a regex name pattern, and id_code
            is the new code assigned to all names that matches this pattern. Names that do not match
            any pattern is assigned unmatched_code
        regex_method : 'search' or 'match'
            use which method to match the pattern to name.
            If 'match', pattern should match exactly to the name.
            If 'search', the pattern can match parts of the name
        unmatched_code : int or None
            code assigned to names that do not match any pattern, or None to keep the original code.
        case_sensitive
            the match is case sensitive or not
            
        return
        ----------
        name_to_code : dict
            mapping object name to new code
        '''
        name_to_newcode : dict = {}
        
        # compile all patterns
        ptn_to_re : dict[str, re.Pattern] = None
        if case_sensitive:
            ptn_to_re = {x:re.compile(x) for x in name_pattern_to_code.keys()}
        else:
            ptn_to_re = {x:re.compile(x.lower()) for x in name_pattern_to_code.keys()}
        
        # for each old code, find its new code based on name pattern
        for name, oldcode in self.crypto_name2code.items():
            if unmatched_code is None:
                name_to_newcode[name] = oldcode
            else:
                name_to_newcode[name] = unmatched_code
                
            for ptn, re_obj in ptn_to_re.items():
                func_match = re_obj.search if regex_method=='search' else re_obj.match
                
                if case_sensitive:
                    m = func_match(name)
                else:
                    m = func_match(name.lower())
                    
                if m is not None:
                    newcode = name_pattern_to_code[ptn]
                    name_to_newcode[name] = newcode
                    
        return name_to_newcode
        
    def get_cryptomatte_id_map(self, name2code : dict = None, 
                               unmatched_code : int = None) -> list:
        output = []
        oldcode_to_newcode : dict = {}
        if name2code is not None:
            # remap id code
            for name, oldcode in self.crypto_name2code.items():
                if name in name2code:
                    newcode = name2code[name]
                    oldcode_to_newcode[oldcode] = newcode
             
        for idmap in self.crypto_idmaps:
            lbmap = np.array(idmap.m_id_label_map)
            
            # remap code
            if oldcode_to_newcode:
                lbmap = sf.replace_values(lbmap, oldcode_to_newcode, unmatch=unmatched_code)
            output.append(lbmap)
            
        return output

    def close_file(self):
        if self.m_exr_filedata is not None:
            self.m_exr_filedata.close()
        self.m_exr_filedata = None
    
    
    
    
        
    
         
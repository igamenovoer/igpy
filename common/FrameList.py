import numpy as np
import igpy.common.shortfunc as sf
import igpy.common.inout as inout
import igpy.common.image_processing as ip

class FrameList:
    ''' an iterator from which you can get a series of frames
    '''
    def __iter__(self):
        for i in range(self.get_number_of_frames()):
            yield self.get_frame(i)
    
    def __len__(self):
        return self.get_number_of_frames()
    
    def get_frame(self, index) -> np.ndarray:
        ''' get the i-th frame
        '''
        return np.random.randint(0,256,(100,100,3), dtype=np.uint8)
    
    def get_frame_size_hwc(self):
        ''' get frame size in (height,width,channel)
        '''
        return (100,100,3)
    
    def get_number_of_frames(self):
        ''' get the total number of frames
        '''
        return 10  #this is a dummy value
    
class FrameListByDirectory(FrameList):
    def __init__(self):
        self.m_filelist : list = None
        self.m_imgsize_hwc : np.ndarray = None
        
        # the frame sequence is repeated this number of times
        self.m_loop_times : int = 1
        
        # callback functions to process the frame
        # each is a dict containing named functions
        # the function signature is func(frame, index)->np.ndarray
        self.m_func_pre = {}
        self.m_func_post = {}
        
    def set_files(self, filelist:list):
        self.m_filelist = filelist
        img = inout.imread(filelist[0])
        self.m_imgsize_hwc = np.array(img.shape)        
        
    def set_frame_callback_preprocess(self, key, func):
        ''' set a frame preprocess callback, which will be called immediately after the 
        frame is read.
        
        parameters
        -----------------
        key
            the name of the function, must be unique among all functions
        func
            a callback function, func(frame, index)->np.ndarray, where frame is the frame content,
            index is the frame index in the directory. Return the processed frame.
            If func is None, the callback is removed
        '''
        if func is not None:
            self.m_func_pre[key] = func
        elif key in self.m_func_pre:
            self.m_func_pre.pop(key)
        
    def set_frame_callback_postprocess(self, key, func):
        ''' set a frame preprocess callback, which will be called after the frame is processed
        by the frame reader (e.g., resizing)
        
        parameters
        -----------------
        key
            the name of the function, must be unique among all functions
        func
            a callback function, func(frame, index)->np.ndarray, where frame is the frame content,
            index is the frame index in the directory. Return the processed frame.
            If func is None, the callback will be removed
        '''        
        if func is not None:
            self.m_func_post[key] = func
        elif key in self.m_func_post:
            self.m_func_post.pop(key)
        
    def init(self, dirname:str, file_extension:str, ignore_case = False):
        if file_extension[0] == '.':
            file_extension=file_extension[1:]
        
        filelist = sf.listdir(dirname, r'.+\.'+file_extension, ignore_case=ignore_case, include_dir=False, prepend_parent_path=True, sort_by='natural')    
        
        self.set_files(filelist)
        
        #self.m_filelist = filelist
        #img = inout.imread(filelist[0])
        #self.m_imgsize_hwc = np.array(img.shape)
        
    def get_frame_filename(self, index):
        assert self.m_filelist is not None
        
        t = index // len(self.m_filelist)
        assert t < self.m_loop_times, 'index out of bound'
        
        idx = index % len(self.m_filelist)
        fn = self.m_filelist[idx]
        
        return fn     
        
    def _read_raw_frame(self, index) -> np.ndarray:
        ''' read the frame directly without processing
        
        return
        ------------
        img
            the frame content
        idx
            the actual index of the frame
        '''
        assert self.m_filelist is not None
        
        t = index // len(self.m_filelist)
        assert t < self.m_loop_times, 'index out of bound'
        
        idx = index % len(self.m_filelist)
        fn = self.m_filelist[idx]
        img = inout.imread(fn)
        
        return img, idx
    
    def _post_process(self, img):
        ''' process the frame
        '''
        return img

    def get_frame(self, index) -> np.ndarray:
        img, index = self._read_raw_frame(index)
        for key, func in self.m_func_pre.items():
            img = func(img, index)
            
        img = self._post_process(img)
        for key, func in self.m_func_post.items():
            img = func(img, index)
        
        return img
    
    def set_loop_times(self, n):
        ''' number of loops of the frame sequence
        '''
        self.m_loop_times = n
    
    def get_frame_size_hwc(self):
        # get the first frame and refresh the size
        return self.m_imgsize_hwc.copy()
    
    def get_number_of_frames(self):
        if self.m_filelist is None:
            return 0
        else:
            return len(self.m_filelist) * self.m_loop_times
    
class FrameListByDirectoryWithScale(FrameListByDirectory):
    def __init__(self):
        super().__init__()
        self.m_scale : np.ndarray = np.ones(2)
        
    def set_scale(self, x_scale, y_scale = None):
        if y_scale is None:
            y_scale = x_scale
        self.m_scale = np.array([x_scale, y_scale], dtype=float)
        
    def get_frame_size_hwc(self):
        imgsize = super().get_frame_size_hwc()
        newsize = imgsize[:2] * self.m_scale
        imgsize[:2] = newsize.astype(int)
        return imgsize
    
    def _post_process(self, img) -> np.ndarray:
        imgsize = self.get_frame_size_hwc()
        img = ip.imresize(img, imgsize[:2])
        return img
    
class FrameListByDirectoryWithResize(FrameListByDirectory):
    def __init__(self):
        super().__init__()
        self.m_target_size_hw : np.ndarray = None
        
    def set_target_size(self, width, height):
        self.m_target_size_hw = np.array([height, width], dtype=int)
        
    def get_frame_size_hwc(self):
        imgsize = super().get_frame_size_hwc()
        if self.m_target_size_hw is not None:
            imgsize[:2] = self.m_target_size_hw
        return imgsize
    
    def _post_process(self, img):
        imgsize = self.get_frame_size_hwc()[:2]
        img = ip.imresize(img, imgsize)
        return img
    
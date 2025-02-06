import numpy as np
from .SingleFrameData import SingleFrameData, SceneObject
from typing import Tuple
from igpy.common.shortfunc import replace_values
from typing import Protocol
import re
from enum import IntEnum

class SemanticLabelerBase(Protocol):
    ''' responsible for assigning a label to an object based on its position in a hierarchy
    '''
    def assign_label(self, target : SceneObject, parents : list[SceneObject]) -> int:
        ''' assign label to the target based on its own info, and its parents' info
        
        parameters
        ---------------
        target
            the target scene object
        parents
            parents of the target, from nearest to farthest, that is, parents[0] is its direct parent,
            and parents[-1] is the root
            
        return
        ---------
        label_code : int
            the id code of the label, can be None to mean no suitable label
        '''
        raise NotImplementedError()
    
    def get_name_by_code(self, idcode : int) -> str:
        ''' get a string name of the id code. By default, just return the string represenation of idcode
        '''
        return str(idcode)
    
    @property
    def background_label_code(self) -> int:
        raise NotImplementedError()

class SemanticClass:
    ''' information of semantic classes in synthetic data
    '''
    def __init__(self) -> None:
        self.m_code : int = None  # class id code
        self.m_name : str = None  # class name
        
        # compiled pattern string for asset path.
        # an object whose asset name that matches this pattern
        # will be assigned this class
        self.m_asset_name_pattern : re.Pattern = None
            
         # the matching considers case?
        self.m_match_case_sensitive : bool = True  
        
    @property
    def code(self):
        return self.m_code
    
    @property
    def name(self):
        return self.m_name

class SemanticLabelRenderer:
    ''' render semantic label into images
    '''
    def __init__(self) -> None:
        self.m_frame_data : SingleFrameData = None
        self.m_labeler : SemanticLabelerBase = None
        
        # mapping instance code to a new label
        self.m_instcode2label : dict[int,int] = {}
        
    @property
    def code2label(self) -> dict[int,int]:
        ''' mapping instance code to label code
        '''
        return self.m_instcode2label
    
    @property
    def label2objects(self) -> dict[int, SceneObject]:
        ''' get scene objects indexed by labels
        '''
        if not self.m_frame_data:
            return None
        
        code2obj : dict[int, SceneObject] = self.m_frame_data.scene_objects_by_instance_code
        output = {label:[] for label in self.m_instcode2label.values()}
        for code, label in self.m_instcode2label.items():
            output[label].append(code2obj[code])
        return output
    
    @property
    def label2codes(self) -> dict[int, list[int]]:
        ''' mapping label code to a list of instance codes
        '''
        if self.m_instcode2label is None:
            return None
        
        all_labels = np.unique(list(self.m_instcode2label.values()))
        out = {x:[] for x in all_labels}
        for code, label in self.m_instcode2label.items():
            out[label].append(code)
            
        return out
            
    @property
    def label2names(self) -> dict[int, list[str]]:
        ''' mapping label code to a list of instance names
        '''
        if self.m_instcode2label is None:
            return None
        
        all_labels = np.unique(list(self.m_instcode2label.values()))
        out = {x:[] for x in all_labels}
        code2name = {val.instance_code: val.instance_name for val in self.m_frame_data.scene_objects_by_name.values()}
        for code, label in self.m_instcode2label.items():
            out[label].append(code2name[code])
        return out
        
    def init(self, frame_data : SingleFrameData, labeler : SemanticLabelerBase):
        self.m_frame_data = frame_data
        self.m_labeler = labeler
    
    def set_labeler(self, labeler : SemanticLabelerBase):
        self.m_labeler = labeler
        
    def generate_labels(self):
        ''' assign label to objects
        '''
        name2sobj = self.m_frame_data.scene_objects_by_name
        idcode_old2new : dict[int,int] = {}
        
        # collect labels from objects
        for name, obj in name2sobj.items():
            # only for root objects
            if obj.m_parent_name:
                continue
            
            # this object has no id, skip it
            if obj.m_instance_code is None:
                continue
            
            parents = []
            self._collect_label_mapping(obj, parents, out_idcode_old2new=idcode_old2new)
            
        self.m_instcode2label = idcode_old2new
                        
        
    def _collect_label_mapping(self, rootObj : SceneObject, 
                              parents : list[SceneObject],
                              out_idcode_old2new : dict[int,int]):
        ''' starting from root object, traverse its children and collect label mapping.
        The label mapping maps instance code of the scene objects to a new code.
        '''
        
        lb = self.m_labeler
        name2sobj = self.m_frame_data.m_name2sobj
        assert name2sobj is not None, 'instance label cannot be found in frame data'
        
        # label this object
        idcode_this = lb.assign_label(rootObj, parents)
        if idcode_this is not None:
            out_idcode_old2new[rootObj.instance_code] = idcode_this
            
        if not rootObj.m_child_names:
            return
        
        # label the children
        for ch_name in rootObj.m_child_names:
            if ch_name not in name2sobj:
                continue
            
            # collect label mapping for this child
            child_obj = name2sobj[ch_name]
            parents.append(child_obj)
            self._collect_label_mapping(child_obj, parents, out_idcode_old2new=out_idcode_old2new)
            parents.pop()
        
    def get_label_map(self) -> np.ndarray:
        ''' get the semantic label map
        
        return
        -----------
        lbmap : np.ndarray
            lbmap[i,j] is the label for pixel (i,j), which is assigned by the labeler.
            To interpret the label map, one shall consult the labeler
        '''
            
        assert self.m_instcode2label is not None, 'labels are not yet generated, call generate_labels() first'
        # generate label map by value substitution
        instance_map : np.ndarray = self.m_frame_data.get_instance_label_map()
        if self.m_instcode2label:
            lbmap = replace_values(instance_map, self.m_instcode2label, unmatch=self.m_labeler.background_label_code)
        return lbmap
    #end of SemanticLabelRenderer
    
class LabelByAssetPath(SemanticLabelerBase):
    ''' assign label to objects based on patterns of asset path
    '''
    class DefaultCode:
        Background = 0
        FirstClass = 1  # code for the first class
        
    class LabelPrecendenceMode(IntEnum):
        ''' define how to assign a label to an object that has parent
        '''
        Unspecified = 0 # custom mode, purely dictated by assign_label() function
        ParentFirst = 1 # parent label always override children
        ChildFirst = 2  # use child label and do not consider parent
        
        
    def __init__(self) -> None:
        super().__init__()
        self.m_name2class : dict[str, SemanticClass] = {}
        self.m_background_code : int = self.DefaultCode.Background
        self.m_label_precedence_mode : int = self.LabelPrecendenceMode.ParentFirst
        
    @property
    def background_label_code(self) -> int:
        return self.m_background_code
        
    def set_label_precedence_mode(self, mode : int):
        assert mode in [x.value for x in self.LabelPrecendenceMode]
        self.m_label_precedence_mode = mode
        
    def generate_class_code(self) -> int:
        if not self.m_name2class:
            return self.DefaultCode.FirstClass
        
        all_code = [x.code for x in self.m_name2class.values()]
        all_code.append(self.m_background_code)
        return np.max(all_code) + 1
        
    def add_class_definition(self, class_name : str, 
                             asset_name_pattern : str, 
                             class_code:int = None,
                             case_sensitive:bool = True) -> SemanticClass:
        ''' add a class definition
        
        parameters
        -------------
        class_name : str
            the unique class name. If the name already exists, it overrides the previous class definition 
        asset_name_pattern : str
            the regex pattern of the object instances belonging to this class
        class_code : int | None
            the integer code of the class. If None, the code will be automatically chosen
        case_sensitive
            whether the match of object_name_pattern to object name is case sensitive
            
        return
        --------
        result : SemanticClass
            the new class added
        '''
        
        # generate class code
        if class_code is None:
            class_code = self.generate_class_code()
        
        sc : SemanticClass = self.m_name2class.get(class_name)
        if sc is None:
            sc = SemanticClass()
        self.m_name2class[class_name] = sc
        
        # assign attributes
        sc.m_name = class_name
        sc.m_code = class_code
        if case_sensitive:
            sc.m_asset_name_pattern = re.compile(asset_name_pattern)
        else:
            sc.m_asset_name_pattern = re.compile(asset_name_pattern.lower())
        sc.m_match_case_sensitive = case_sensitive
        
        return sc
        
    def assign_label(self, target : SceneObject, parents : list[SceneObject]) -> int:
        output : int = self.m_background_code
        
        # for each class, try to match the asset pattern to the objects in question
        for name, cls in self.m_name2class.items():
            if self.m_label_precedence_mode == self.LabelPrecendenceMode.ChildFirst:
                obj = target
            elif self.m_label_precedence_mode == self.LabelPrecendenceMode.ParentFirst:
                if parents:
                    obj = parents[-1]   # use the root
                else:
                    obj = target  # no parent, use itself
            else:
                assert False, 'this label precedence mode is not supported yet'

            if cls.m_match_case_sensitive:
                match_info = cls.m_asset_name_pattern.match(obj.asset_path)
            else:
                match_info = cls.m_asset_name_pattern.match(obj.asset_path.lower())
                
            if match_info:  # found match
                output = cls.code
                break
            
        return output
            
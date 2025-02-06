import numpy as np
from .UE_Snapshot import UE_SnapshotNode, UE_Snapshot
from typing import Iterable, Callable
from .UE_Labeling import UE_LabelDef
import functools


def generate_labels_by_class_extractor(
    nodes : Iterable[UE_SnapshotNode],
    func_extract_class : Callable[[UE_SnapshotNode], str]) -> dict[str, UE_LabelDef]:
    ''' extract semantic classes from nodes, then organize them into label definitions
    
    parameters
    --------------
    nodes
        a list of snapshot nodes, where semantic class name can be extracted.
    func_extract_class : func(UE_SnapshotNode)->str
        given a node, return the extracted class name. If it does not
        belong to any class, return None
        
    return
    -----------
    class2lbdef : dict[str, UE_LabelDef]
        label definitions indexed by class names
    '''
    pass
    
class UE_LabelDefGenerator:
    ''' generate labels by extracting info from snapshot
    '''
    def __init__(self) -> None:
        self.m_nodes : list[UE_SnapshotNode] = None
        
        # given a node, extract the class name
        self.m_class_extractor : Callable[[UE_SnapshotNode], str] = None
    
    def init(self, nodes : list[UE_SnapshotNode]):
        ''' initialize with all the relevant nodes
        '''
        self.m_nodes = nodes
        
    @staticmethod
    def _match_class_by_func(node : UE_SnapshotNode, target_class : str,
                             func_extract_class : Callable[[UE_SnapshotNode], str],
                             case_sensitive : bool = True) -> bool:
        ''' is the node belong to the target class?
        '''
        cls_name = func_extract_class(node)
        if cls_name is None:
            return False
        
        if case_sensitive:
            return cls_name == target_class
        else:
            return cls_name.lower() == target_class.lower()
    
    def generate_by_class_extractor(self, 
                                    func_extract_class : Callable[[UE_SnapshotNode], str],
                                    case_sensitive : bool = True,
                                    starting_label_code : int = 1) -> dict[str, UE_LabelDef]:
        ''' generate labels by a function that extracts class name from the node
        
        parameters
        ------------
        func_extract_class : func(UE_SnapshotNode)->str
            a function that takes a node and return its class name. If return None, the node has no class
        case_sensitive
            whether the class name matching is case sensitive or not. If True, class names will be in lower case
        starting_label_code
            when generating labels, the label code will be started from this value, and increments 1 for each new label.
            
        return
        ---------
        class2label
            label definitions indexed by class names. The class codes will be 
        '''
        if self.m_nodes is None:
            return None
        
        out : dict[str, UE_LabelDef] = {}
        current_code = starting_label_code
        for node in self.m_nodes:
            cls_name = func_extract_class(node)
            if cls_name is None:
                continue
            
            # already discovered
            if cls_name in out:
                continue
        
            if not case_sensitive:
                cls_name = cls_name.lower()
                
            # create label def
            func_match = functools.partial(__class__.generate_by_class_extractor, 
                                           target_class = cls_name, 
                                           func_extract_class = func_extract_class, 
                                           case_sensitive = case_sensitive)
            lbdef = UE_LabelDef.create_from_match_function(cls_name, current_code, func_match)
            
            current_code += 1
            out[cls_name] = lbdef
        return out
    
    
    
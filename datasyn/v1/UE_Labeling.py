import numpy as np
from . import UE_Snapshot as ue_snp
from . import Labeling
from .Labeling import LabelDef, LabelAssignment
from typing import Callable

class UE_LabelDef(LabelDef):
    def __init__(self) -> None:
        super().__init__()

        # func(node) -> bool, given a node, returns whether it is selected
        self.m_select_func: Callable[[ue_snp.UE_SnapshotNode], bool] = None

    @property
    def select_func(self) -> Callable[[ue_snp.UE_SnapshotNode], bool]:
        return self.m_select_func

    def init(self, name: str, code: int, select_func: Callable[[ue_snp.UE_SnapshotNode], bool]):
        ''' initialize with a unique name, code and a node selector
        '''
        self.m_name = name
        self.m_code = code
        self.m_select_func = select_func
        
    @staticmethod
    def create_from_match_function(name : str, code : int,
                                   matchfunc : Callable[[ue_snp.UE_SnapshotNode], bool]) -> "UE_LabelDef":
        out = UE_LabelDef()
        out.m_select_func = matchfunc
        out.m_name = name
        out.m_code = code
        return out

    @staticmethod
    def create_from_asset_path_pattern(name: str, code: int,
                                       ptn_string: str,
                                       case_sensitive: bool = True,
                                       invert_selection: bool = False) -> "UE_LabelDef":
        ''' create label def with asset path pattern
        '''
        sel = ue_snp.UE_SnapshotNodeSelector()
        desc_sel = ue_snp.ue_desc.UE_DescNodeSelector()
        desc_sel.set_asset_path_pattern(
            ptn_string, inv_select=invert_selection)
        desc_sel.set_case_sensitive(case_sensitive)
        sel.set_desc_selectors([desc_sel])

        out = UE_LabelDef()
        out.init(name, code, sel)
        return out

    @staticmethod
    def create_from_name_pattern(name: str, code: int,
                                 ptn_string: str,
                                 case_sensitive: bool = True,
                                 invert_selection: bool = False) -> "UE_LabelDef":
        ''' create label def with name pattern
        '''
        sel = ue_snp.UE_SnapshotNodeSelector()
        desc_sel = ue_snp.ue_desc.UE_DescNodeSelector()
        desc_sel.set_name_pattern(ptn_string, inv_select=invert_selection)
        desc_sel.set_case_sensitive(case_sensitive)
        sel.set_desc_selectors([desc_sel])

        out = UE_LabelDef()
        out.init(name, code, sel)
        return out

class UE_LabelAssignment(LabelAssignment):
    def __init__(self) -> None:
        super().__init__()
        
    @property
    def assign_by_who(self) -> ue_snp.UE_SnapshotNode:
        return super().assign_by_who
    
####
DefaultBackgroundLabel : UE_LabelDef = UE_LabelDef()
DefaultBackgroundLabel.__dict__.update(Labeling.DefaultBackgroundLabel.__dict__)
import numpy as np
from . import Labeling
from . import UE_Labeling
from .UE_Labeling import UE_LabelAssignment, UE_LabelDef
from . import UE_Snapshot as ue_snp

class UE_LabelBySelector(Labeling.LabelerBase):
    ''' define labels based on node selectors
    '''
    BackgroundLabelName = '__background'
    BackgroundLabelCode = 0

    def __init__(self) -> None:
        super().__init__()

        # mapping label name to label definition
        self.m_name2label: dict[str, UE_LabelDef] = {}
        
        # the snapshot whose nodes will be labeled
        self.m_snapshot : ue_snp.Snapshot = None
        
        # mapping snapshot node to assignments
        self.m_node2assignments : dict[ue_snp.UE_SnapshotNode, list[UE_LabelAssignment]] = {}
        
        # the background label
        self.m_background_label : UE_LabelDef = UE_Labeling.DefaultBackgroundLabel

    @property
    def background_label(self) -> UE_LabelDef:
        return self.m_background_label
    
    def init(self, snapshot : ue_snp.UE_Snapshot):
        self.m_snapshot = snapshot
    
    def add_label_definition(self, lbdef : UE_LabelDef):
        ''' add or replace a label definition (if name exists)
        '''
        assert lbdef.name is not None and lbdef.code is not None
        self.m_name2label[lbdef.name] = lbdef
        
    def _assign_label_to_node(self, label : UE_LabelDef, 
                              node : ue_snp.UE_SnapshotNode, 
                              src_node : ue_snp.UE_SnapshotNode) -> UE_LabelAssignment:
        ''' assign a label to a node, including its chilren if needed. This function will update node2assignment
        '''
        if src_node is None:
            src_node = node
        
        # check if assignment is applicable
        is_assign = label.select_func(node)
        if not is_assign:   # not assignable, skip it
            return None
        
        # label assigned, record it
        asn = UE_LabelAssignment()
        asn.m_label = label
        asn.m_node = node
        asn.m_src_node = src_node
        if node in self.m_node2assignments:
            self.m_node2assignments[node].append(asn)
        else:
            self.m_node2assignments[node] = [asn]
        
        # assign to children as well?
        if label.include_children:
            children = node.get_children(depth=1)
            for child_node in children:
                self._assign_label_to_node(label, child_node, src_node)
        return asn
        
    def parse(self):
        ''' assign labels to snapshot
        '''
        assert self.m_snapshot, 'initialize with snapshot first'
        
        self.m_node2assignments = {}
        
        # assign labels
        all_nodes = self.m_snapshot.get_nodes()
        for node in all_nodes:
            for _, label in self.m_name2label.items():
                self._assign_label_to_node(label, node, node)

    def get_label_assignments(self, node: ue_snp.UE_Snapshot) -> list[UE_LabelAssignment]:
        asnlist = self.m_node2assignments.get(node)
        if not asnlist:
            asn = UE_LabelAssignment()
            asn.m_label = self.background_label
            asn.m_node = node
            asn.m_src_node = None
            asnlist = {asn}
        return asnlist
                    
                    

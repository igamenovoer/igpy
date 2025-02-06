import numpy as np
from typing import Iterable
from . import Labeling
from .Labeling import AssignmentFilter, LabelDef, LabelPreference
from .UE_Labeling import UE_LabelAssignment, UE_LabelDef

from . import UE_Snapshot as ue_snp
from .UE_Snapshot import UE_Snapshot, UE_SnapshotNode

from enum import IntEnum
import igpy.common.shortfunc as sf

class UE_LabelRenderer:
    ''' render label assignments into image
    '''
    
    class OutputKey:
        FilteredLabel = 'selected_label'    # dict[node, LabelDef], selected label according to preference
    
    def __init__(self) -> None:
        self.m_snapshot : UE_Snapshot = None
        self.m_node2assignments : dict[UE_SnapshotNode, list[UE_LabelAssignment]] = None
    
    def init(self, snapshot : UE_Snapshot, label_assignments : dict[UE_Snapshot, list[UE_LabelAssignment]]):
        self.m_snapshot = snapshot
        self.m_node2assignments = label_assignments.copy()
        
    def set_label_assignments(self, assignments : dict[UE_Snapshot, list[UE_LabelAssignment]]):
        self.m_node2assignments = assignments.copy()
        
    def get_label_map(self, label_preference: LabelPreference = None, 
                      background_label : LabelDef = None,
                      additional_output : dict = None) -> np.ndarray:
        ''' get a label map where each pixel is the label code
        
        parameters
        -------------
        label_preference : LabelPreference
            How to select label among multiple assignments of a node.
            By default, we use LabelPreference.ChildFirst
        background_label
            label for pixels not covered by any object. Use the default backround if not specified
        additional_output : dict[str, object]
            additional outputs, indexed by OutputKey, see OutputKey for more
            
        return
        ----------
        label_map
            label_map[i,j] is the label code for pixel (i,j)
        '''
        
        # select labeling
        if label_preference is None:
            label_preference = LabelPreference.ChildFirst
            
        af = AssignmentFilter.create_from_assignments(self.m_node2assignments)
        node2label : dict[UE_SnapshotNode, UE_LabelDef] = None
        
        if label_preference == LabelPreference.ChildFirst:
            node2label = af.get_label_by_leaf()
        elif label_preference == LabelPreference.ParentFirst:
            node2label = af.get_label_by_root()
        elif label_preference == LabelPreference.MaxPriority:
            node2label = af.get_label_by_highest_priority()
        else:
            assert False, 'incorrect label preference'
            
        if additional_output is not None:
            additional_output[self.OutputKey.FilteredLabel] = node2label
            
        if background_label is None:
            background_label = Labeling.DefaultBackgroundLabel
            
        # create label map
        lbmap = np.zeros(self.m_snapshot.image_size_hw, dtype=int)
        lbmap.flat[:] = background_label.code
        for node, label in node2label.items():
            pix = node.get_pixel_indices()
            if pix is not None and len(pix) > 0:
                lbmap.flat[pix] = label.code
                
        return lbmap
    
    def render_label_map(self, label_map : np.ndarray, 
                         label2color3u : dict[int, np.ndarray] = None,
                         with_snapshot_overlay : bool = False,
                         snapshot_gamma : float = 2.2) -> np.ndarray:
        ''' render the label map as rgb image
        
        parameters
        -------------
        label_map
            the label map where label_map[i,j]==k iff pixel (i,j)'s label is k
        label2color3u
            assign a color to each label code, the color must be rgb in uint8
        with_snapshot_overlay
            if True, the return image will be RGBA where A channel is the grayscale image of the snapshot
        snapshot_gamma
            when snapshot overlay is used, this is the gamma value used to process the snapshot image
            
        return
        ----------
        image
            RGB image to represent the label, or RGBA if snapshot overlay is used
        '''
        ulbs, inv_idxmap = np.unique(label_map, return_inverse=True)
        colors : np.ndarray = None  # colors[i] is RGB color for ulbs[i]
        
        if label2color3u is None:
            # generate color map
            import distinctipy
            colors = distinctipy.get_colors(len(ulbs), n_attempts=50)
            colors = (np.array(colors) * 255).astype(np.uint8)
            label2color3u = {lbcode : lbcolor for lbcode, lbcolor in zip(ulbs, colors)}
        else:
            colors = np.zeros((len(ulbs),3), dtype=np.uint8)
            for i, x in enumerate(ulbs):
                colors[i] = label2color3u[x]
        
        label_image : np.ndarray = colors[inv_idxmap].reshape((*label_map.shape,3)).astype(np.uint8)
        if with_snapshot_overlay:
            snapshot_image : np.ndarray = self.m_snapshot.scene_rendering.get_color_image(gamma=snapshot_gamma, rgb_only=True)
            gray = snapshot_image.mean(axis=2).astype(np.uint8)
            label_image = np.dstack((label_image, gray))
        return label_image
            
        
        
    
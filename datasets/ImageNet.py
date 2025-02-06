import scipy.io as sio

def parse_meta_mat(metafile):
    ''' parse the ImageNet meta file to get class info
    
    parameters
    --------------
    metafile
        the meta.mat file in imagenet dataset
        
    return
    --------------
    idx_to_wnid : dict
        mapping class index to wordnet ID
    wnid_to_classes
        mapping wordnet ID to descriptive names
    '''
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes
# graph-related functions
import numpy as np

def find_min_span_tree(pts:np.ndarray) -> np.ndarray:
    ''' find minimum spanning tree to connect a set of spatial points
    
    parameters
    -------------
    pts
        NxD matrix with N D-dimentional points
        
    return
    ------------
    edges
        edges[i]=(u,v) such that pts[u] connects to pts[v]
    '''
    import scipy.sparse as sparse
    import scipy.spatial as spatial
    from .geometry import adjmat_from_mesh_with_distance
    
    obj = spatial.Delaunay(pts)
    tris = obj.simplices
    
    # create the graph
    adjmat = adjmat_from_mesh_with_distance(pts, tris)
    treemat = sparse.csgraph.minimum_spanning_tree(adjmat, overwrite=True)
    ii,jj,v = sparse.find(treemat)
    edges = np.column_stack((ii,jj))
    return edges
    
    
    
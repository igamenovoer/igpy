# construction of 3d mesh models
import numpy as np

def point_cloud_to_cubes(pts:np.ndarray, size:float, return_single_mesh = False):
    ''' for each 3d point, create a cube centered at it
    
    parameters
    -------------
    pts
        Nx3 point list
    size
        edge length of the cube
    return_single_mesh
        if True, all the cubes will be merged into a single mesh. Otherwise,
        return a list of cubes
        
    return
    -------------
    cubes
        a single mesh (vertices, faces) if return_single_mesh==True,
        or a list of meshes [(v,f),...] if return_single_mesh==False
    '''
    pts = np.atleast_2d(pts)
    meshlist = []
    for p in pts:
        v,f = make_cube(size, center=p)
        meshlist.append((v,f))
        
    if return_single_mesh:
        n_vert = 0
        vts = []
        fcs = []
        for i in range(len(meshlist)):
            v,f = meshlist[i]
            vts.append(v)
            fcs.append(f + n_vert)
            n_vert += len(v)
            
        vts = np.row_stack(vts)
        fcs = np.row_stack(fcs)
        return vts,fcs
    else:
        return meshlist

def make_cube(size, center = None):
    ''' create an axis-aligned cube
    
    parameters
    ---------------------
    center
        the position of the center of this cube. If not set, 
        it is equal to size/2. That is, the box's top left corner
        is at the origin.
    size
        the edge length of the cube
        
    return
    -------------------
    vertices
        the nx3 vertices
    faces
        the face array
    '''
    size = np.atleast_1d(size).flatten()
    
    import itertools
    corners = list(itertools.product([0,1], repeat=3))
    corners = np.row_stack(corners).astype(float)
    if center is None:
        corners = corners * size
    else:
        center = np.array(center).flatten()
        corners = (corners - 0.5)*size + center
    
    import scipy.spatial as spatial
    ch = spatial.ConvexHull(corners)
    faces = ch.simplices.astype(int)
    
    import trimesh
    obj = trimesh.Trimesh(corners, faces)
    obj.fix_normals()
    
    return obj.vertices, obj.faces
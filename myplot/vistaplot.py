# functions related to pyvista

import pyvista as pv
import numpy as np

import trimesh
import trimesh.visual

import igpy.common.util_trimesh as u_trimesh
from igpy.geometry.Box_3 import Box_3

class VistaObject:
    def __init__(self) -> None:
        self.m_pvmesh : pv.PolyData = None # the vista mesh
        self.m_mesh_info : u_trimesh.SceneMesh = None # the trimesh object

def read_gltf(fn_gltf_scene : str, 
              with_texture : bool = True,
              exclude_geometry_regex: str = None, 
              include_geometry_regex:str = None,
              verbose : bool = False) -> list[VistaObject]:
    ''' read gltf scene into a list of world-space objects
    
    parameters
    -------------
    fn_gltf_scene
        scene filename
    with_texture
        if the texture 
    exclude_geometry_regex
        exclude the geometry if its name matches this pattern
    include_geometry_regex
        include the geometry only if its name matches this pattern
    verbose
        whether to print some progress info during reading
        
    return
    ----------
    vista_object_list : list[VistaObject]
        a list of objects suitable for pyvista plotting
    '''
    scene : trimesh.Scene = trimesh.load(fn_gltf_scene)
    if verbose:
        print('loading the whole scene with trimesh', flush=True)
    infolist = u_trimesh.flatten_trimesh_scene(scene, 
                                    exclude_geometry_regex=exclude_geometry_regex,
                                    include_geometry_regex=include_geometry_regex)
    texture_store = {}
    output = []
    for idx, mesh_info in enumerate(infolist):
        if verbose:
            print('loading mesh ({ith}/{total}) {name}'.format(ith=idx, total=len(infolist)-1, name=mesh_info.m_instance_name), flush=True)
        tmesh = mesh_info.get_transformed_mesh()
        pvmesh = make_pvmesh_from_trimesh(tmesh=tmesh, with_texture=with_texture, cached_texture=texture_store)
        vobj = VistaObject()
        vobj.m_pvmesh = pvmesh
        vobj.m_mesh_info = mesh_info
        output.append(vobj)
    return output
    
def make_pvmesh_from_trimesh(tmesh : trimesh.Trimesh, 
                             with_texture = True, 
                             cached_texture : dict = None) -> pv.PolyData:
    ''' convert a list a trimesh into pyvista's dataset
    
    parameters
    ----------------
    tmesh
        a list of trimesh
    with_texture
        if True, the texture will be retained in pvmesh, otherwise it is ignored
    cached_texture
        in/out parameter, mapping trimesh material to pv texture, to avoid creating the same texture multiple times

    return
    ----------
    pvmesh
        pyvista.PolyData, if the input mesh has texture, it is stored in pvmesh.textures['albedo']
    '''
    
    # fix something    
    if hasattr(tmesh.visual, 'uv') and tmesh.visual.uv is None:
        cmesh = tmesh.copy()
        cmesh.visual = tmesh.visual.to_color()
        pmesh = pv.wrap(cmesh)
    else:
        pmesh = pv.wrap(tmesh)
    
    if with_texture and hasattr(tmesh.visual, 'material'):
        mtl = tmesh.visual.material
        tm_texture = None
        if isinstance(mtl, trimesh.visual.material.SimpleMaterial):
            tm_texture = mtl.image
        elif isinstance(mtl, trimesh.visual.material.PBRMaterial):
            tm_texture = mtl.baseColorTexture
            
        if tm_texture:
            vtex = None
            if cached_texture is not None:
                vtex = cached_texture.get(mtl)
            
            if vtex is None:
                vtex = pv.numpy_to_texture(np.asarray(tm_texture))
                if cached_texture is not None:
                    cached_texture[mtl] = vtex
            pmesh.textures['albedo'] = vtex
    return pmesh
    
def make_pvmesh_from_trimesh_list(meshlist : list[trimesh.Trimesh], with_texture = True) -> list[pv.DataSet]:
    ''' convert a list of trimesh into pyvista's dataset
    
    parameters
    ----------------
    meshlist
        a list of trimesh
    with_texture
        if True, the texture will be retained in pvmesh, otherwise it is ignored
    
    return
    ----------
    pvmeshlist
        a list of pyvista.DataSet. If the input mesh has texture, it is stored in pvmesh.textures['albedo']
    '''
    texture_store = {}
    output = []
    for obj in meshlist:
        pvmesh = make_pvmesh_from_trimesh(obj, with_texture=with_texture, cached_texture=texture_store)
        output.append(pvmesh)
    # for obj in meshlist:
    #     pmesh = pv.wrap(obj)
    #     if with_texture:
    #         tm_texture = obj.visual.material.baseColorTexture
    #         if tm_texture:
    #             vtex = texture_store.get(obj.visual.material)
    #             if vtex is None:
    #                 vtex = pv.numpy_to_texture(np.asarray(tm_texture))
    #                 texture_store[obj.visual.material] = vtex
    #             pmesh.textures['albedo'] = vtex
    #     output.append(pmesh)
        
    return output

def make_axis(origin, xyz_dirs, axlen = 1.0, thickness=1.0) -> pv.AxesActor:
    ''' create axis actor
    
    parameters
    -------------
    origin
        the origin position of the axis
    xyz_dirs
        x,y,z directions of the axis
    axlen
        length of the axis
    thickness
        the line thickness
        
    return
    ----------
    axis_actor
        the axis actor that can be added to plotter
    '''
    obj = pv.create_axes_marker(line_width=thickness, shaft_length=axlen, labels_off=True)
    tmat = np.eye(4)
    tmat[:3,:3] = xyz_dirs
    tmat[-1,:3] = origin
    tmat_vtk = pv.vtkmatrix_from_array(tmat.T)
    obj.SetUserMatrix(tmat_vtk)
    return obj

def camera_set_transform_by_4x4(cam: pv.Camera, 
                                transmat : np.ndarray, 
                                focal_distance : float = None):
    ''' set camera transform using GLTF 4x4 right-mul transformation matrix
    '''
    R = transmat[:3,:3]
    left, up, forward = R
    pos = transmat[-1,:3]
    camera_set_transform_by_vectors(cam, pos=pos, view_dir=forward, up_dir=up, focal_distance=focal_distance)

def camera_set_transform_by_vectors(cam : pv.Camera, 
                                    pos: np.ndarray, 
                                    view_dir: np.ndarray, 
                                    up_dir: np.ndarray, 
                                    focal_distance = None):
    ''' set world transformation of the camera
    
    parameters
    -------------
    cam
        the target camera
    pos
        world-space position
    view_dir
        world-space view direction
    up_dir
        world-space up direction
    focal_distance
        distance from focal point to camera
    '''
    if focal_distance is None:
        focal_distance = cam.distance
        
    pos = np.array(pos).flatten()
    view_dir = np.array(view_dir) / np.linalg.norm(view_dir)
    
    cam.position = pos
    cam.up = up_dir
    cam.focal_point = pos + view_dir * focal_distance
    
def camera_get_transform(cam : pv.Camera) -> np.ndarray:
    ''' get the 4x4 GLTF right-mul transformation matrix
    '''
    out = np.eye(4)
    dy = np.array(cam.up)
    dz = np.array(cam.GetDirectionOfProjection())
    dx = np.cross(dy, dz)
    dx /= np.linalg.norm(dx)
    out[:3,:3] = (dx, dy, dz)
    out[-1,:3] = cam.position
    return out

def camera_make_transform_by_view_up_pos(view_dir : np.ndarray, 
                                         up_dir : np.ndarray, 
                                         pos: np.ndarray = (0,0,0)) -> np.ndarray:
    ''' create 4x4 right-mul camera world transformation given position, view and up directions.
    The 4x4 right-mul transformation follows GLTF convention (x+ left, y+ up, z+ view)
    '''
    tmat = np.eye(4)
    
    y = np.array(up_dir) / np.linalg.norm(up_dir)
    z = np.array(view_dir) / np.linalg.norm(view_dir)
    x = np.cross(y,z)
    x /= np.linalg.norm(x)
    tmat[:3,:3] = [x,y,z]
    tmat[-1,:3] = pos
    return tmat
    
def camera_apply_transform(cam : pv.Camera, transmat : np.ndarray, relative_to = 'global'):
    ''' apply a 4x4 transformation to the camera
    
    parameters
    -------------
    cam
        the target camera, the local coordinate is x+ right, y+ up, z- view
    transmat
        4x4 transformation matrix in right-multiply format
    relative_to
        'global' or 'local', specify that the transformation is relative to world or local coordinate
    '''
    assert np.allclose(transmat.shape, (4,4))
    assert relative_to in ('global','local')
    T_now = camera_get_transform(cam)
    if relative_to == 'local':
        T_next = transmat @ T_now   # note that we use right-mul transmat, so the order is reverse from std (left-mul transmat)
    elif relative_to == 'global':
        T_next = T_now @ transmat
        
    camera_set_transform_by_4x4(cam, T_next, focal_distance = cam.distance)
        
class ExPlotter:
    ''' extended pyvista plotter with some helper functions
    '''
    def __init__(self) -> None:
        self.m_plotter : pv.BasePlotter = None
        
    @classmethod
    def init_with_background_plotter(cls, with_menu = False, with_toolbar = False, title:str = None, *args, **kwargs):
        from pyvistaqt import BackgroundPlotter
        out = ExPlotter()
        out.m_plotter = BackgroundPlotter(*args, title=title, **kwargs)
        if not with_menu:
            out.m_plotter.main_menu.setVisible(False)
            
        if not with_toolbar:
            out.m_plotter.default_camera_tool_bar.setVisible(False)
            out.m_plotter.saved_cameras_tool_bar.setVisible(False)
            
        out.m_plotter.show_axes()   # by default, show axes
        return out
     
    @classmethod   
    def init_with_std_plotter(cls, title : str = None, *args, **kwargs):
        out = ExPlotter()
        out.m_plotter = pv.Plotter(*args, notebook=False, title = title, **kwargs)
        out.m_plotter.show_axes()
        return out
        
    def set_plotter(self, plt : pv.BasePlotter):
        self.m_plotter = plt
        
    def save_image(self, outfilename: str):
        self.m_plotter.screenshot(outfilename)
        
    def get_image(self) -> np.ndarray:
        return self.m_plotter.screenshot()
        
    def set_image_size(self, height : int, width : int):
        self.m_plotter.window_size = [width, height]
        self.m_plotter.setGeometry(0,0,width,height)
        
    def add_mesh_with_trimesh_object(self, 
                                     mesh_object : trimesh.Trimesh, 
                                     with_texture = False,
                                     color3f : np.ndarray = None,
                                     show_edges : bool = False,
                                     edge_color3f : np.ndarray = None,
                                     line_width : float = None,
                                     style:str = None,
                                     **kwargs) -> pv.Actor:
        ''' add mesh by providing trimesh object
        
        parameters
        -------------
        mesh_object
            the trimesh object to be added
        with_texture
            whether to show texture of the mesh
        color3f: np.ndarray
            the color of the mesh
        show_edges: bool
            whether to show edges
        edge_color3f: np.ndarray
            the color of the edges
        line_width: float
            the width of the edges
        style : 'surface'|'wireframe'|'points'
            how to draw the mesh
        **kwargs
            the additional parameters passed to self.m_plotter.add_mesh()
            
        return
        ---------
        actor
            the pv actor of the mesh
        '''
        pvmesh = make_pvmesh_from_trimesh(mesh_object, with_texture=with_texture)
        
        if color3f is not None:
            color3f = np.array(color3f, dtype=float).tolist()
        if edge_color3f is not None:
            edge_color3f = np.array(edge_color3f, dtype=float).tolist()
        
        actor = self.m_plotter.add_mesh(pvmesh, 
                                        color=color3f, 
                                        edge_color=edge_color3f,
                                        show_edges=show_edges,
                                        style=style,
                                        line_width=line_width,
                                        **kwargs)
        return actor
        
    def add_polyline(self, pts : np.ndarray, color3f = None, 
                     line_width = None, show_marker = False, 
                     marker_color3f = (0,0,0), marker_size = 5.0) -> pv.Actor:
        ''' draw a polyline defined by a list of points
        '''
        pts = np.atleast_2d(pts)
        pvdata = pv.lines_from_points(pts)
        if color3f is not None:
            color3f = np.array(color3f, dtype=float)
            
        marker_color3f = np.array(marker_color3f, dtype=float)
        out = self.m_plotter.add_mesh(pvdata,
                                      color=color3f,
                                      show_edges=True,
                                      line_width=line_width,
                                      show_vertices=show_marker,
                                      vertex_color=marker_color3f,
                                      point_size=marker_size)
        # self.plotter.add_composite()
        
        return out
    
    def add_volume_as_occupancy(self, vol : np.ndarray, 
                                color3f : np.ndarray = None, 
                                opacity = 0.5, **kwargs) -> pv.Actor:
        ''' add a solid volume to display, where the volume is represented as a boolean array (or convertable to boolean array).
        In the volume, True means the voxel is occupied and False means the voxel is empty.
        
        parameters
        --------------
        vol : (N,M,P) bool
            the volume to be displayed
        color3f : (3,)
            the color of the volume
        opacity : float
            the opacity of the volume
        **kwargs
            additional parameters passed to self.m_plotter.add_volume()
        '''
        import einops
        vol = np.atleast_3d(vol)
        
        if color3f is None:
            color3f = np.array([0,0,1], dtype=float)
            
        if vol.dtype != bool:
            vol = vol.astype(bool)
            
        scalars = np.zeros((*vol.shape, 4), dtype=np.uint8)
        
        # HACK: must have something visible to display for empty space, otherwise some voxel will be missing in plot
        # weird behaviour in pyvista
        scalars[...,:] = (0,0,0,1)
        
        # set colors
        scalars[vol, :] = (np.array([*color3f, opacity]) * 255).astype(np.uint8)
        scalars = einops.rearrange(scalars, 'x y z c -> (z y x) c')
        out = self.m_plotter.add_volume(vol, scalars=scalars, **kwargs)
        return out
    
    def add_volume_as_label_map(self, vol : np.ndarray, color4f : np.ndarray = None, 
                                opacity : float = None, **kwargs) -> pv.Actor:
        ''' add a solid volume to display, where the volume is represented as a label map.
        Each voxel is assigned a label, and the label is used to determine the color of the voxel.
        
        parameters
        -------------
        vol : (N,M,P) int
            the volume to be displayed
        color4f : (K,4)
            the color of each label, color4f[k] is the color of label k. The color is in rgba format, range [0,1].
            Note that the background color is also specified by this parameter. 
            If None, the colors are automatically generated, assuming 0=background.
            Note that alpha=0 is not allowed, it will be changed to alpha=1.
        opacity : float
            the opacity of the volume, if specified, it will override the alpha channel of color4f
        '''
        assert vol.ndim == 3 and vol.dtype == int, 'only support 3d int-type label map'
        
        import einops
        vol = np.atleast_3d(vol)
        min_alpha = 1
        
        if color4f is None:
            import distinctipy
            idcode, idmap = np.unique(vol, return_inverse=True)
            idmap = idmap.reshape(vol.shape)
            colors = distinctipy.get_colors(len(idcode), n_attempts=100)
            color4u = np.zeros((len(idcode), 4), dtype=np.uint8)
            color4u[:,:3] = (np.array(colors) * 255).astype(np.uint8)
            if opacity is None:
                color4u[:,-1] = 255
            else:
                color4u[:,-1] = np.maximum(min_alpha, opacity * 255).astype(np.uint8)
            color4u[0] = (0,0,0,min_alpha)
            scalars = color4u[idmap]
        else:
            color4u = (np.array(color4f) * 255).astype(np.uint8)
            if opacity is not None:
                color4u[:,-1] = np.maximum(min_alpha, opacity * 255).astype(np.uint8)
            color4u[:,-1] = np.maximum(color4u[:,-1], min_alpha)
            scalars = color4u[vol]
        
        scalars = einops.rearrange(scalars, 'x y z c -> (z y x) c')
        out = self.m_plotter.add_volume(vol, scalars=scalars, **kwargs)
        return out
        
    def add_volume_as_scalar_field(self, vol : np.ndarray, cmap : str = None, **kwargs) -> pv.Actor:
        ''' add a solid volume to display, where the volume is represented as a scalar field.
        
        parameters
        ------------
        vol : (N,M,P) float
            the volume to be displayed
        cmap : str
            the colormap to be used, in matplotlib format
        **kwargs
            additional parameters passed to self.m_plotter.add_volume()
        '''
        if vol.dtype not in (float, np.float32):
            vol = vol.astype(float)
            
        return self.m_plotter.add_volume(vol, cmap=cmap, **kwargs)
        
    def add_ground_plane(self, x_size, y_size, 
                         up_vector = (0,0,1),
                         x_seg = 20, y_seg = 20, 
                         color3f=(0,0,0), line_width = None) -> pv.Actor:
        ''' add a ground plane represented as a grid
        '''
        plane = pv.Plane(direction = up_vector, i_size = x_size, j_size = y_size, i_resolution=x_seg, j_resolution=y_seg)
        actor = self.m_plotter.add_mesh(plane, style='wireframe',color=color3f, line_width=line_width)
        return actor
    
    def add_axes(self, origin, xyz_dirs, axis_length = 1.0, 
                 line_width = None) -> pv.AxesActor:
        if line_width is None:
            line_width = 1.0
            
        # xyz_dirs = np.atleast_2d(xyz_dirs)
        # origin = np.array(origin).flatten()
        
        # pts1 = np.row_stack([origin]*3)
        # pts2 = pts1 + xyz_dirs * axis_length
        # pts_draw = np.column_stack([pts1, pts2]).reshape(-1,3)
        # axis_actor = self.m_plotter.add_lines(pts_draw, color=np.eye(3))
        
        axis_actor = make_axis(origin, xyz_dirs, axlen=axis_length, thickness=line_width)
        axis_actor.SetConeResolution(4)
        axis_actor.SetConeRadius(0.01)
        self.m_plotter.add_actor(axis_actor)
        return axis_actor
    
    def add_axes_many(self, origins : np.ndarray, xyz_dirs : np.ndarray, 
                      axis_length=1.0, line_width=None):
        ''' add many axes as line segments to the plot
        
        parameters
        --------------
        origins : Nx3
            origins[i] is the origin of the i-ith axes
        xyz_dirs: Nx3x3
            xyz_dirs[i] = (x,y,z) axis directions
        '''
        if line_width is None:
            line_width = 1.0
            
        origins = np.atleast_2d(origins)
        xyz_dirs = np.atleast_3d(xyz_dirs)
        
        colors = np.eye(3)
        for i in (0,1,2):
            pts_1 = origins
            pts_2 = xyz_dirs[:,i,:] * axis_length + pts_1
            self.add_line_segments(pts_1, pts_2, color3f = colors[i], line_width = line_width)
            
    
    def add_box(self, box : Box_3, color3f : np.ndarray = None, 
                line_width : float = None) -> pv.Actor:
        ''' draw a box
        '''
        if color3f is None:
            color3f = np.ones(3, dtype=float)
        if line_width is None:
            line_width = 1.0
            
        color3f = np.array(color3f).astype(float)
        pts = box.get_points()
        u,v = box.get_edges().T
        obj = self.add_line_segments(pts[u], pts[v], color3f=color3f, line_width=line_width)
        return obj
    
    def add_line_segments(self, pts1 : np.ndarray, pts2 : np.ndarray, 
                          color3f : np.ndarray = None, line_width : float =None) -> pv.Actor:
        ''' add lines connecting pts1[k] to pts2[k]
        '''
        if color3f is None:
            color3f = np.ones(3, dtype=float)
        if line_width is None:
            line_width = 1.0
            
        color3f = np.array(color3f).astype(float)
        pts1 = np.atleast_2d(pts1)
        pts2 = np.atleast_2d(pts2)
        pts_input = np.column_stack((pts1,pts2)).reshape((-1,3))
        obj = self.m_plotter.add_lines(pts_input, color=color3f, width=line_width)
        return obj
    
    def add_points(self, pts : np.ndarray, 
                   color3f : np.ndarray = None, 
                   style : str = 'points',
                   **kwargs) -> pv.Actor:
        ''' add points to the plot
        
        parameters
        --------------
        pts : (N,3)
            N points
        color3f : (3,)
            a single color for all points
        style : 'points' | 'points_gaussian' | 'sphere'
            the geometry that represents the points
        '''
        _style : str = None
        _render_points_as_spheres : bool = False
        if style.lower() in ('points','points_gaussian'):
            _style = style
        elif style.lower() == 'sphere':
            _style = 'points'
            _render_points_as_spheres = True
        
        if color3f is None:
            color3f = np.ones(3)
        else:
            color3f = np.atleast_1d(color3f).astype(float)
            if color3f.size == 3:
                color3f = np.atleast_1d(color3f).flatten().astype(float)
            else:
                assert len(color3f) == len(pts), f'number of points should match number of given colors'

        if color3f.size == 3:        
            actor = self.m_plotter.add_points(pts, style = _style, 
                                render_points_as_spheres = _render_points_as_spheres,
                                color = color3f,
                                emissive = True, **kwargs)
        else:
            actor = self.m_plotter.add_points(pts, style = _style,
                                              render_points_as_spheres = _render_points_as_spheres,
                                              scalars = color3f,
                                              rgb = True)

        return actor
    
    def set_camera_transform_by_vectors(self, view_dir : np.ndarray, up_dir : np.ndarray, position : np.ndarray, focal_distance = None):
        camera_set_transform_by_vectors(self.m_plotter.camera, pos=position, view_dir=view_dir, 
                                        up_dir=up_dir, focal_distance=focal_distance)
        self.m_plotter.camera.Modified()
        
    def set_camera_transform_by_4x4(self, gltf_transmat : np.ndarray):
        camera_set_transform_by_4x4(self.m_plotter.camera, gltf_transmat)
        self.m_plotter.camera.Modified()       
        
    def set_camera_vertical_fov(self, fovy_rad : float):
        self.m_plotter.camera.view_angle = np.rad2deg(fovy_rad)
    
    def show(self):
        self.m_plotter.show()
        
    @property
    def plotter(self)->pv.BasePlotter:
        return self.m_plotter
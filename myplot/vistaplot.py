# functions related to pyvista

import pyvista as pv
import numpy as np

import trimesh
import trimesh.visual

import igpy.common.util_trimesh as u_trimesh
from igpy.geometry.Box_3 import Box_3


class VistaObject:
    def __init__(self) -> None:
        self.m_pvmesh: pv.PolyData = None  # the vista mesh
        self.m_mesh_info: u_trimesh.SceneMesh = None  # the trimesh object


def read_gltf(
    fn_gltf_scene: str,
    with_texture: bool = True,
    exclude_geometry_regex: str = None,
    include_geometry_regex: str = None,
    verbose: bool = False,
) -> list[VistaObject]:
    """read gltf scene into a list of world-space objects

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
    """
    scene: trimesh.Scene = trimesh.load(fn_gltf_scene)
    if verbose:
        print("loading the whole scene with trimesh", flush=True)
    infolist = u_trimesh.flatten_trimesh_scene(
        scene,
        exclude_geometry_regex=exclude_geometry_regex,
        include_geometry_regex=include_geometry_regex,
    )
    texture_store = {}
    output = []
    for idx, mesh_info in enumerate(infolist):
        if verbose:
            print(
                "loading mesh ({ith}/{total}) {name}".format(
                    ith=idx,
                    total=len(infolist) - 1,
                    name=mesh_info.m_instance_name,
                ),
                flush=True,
            )
        tmesh = mesh_info.get_transformed_mesh()
        pvmesh = make_pvmesh_from_trimesh(
            tmesh=tmesh, with_texture=with_texture, cached_texture=texture_store
        )
        vobj = VistaObject()
        vobj.m_pvmesh = pvmesh
        vobj.m_mesh_info = mesh_info
        output.append(vobj)
    return output


def make_pvmesh_from_trimesh(
    tmesh: trimesh.Trimesh, with_texture=True, cached_texture: dict = None
) -> pv.PolyData:
    """convert a list a trimesh into pyvista's dataset

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
    """

    # fix something
    if hasattr(tmesh.visual, "uv") and tmesh.visual.uv is None:
        cmesh = tmesh.copy()
        cmesh.visual = tmesh.visual.to_color()
        pmesh = pv.wrap(cmesh)
    else:
        pmesh = pv.wrap(tmesh)

    if with_texture and hasattr(tmesh.visual, "material"):
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
            pmesh.textures["albedo"] = vtex
    return pmesh


def make_pvmesh_from_trimesh_list(
    meshlist: list[trimesh.Trimesh], with_texture=True
) -> list[pv.DataSet]:
    """convert a list of trimesh into pyvista's dataset

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
    """
    texture_store = {}
    output = []
    for obj in meshlist:
        pvmesh = make_pvmesh_from_trimesh(
            obj, with_texture=with_texture, cached_texture=texture_store
        )
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


def make_axis(origin, xyz_dirs, axlen=1.0, thickness=1.0) -> pv.AxesActor:
    """create axis actor

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
    """
    obj = pv.create_axes_marker(
        line_width=thickness, shaft_length=axlen, labels_off=True
    )
    tmat = np.eye(4)
    tmat[:3, :3] = xyz_dirs
    tmat[-1, :3] = origin
    tmat_vtk = pv.vtkmatrix_from_array(tmat.T)
    obj.SetUserMatrix(tmat_vtk)
    return obj


def camera_set_transform_by_4x4(
    cam: pv.Camera, transmat: np.ndarray, focal_distance: float = None
):
    """set camera transform using GLTF 4x4 right-mul transformation matrix"""
    R = transmat[:3, :3]
    left, up, forward = R
    pos = transmat[-1, :3]
    camera_set_transform_by_vectors(
        cam, pos=pos, view_dir=forward, up_dir=up, focal_distance=focal_distance
    )


def camera_set_transform_by_vectors(
    cam: pv.Camera,
    pos: np.ndarray,
    view_dir: np.ndarray,
    up_dir: np.ndarray,
    focal_distance=None,
):
    """set world transformation of the camera

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
    """
    if focal_distance is None:
        focal_distance = cam.distance

    pos = np.array(pos).flatten()
    view_dir = np.array(view_dir) / np.linalg.norm(view_dir)

    cam.position = pos
    cam.up = up_dir
    cam.focal_point = pos + view_dir * focal_distance


def camera_get_transform(cam: pv.Camera) -> np.ndarray:
    """get the 4x4 GLTF right-mul transformation matrix"""
    out = np.eye(4)
    dy = np.array(cam.up)
    dz = np.array(cam.GetDirectionOfProjection())
    dx = np.cross(dy, dz)
    dx /= np.linalg.norm(dx)
    out[:3, :3] = (dx, dy, dz)
    out[-1, :3] = cam.position
    return out


def camera_make_transform_by_view_up_pos(
    view_dir: np.ndarray, up_dir: np.ndarray, pos: np.ndarray = (0, 0, 0)
) -> np.ndarray:
    """create 4x4 right-mul camera world transformation given position, view and up directions.
    The 4x4 right-mul transformation follows GLTF convention (x+ left, y+ up, z+ view)
    """
    tmat = np.eye(4)

    y = np.array(up_dir) / np.linalg.norm(up_dir)
    z = np.array(view_dir) / np.linalg.norm(view_dir)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    tmat[:3, :3] = [x, y, z]
    tmat[-1, :3] = pos
    return tmat


def camera_apply_transform(
    cam: pv.Camera, transmat: np.ndarray, relative_to="global"
):
    """apply a 4x4 transformation to the camera

    parameters
    -------------
    cam
        the target camera, the local coordinate is x+ right, y+ up, z- view
    transmat
        4x4 transformation matrix in right-multiply format
    relative_to
        'global' or 'local', specify that the transformation is relative to world or local coordinate
    """
    assert np.allclose(transmat.shape, (4, 4))
    assert relative_to in ("global", "local")
    T_now = camera_get_transform(cam)
    if relative_to == "local":
        T_next = (
            transmat @ T_now
        )  # note that we use right-mul transmat, so the order is reverse from std (left-mul transmat)
    elif relative_to == "global":
        T_next = T_now @ transmat

    camera_set_transform_by_4x4(cam, T_next, focal_distance=cam.distance)


class ExPlotter:
    """extended pyvista plotter with some helper functions"""

    DefaultBackgroundColor = np.array(
        [0.8, 0.8, 0.8]
    )  # Default background color
    ArrowHeadSizeRelative = (
        0.7  # Default arrow head size relative to arrow length
    )
    ArrowShaftRadiusRelative = (
        0.015  # Default arrow shaft radius relative to arrow length
    )

    def __init__(self) -> None:
        self.m_plotter: pv.BasePlotter = None

    @classmethod
    def init_with_background_plotter(
        cls,
        with_menu=False,
        with_toolbar=False,
        title: str = None,
        background_color3f=None,  # Renamed parameter
        *args,
        **kwargs,
    ):
        from pyvistaqt import BackgroundPlotter

        if background_color3f is None:
            background_color3f = cls.DefaultBackgroundColor

        out = ExPlotter()
        out.m_plotter = BackgroundPlotter(*args, title=title, **kwargs)
        if not with_menu:
            out.m_plotter.main_menu.setVisible(False)

        if not with_toolbar:
            out.m_plotter.default_camera_tool_bar.setVisible(False)
            out.m_plotter.saved_cameras_tool_bar.setVisible(False)

        out.m_plotter.set_background(background_color3f)

        out.m_plotter.show_axes()  # by default, show axes
        return out

    @classmethod
    def init_with_std_plotter(
        cls,
        title: str = None,
        background_color3f=None,
        *args,
        **kwargs,  # Renamed parameter
    ):
        out = ExPlotter()
        out.m_plotter = pv.Plotter(*args, notebook=False, title=title, **kwargs)

        if background_color3f is None:
            background_color3f = cls.DefaultBackgroundColor
        out.m_plotter.set_background(background_color3f)

        out.m_plotter.show_axes()
        return out

    def set_plotter(self, plt: pv.BasePlotter):
        self.m_plotter = plt

    def save_image(self, outfilename: str):
        self.m_plotter.screenshot(outfilename)

    def get_image(self) -> np.ndarray:
        return self.m_plotter.screenshot()

    def set_background_color(
        self, color3f, top=None, right=None, all_renderers=True
    ):  # Renamed parameter
        """Set the background color of the plotter.

        Parameters
        ----------
        color3f : str or sequence
            Color of the background. Ideally a sequence of 3 float RGB values
            (e.g., [0.0, 0.5, 1.0], range 0-1).
            PyVista also accepts string names (e.g., 'white') or hex strings.
        top : str or sequence, optional
            Color of the background at the top. If specified, a gradient is created.
        right : str or sequence, optional
            Color of the background at the right. If specified, a gradient is created.
        all_renderers : bool, optional
            If True (default), applies the background color to all renderers.
        """
        if self.m_plotter:
            self.m_plotter.set_background(
                color3f, top=top, right=right, all_renderers=all_renderers
            )  # Use renamed parameter

    def set_image_size(self, height: int, width: int):
        self.m_plotter.window_size = [width, height]
        self.m_plotter.setGeometry(0, 0, width, height)

    def add_mesh_with_trimesh_object(
        self,
        mesh_object: trimesh.Trimesh,
        with_texture=False,
        color3f: np.ndarray = None,
        show_edges: bool = False,
        edge_color3f: np.ndarray = None,
        line_width: float = None,
        style: str = None,
        shading: str = None,
        metallic: float = None,
        roughness: float = None,
        **kwargs,
    ) -> pv.Actor:
        """add mesh by providing trimesh object

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
        shading : str, optional
            Shading style for the mesh. Options are 'flat', 'smooth', 'pbr', or 'albedo'.
            If None, PyVista's default shading is used.
            - 'flat': Flat shading.
            - 'smooth': Smooth shading (Gouraud).
            - 'pbr': Physically Based Rendering. Uses `metallic` and `roughness` parameters.
            - 'albedo': Disables lighting for the mesh, showing only its base color/texture at full brightness.
        metallic : float, optional
            Metallic property for PBR (0.0 to 1.0). Used if shading='pbr'.
        roughness : float, optional
            Roughness property for PBR (0.0 to 1.0). Used if shading='pbr'.
        **kwargs
            the additional parameters passed to self.m_plotter.add_mesh()

        return
        ---------
        actor
            the pv actor of the mesh
        """
        pvmesh = make_pvmesh_from_trimesh(
            mesh_object, with_texture=with_texture
        )

        if color3f is not None:
            color3f = np.array(color3f, dtype=float).tolist()
        if edge_color3f is not None:
            edge_color3f = np.array(edge_color3f, dtype=float).tolist()

        # Determine PyVista shading parameters and lighting
        pv_smooth_shading = None
        pv_pbr = None
        processed_kwargs = kwargs.copy()

        if shading is not None:
            shading_lower = shading.lower()
            if shading_lower == "albedo":
                processed_kwargs["lighting"] = False
                pv_smooth_shading = (
                    False  # Unlit, so no real smoothing effect from light
                )
                pv_pbr = False
            elif shading_lower == "flat":
                pv_smooth_shading = False
                pv_pbr = False
                if (
                    "lighting" not in kwargs
                ):  # Default to lit for this style if not specified
                    processed_kwargs["lighting"] = True
            elif shading_lower == "smooth":
                pv_smooth_shading = True
                pv_pbr = False
                if (
                    "lighting" not in kwargs
                ):  # Default to lit for this style if not specified
                    processed_kwargs["lighting"] = True
            elif shading_lower == "pbr":
                pv_smooth_shading = True  # PBR typically implies smooth shading
                pv_pbr = True
                if "lighting" not in kwargs:  # PBR implies lighting
                    processed_kwargs["lighting"] = True
            # Else (unrecognized shading string), pv_smooth_shading and pv_pbr remain None.
            # Lighting will be from original kwargs or PyVista's add_mesh default (which is lighting=False).

        actor = self.m_plotter.add_mesh(
            pvmesh,
            color=color3f,
            edge_color=edge_color3f,
            show_edges=show_edges,
            style=style,
            line_width=line_width,
            smooth_shading=pv_smooth_shading,
            pbr=pv_pbr,
            metallic=metallic if pv_pbr else None,
            roughness=roughness if pv_pbr else None,
            **processed_kwargs,
        )
        return actor

    def add_polyline(
        self,
        pts: np.ndarray,
        color3f=None,
        line_width=None,
        show_marker=False,
        marker_color3f=(0, 0, 0),
        marker_size=5.0,
    ) -> pv.Actor:
        """draw a polyline defined by a list of points"""
        pts = np.atleast_2d(pts)
        pvdata = pv.lines_from_points(pts)
        if color3f is not None:
            color3f = np.array(color3f, dtype=float)

        marker_color3f = np.array(marker_color3f, dtype=float)
        out = self.m_plotter.add_mesh(
            pvdata,
            color=color3f,
            show_edges=True,
            line_width=line_width,
            show_vertices=show_marker,
            vertex_color=marker_color3f,
            point_size=marker_size,
        )
        # self.plotter.add_composite()

        return out

    def add_volume_as_occupancy(
        self, vol: np.ndarray, color3f: np.ndarray = None, opacity=0.5, **kwargs
    ) -> pv.Actor:
        """add a solid volume to display, where the volume is represented as a boolean array (or convertable to boolean array).
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
        """
        import einops

        vol = np.atleast_3d(vol)

        if color3f is None:
            color3f = np.array([0, 0, 1], dtype=float)

        if vol.dtype != bool:
            vol = vol.astype(bool)

        scalars = np.zeros((*vol.shape, 4), dtype=np.uint8)

        # HACK: must have something visible to display for empty space, otherwise some voxel will be missing in plot
        # weird behaviour in pyvista
        scalars[..., :] = (0, 0, 0, 1)

        # set colors
        scalars[vol, :] = (np.array([*color3f, opacity]) * 255).astype(np.uint8)
        scalars = einops.rearrange(scalars, "x y z c -> (z y x) c")
        out = self.m_plotter.add_volume(vol, scalars=scalars, **kwargs)
        return out

    def add_volume_as_label_map(
        self,
        vol: np.ndarray,
        color4f: np.ndarray = None,
        opacity: float = None,
        **kwargs,
    ) -> pv.Actor:
        """add a solid volume to display, where the volume is represented as a label map.
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
        """
        assert (
            vol.ndim == 3 and vol.dtype == int
        ), "only support 3d int-type label map"

        import einops

        vol = np.atleast_3d(vol)
        min_alpha = 1

        if color4f is None:
            import distinctipy

            idcode, idmap = np.unique(vol, return_inverse=True)
            idmap = idmap.reshape(vol.shape)
            colors = distinctipy.get_colors(len(idcode), n_attempts=100)
            color4u = np.zeros((len(idcode), 4), dtype=np.uint8)
            color4u[:, :3] = (np.array(colors) * 255).astype(np.uint8)
            if opacity is None:
                color4u[:, -1] = 255
            else:
                color4u[:, -1] = np.maximum(min_alpha, opacity * 255).astype(
                    np.uint8
                )
            color4u[0] = (0, 0, 0, min_alpha)
            scalars = color4u[idmap]
        else:
            color4u = (np.array(color4f) * 255).astype(np.uint8)
            if opacity is not None:
                color4u[:, -1] = np.maximum(min_alpha, opacity * 255).astype(
                    np.uint8
                )
            color4u[:, -1] = np.maximum(color4u[:, -1], min_alpha)
            scalars = color4u[vol]

        scalars = einops.rearrange(scalars, "x y z c -> (z y x) c")
        out = self.m_plotter.add_volume(vol, scalars=scalars, **kwargs)
        return out

    def add_volume_as_scalar_field(
        self, vol: np.ndarray, cmap: str = None, **kwargs
    ) -> pv.Actor:
        """add a solid volume to display, where the volume is represented as a scalar field.

        parameters
        ------------
        vol : (N,M,P) float
            the volume to be displayed
        cmap : str
            the colormap to be used, in matplotlib format
        **kwargs
            additional parameters passed to self.m_plotter.add_volume()
        """
        if vol.dtype not in (float, np.float32):
            vol = vol.astype(float)

        return self.m_plotter.add_volume(vol, cmap=cmap, **kwargs)

    def add_ground_plane(
        self,
        x_size,
        y_size,
        up_vector=(0, 0, 1),
        x_seg=20,
        y_seg=20,
        color3f=(0, 0, 0),
        line_width=None,
    ) -> pv.Actor:
        """add a ground plane represented as a grid"""
        plane = pv.Plane(
            direction=up_vector,
            i_size=x_size,
            j_size=y_size,
            i_resolution=x_seg,
            j_resolution=y_seg,
        )
        actor = self.m_plotter.add_mesh(
            plane, style="wireframe", color=color3f, line_width=line_width
        )
        return actor

    def add_axes(
        self, origin, xyz_dirs, axis_length=1.0, line_width=None
    ) -> pv.AxesActor:
        if line_width is None:
            line_width = 1.0
        return self.add_axes_many(
            [np.atleast_2d(origin)],
            [np.atleast_2d(xyz_dirs)],
            axis_length=axis_length,
            line_width=line_width,
        )

        # xyz_dirs = np.atleast_2d(xyz_dirs)
        # origin = np.array(origin).flatten()

        # pts1 = np.row_stack([origin]*3)
        # pts2 = pts1 + xyz_dirs * axis_length
        # pts_draw = np.column_stack([pts1, pts2]).reshape(-1,3)
        # axis_actor = self.m_plotter.add_lines(pts_draw, color=np.eye(3))

        # axis_actor = make_axis(
        #     origin, xyz_dirs, axlen=axis_length, thickness=line_width
        # )
        # axis_actor.SetConeResolution(4)
        # axis_actor.SetConeRadius(0.01)
        # self.m_plotter.add_actor(axis_actor)
        # return axis_actor

    def add_axes_many(
        self,
        origins: np.ndarray,
        xyz_dirs: np.ndarray,
        axis_length=1.0,
        line_width=None,
    ):
        """add many axes as line segments to the plot

        parameters
        --------------
        origins : Nx3
            origins[i] is the origin of the i-ith axes
        xyz_dirs: Nx3x3
            xyz_dirs[i] = (x,y,z) axis directions
        """
        if line_width is None:
            line_width = 1.0

        origins = np.atleast_2d(origins)
        xyz_dirs = np.atleast_3d(xyz_dirs)

        colors = np.eye(3)
        for i in (0, 1, 2):
            pts_1 = origins
            pts_2 = xyz_dirs[:, i, :] * axis_length + pts_1
            self.add_line_segments(
                pts_1, pts_2, color3f=colors[i], line_width=line_width
            )

    def add_box(
        self, box: Box_3, color3f: np.ndarray = None, line_width: float = None
    ) -> pv.Actor:
        """draw a box"""
        if color3f is None:
            color3f = np.ones(3, dtype=float)
        if line_width is None:
            line_width = 1.0

        color3f = np.array(color3f).astype(float)
        pts = box.get_points()
        u, v = box.get_edges().T
        obj = self.add_line_segments(
            pts[u], pts[v], color3f=color3f, line_width=line_width
        )
        return obj

    def add_line_segments(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        color3f: np.ndarray = None,
        line_width: float | None = None,
        with_arrow: bool = False,
        arrow_head_size: float | None = None,
        shading: str = None,
        metallic: float = None,
        roughness: float = None,
        **kwargs,
    ) -> pv.Actor:
        """add lines connecting pts1[k] to pts2[k], or arrows if specified.

        parameters
        --------------
        pts1 : np.ndarray
            (N,3) array of starting points.
        pts2 : np.ndarray
            (N,3) array of ending points.
        color3f : np.ndarray, optional
            Color for the lines/arrows.
            - If None (default): uses white [1.0, 1.0, 1.0].
            - If a (3,) array: specifies a single RGB color (0.0-1.0) for all segments/arrows.
            - If an (N,3) array: specifies an RGB color (0.0-1.0) for each of the N segments/arrows.
        line_width : float, optional
            Controls the thickness:
            - For simple lines (`with_arrow=False`): Sets the line width. Defaults to 1.0 if not specified.
            - For arrows (`with_arrow=True`): Sets the **radius of the arrow shaft**.
              If not specified, automatically computed based on the spatial extent of the points.
            This parameter also sets the line width for arrow edges if they are shown (e.g., via `style='wireframe'` or `show_edges=True` in `kwargs`); PyVista's default is used if `line_width` is not specified here.
        with_arrow : bool, optional
            If True, plot arrows from pts1 to pts2 instead of lines. Defaults to False.        arrow_head_size : float, optional
            Radius of the arrow head for the template arrow. This radius is relative
            to a template arrow of length 1. The final head size will scale
            with the arrow's actual length.
            If not specified, automatically computed based on the typical length of the arrow segments.
        shading : str, optional
            Shading style, primarily for arrows (when `with_arrow=True`).
            Options are 'flat', 'smooth', 'pbr', or 'albedo'.
            If None, PyVista's default shading is used for arrows.
            - 'flat': Flat shading.
            - 'smooth': Smooth shading (Gouraud).
            - 'pbr': Physically Based Rendering. Uses `metallic` and `roughness`.
            - 'albedo': Disables lighting for the arrows, showing only their base color at full brightness.
            This parameter has no direct effect on simple lines (`with_arrow=False`).
        metallic : float, optional
            Metallic property for PBR arrows (0.0 to 1.0). Used if `shading='pbr'`.
        roughness : float, optional
            Roughness property for PBR arrows (0.0 to 1.0). Used if `shading='pbr'`.
        **kwargs
            Additional keyword arguments passed to PyVista's add_lines() or add_mesh() (for arrows).        return
        ---------
        actor
            The pv actor of the lines or arrows, or None if no geometry was created.
        """
        pts1_arr = np.atleast_2d(pts1)
        pts2_arr = np.atleast_2d(pts2)
        num_segments = pts1_arr.shape[0]

        if num_segments == 0 or pts1_arr.shape[0] != pts2_arr.shape[0]:
            # No segments to draw or mismatched input arrays
            return None

        _single_color_rgb_float = None  # For single color (0.0-1.0 float)
        _multiple_colors_rgb_uint8 = None  # For N colors (0-255 uint8)

        if color3f is None:
            _single_color_rgb_float = np.array([1.0, 1.0, 1.0])  # Default color
        else:
            _color_input_arr = np.array(color3f, dtype=float)
            if (
                _color_input_arr.ndim == 2
                and _color_input_arr.shape[0] == num_segments
                and _color_input_arr.shape[1] == 3
            ):
                # (N,3) colors provided
                _multiple_colors_rgb_uint8 = (
                    np.clip(_color_input_arr, 0.0, 1.0) * 255
                ).astype(np.uint8)
            elif _color_input_arr.size == 3:  # Single color
                _single_color_rgb_float = np.clip(
                    _color_input_arr.flatten(), 0.0, 1.0
                )
            else:
                print(
                    f"Warning: Invalid color3f format (shape {_color_input_arr.shape}). Expected (3,) or ({num_segments},3) for {num_segments} segments. Using default color."
                )
                _single_color_rgb_float = np.array([1.0, 1.0, 1.0])

        obj = None  # Initialize object to be returned

        if with_arrow:
            # Compute directions and magnitudes first (needed for auto-sizing)
            directions = pts2_arr - pts1_arr
            magnitudes = np.linalg.norm(directions, axis=1)

            computed_line_width = None
            computed_arrow_head_size = None
            if line_width is None or arrow_head_size is None:
                all_points = np.vstack([pts1_arr, pts2_arr])
                bbox_min = np.min(all_points, axis=0)
                bbox_max = np.max(all_points, axis=0)
                bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
                avg_segment_length = (
                    np.mean(magnitudes) if len(magnitudes) > 0 else 1.0
                )
                if hasattr(self, "ArrowShaftRadiusRelative") and hasattr(
                    self, "ArrowHeadSizeRelative"
                ):
                    computed_line_width = (
                        self.ArrowShaftRadiusRelative * bbox_diagonal
                    )
                    computed_arrow_head_size = (
                        self.ArrowHeadSizeRelative * avg_segment_length
                    )
                else:  # Fallback if attributes are missing
                    computed_line_width = (
                        0.01 * bbox_diagonal if bbox_diagonal > 0 else 0.01
                    )
                    computed_arrow_head_size = (
                        0.1 * avg_segment_length
                        if avg_segment_length > 0
                        else 0.1
                    )

            final_arrow_head_size = (
                arrow_head_size
                if arrow_head_size is not None
                else computed_arrow_head_size
            )
            shaft_radius_from_line_width = (
                line_width if line_width is not None else computed_line_width
            )
            final_arrow_tip_length_ratio = 0.25

            arrow_geom = pv.Arrow(
                direction=(1.0, 0.0, 0.0),
                tip_length=final_arrow_tip_length_ratio,
                tip_radius=final_arrow_head_size,
                shaft_radius=shaft_radius_from_line_width,
                scale=1.0,
            )

            norm_directions = np.zeros_like(directions, dtype=float)
            non_zero_mag_mask = magnitudes > 1e-9
            if np.any(non_zero_mag_mask):
                valid_magnitudes_for_division = magnitudes[non_zero_mag_mask]
                if valid_magnitudes_for_division.ndim == 1:
                    valid_magnitudes_for_division = (
                        valid_magnitudes_for_division[:, np.newaxis]
                    )
                norm_directions[non_zero_mag_mask] = (
                    directions[non_zero_mag_mask]
                    / valid_magnitudes_for_division
                )
            norm_directions = np.nan_to_num(
                norm_directions, nan=0.0, posinf=0.0, neginf=0.0
            )

            glyph_points_pd = pv.PolyData(pts1_arr)
            glyph_points_pd.point_data["vectors"] = norm_directions
            glyph_points_pd.set_active_vectors("vectors")
            glyph_points_pd.point_data["scale_factor"] = magnitudes

            pv_call_kwargs = kwargs.copy()
            if (
                line_width is not None
            ):  # For arrow edges if rendered as wireframe/etc.
                pv_call_kwargs["line_width"] = line_width
            pv_call_kwargs.setdefault("style", "surface")

            # Shading logic (copied and adapted from original)
            pv_smooth_shading_arrow = None
            pv_pbr_arrow = None
            if shading is not None:
                shading_lower_arrow = shading.lower()
                if shading_lower_arrow == "albedo":
                    pv_call_kwargs["lighting"] = False
                    pv_smooth_shading_arrow = False
                    pv_pbr_arrow = False
                elif shading_lower_arrow == "flat":
                    pv_smooth_shading_arrow = False
                    pv_pbr_arrow = False
                    if "lighting" not in pv_call_kwargs:
                        pv_call_kwargs["lighting"] = True
                elif shading_lower_arrow == "smooth":
                    pv_smooth_shading_arrow = True
                    pv_pbr_arrow = False
                    if "lighting" not in pv_call_kwargs:
                        pv_call_kwargs["lighting"] = True
                elif shading_lower_arrow == "pbr":
                    pv_smooth_shading_arrow = True
                    pv_pbr_arrow = True
                    if "lighting" not in pv_call_kwargs:
                        pv_call_kwargs["lighting"] = True

            pv_call_kwargs["smooth_shading"] = pv_smooth_shading_arrow
            pv_call_kwargs["pbr"] = pv_pbr_arrow
            if pv_pbr_arrow:
                pv_call_kwargs["metallic"] = metallic
                pv_call_kwargs["roughness"] = roughness
            else:
                pv_call_kwargs.pop("metallic", None)
                pv_call_kwargs.pop("roughness", None)

            # Color handling for arrows
            if _multiple_colors_rgb_uint8 is not None:
                glyph_points_pd.point_data["actor_colors"] = (
                    _multiple_colors_rgb_uint8
                )
                pv_call_kwargs.pop(
                    "color", None
                )  # Ensure single color is not used
            elif _single_color_rgb_float is not None:
                pv_call_kwargs["color"] = _single_color_rgb_float

            glyphs_mesh = glyph_points_pd.glyph(
                geom=arrow_geom,
                orient="vectors",
                scale="scale_factor",
                factor=1.0,
            )

            if glyphs_mesh.n_points > 0:
                if _multiple_colors_rgb_uint8 is not None:
                    obj = self.m_plotter.add_mesh(
                        glyphs_mesh,
                        scalars="actor_colors",
                        rgb=True,
                        **pv_call_kwargs,
                    )
                else:  # Single color (either from _single_color_rgb_float or already in pv_call_kwargs)
                    obj = self.m_plotter.add_mesh(glyphs_mesh, **pv_call_kwargs)
        else:
            # Lines
            _effective_width_for_add_lines = 1.0
            if line_width is not None:
                _effective_width_for_add_lines = line_width

            if _multiple_colors_rgb_uint8 is not None:
                pts_for_polydata = np.column_stack(
                    (pts1_arr, pts2_arr)
                ).reshape((-1, 3))
                line_connectivity = np.arange(2 * num_segments).reshape(
                    (num_segments, 2)
                )
                cells_arg = np.hstack(
                    (
                        np.full((num_segments, 1), 2, dtype=int),
                        line_connectivity,
                    )
                ).ravel()

                poly_lines = pv.PolyData(pts_for_polydata, lines=cells_arg)
                poly_lines.cell_data["actor_colors"] = (
                    _multiple_colors_rgb_uint8
                )

                line_mesh_kwargs = kwargs.copy()
                line_mesh_kwargs.pop("color", None)
                line_mesh_kwargs["line_width"] = _effective_width_for_add_lines
                line_mesh_kwargs["style"] = (
                    "wireframe"  # Ensure lines are rendered
                )
                line_mesh_kwargs.setdefault("render_lines_as_tubes", False)

                obj = self.m_plotter.add_mesh(
                    poly_lines,
                    scalars="actor_colors",
                    rgb=True,
                    **line_mesh_kwargs,
                )
            else:  # Single color for lines
                pts_input = np.column_stack((pts1_arr, pts2_arr)).reshape(
                    (-1, 3)
                )
                obj = self.m_plotter.add_lines(
                    pts_input,
                    color=_single_color_rgb_float,
                    width=_effective_width_for_add_lines,
                    **kwargs,
                )
        return obj

    def add_points(
        self,
        pts: np.ndarray,
        color3f: np.ndarray = None,
        style: str = "points",
        **kwargs,
    ) -> pv.Actor:
        """add points to the plot

        parameters
        --------------
        pts : (N,3)
            N points
        color3f : (3,)
            a single color for all points
        style : 'points' | 'points_gaussian' | 'sphere'
            the geometry that represents the points
        """
        _style: str = None
        _render_points_as_spheres: bool = False
        if style.lower() in ("points", "points_gaussian"):
            _style = style
        elif style.lower() == "sphere":
            _style = "points"
            _render_points_as_spheres = True

        if color3f is None:
            color3f = np.ones(3)
        else:
            color3f = np.atleast_1d(color3f).astype(float)
            if color3f.size == 3:
                color3f = np.atleast_1d(color3f).flatten().astype(float)
            else:
                assert len(color3f) == len(
                    pts
                ), f"number of points should match number of given colors"

        if color3f.size == 3:
            actor = self.m_plotter.add_points(
                pts,
                style=_style,
                render_points_as_spheres=_render_points_as_spheres,
                color=color3f,
                emissive=True,
                **kwargs,
            )
        else:
            actor = self.m_plotter.add_points(
                pts,
                style=_style,
                render_points_as_spheres=_render_points_as_spheres,
                scalars=color3f,
                rgb=True,
            )

        return actor

    def add_text(
        self,
        text_content: str,
        position: np.ndarray,
        font_size: float | None = None,
        color3f=None,  # Changed from color to color3f
        font_family: str = "arial",
        bold: bool = False,
        italic: bool = False,
        shadow: bool = False,
        **kwargs,
    ) -> pv.Actor:
        """Add text in 3D space that always faces the camera (billboard text).

        parameters
        -------------
        text_content : str
            The text string to display.
        position : np.ndarray
            (3,) array for the 3D position of the text.
        font_size : float
            The size of the font.
        color3f : str or sequence, optional # Changed from color to color3f
            Color of the text. Can be a string name (e.g., 'white'), a hex string, or a sequence of RGB values.
            Defaults to PyVista's theme color.
        font_family : str
            Font family for the text (e.g., 'arial', 'courier', 'times').
        bold : bool
            Whether the text should be bold.
        italic : bool
            Whether the text should be italic.
        shadow : bool
            Whether the text should have a shadow (can improve readability).
        **kwargs
            Additional keyword arguments passed to self.m_plotter.add_point_labels().

        return
        ---------
        actor
            The pv actor for the text labels.
        """
        if font_size is None:
            font_size = 10

        position = np.array(position).reshape(
            1, 3
        )  # add_point_labels expects a list/array of points
        labels = [text_content]

        # Ensure no actual point marker is drawn, only the label
        kwargs.setdefault("show_points", False)
        # Make the backing shape for the label transparent if one exists by default
        kwargs.setdefault("shape_opacity", 0.0)

        actor = self.m_plotter.add_point_labels(
            points=position,
            labels=labels,
            font_size=font_size,
            text_color=color3f,  # Changed from color to color3f
            font_family=font_family,
            bold=bold,
            italic=italic,
            shadow=shadow,
            **kwargs,
        )
        return actor

    def set_camera_transform_by_vectors(
        self,
        view_dir: np.ndarray,
        up_dir: np.ndarray,
        position: np.ndarray,
        focal_distance=None,
    ):
        camera_set_transform_by_vectors(
            self.m_plotter.camera,
            pos=position,
            view_dir=view_dir,
            up_dir=up_dir,
            focal_distance=focal_distance,
        )
        self.m_plotter.camera.Modified()

    def set_camera_transform_by_4x4(self, gltf_transmat: np.ndarray):
        camera_set_transform_by_4x4(self.m_plotter.camera, gltf_transmat)
        self.m_plotter.camera.Modified()
        
    def show(self):
        """Show the plotter window."""
        if self.m_plotter is not None and isinstance(self.m_plotter, pv.BasePlotter):
            # background plotter do not need this, just for std plotter
            self.m_plotter.show()

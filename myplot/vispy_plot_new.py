import numpy as np
import vispy
from vispy import app, gloo, visuals, scene, io
from vispy.visuals.transforms import TransformSystem
from vispy.visuals.mesh import MeshVisual
from vispy.visuals import LinePlotVisual, LineVisual
from vispy.gloo.util import _screenshot
from vispy.gloo import Program, VertexBuffer, IndexBuffer
import vispy.util.transforms as vistrans
from igpy.modeling.modeldata import ModelData
from . import vispy_config

MAX_RENDER_IMAGE_DIMENSION = 1024

#vispy.use('PyQt5')
#vispy.use(app='PySide')
usable_backends = vispy_config.usable_backends
#usable_backends = ['SDL2', 'PySide', 'PyQt5']
for x in usable_backends:
    try:
        vispy.use(app = x)
        print('vispy backend = ' + x)
        break
    except:
        pass
    
#try:
    #vispy.use(app='SDL2')
#except:
    #vispy.use(app='PySide')

def direction_to_azimuth_elevation(v):
    ''' convert a directional vector into (azimuth, elevation)
    
    parameters
    ---------------
    v
        a 3d directional vector
    
    returns
    ------------------
    azimuth
        the azimuth angle in radian
    elevation
        the elevation angle in radian
    '''
    
    # in vispy convention
    # azimuth = angle to Y
    # elevation = angle to Z
    # a point in space is (cos(el)sin(az), cos(el)cos(az), sin(el))
    
    v = -np.array(v) #take the reverse direction, because we are looking towards the center
    v = v/np.linalg.norm(v)
    elevation = np.arcsin(v[-1])
    azimuth = np.arctan2(v[0],v[1]) #azimuth angle is relativ to Y axis, so we use tan(az)=x/y rather thand y/x
    return azimuth, elevation

def show_single_mesh_front(vertices, faces, texcoord = None, texture_image = None,
                           vertex_color = None, vertex_normals = None, bgcolor4f = None, 
                           show_now = True, width = 500, viewdir = None,
                           idxface_show = None, view_rect_xywh = None,
                           show_backface = True, w_front_light = None):
    ''' show a mesh by looking at -z axis
    
    parameters
    ---------------------------
    view_rect_xywh
        (x,y,width,height) in world unit, specify the view
    '''
    if bgcolor4f is None:
        bgcolor4f = (0,0,0,1)
    
    if vertices.shape[-1] == 2:
        vertices = np.column_stack((vertices, np.zeros(len(vertices))))
        
    if texture_image is None and vertex_color is None:
        vertex_color = np.ones(vertices.shape)
        
    if w_front_light is None:
        w_front_light = 0
        
    if texture_image is not None and texcoord is None:
        assert False, 'texcoord is missing'
    
    if idxface_show is not None:
        # use the point with lowest value in texture image as dummy
        if np.ndim(texture_image) == 3:
            h, w = texture_image.shape[:2]
            x = np.sum(texture_image, axis=2)
        else:
            x = texture_image
            
        ii, jj = np.nonzero(x == x.min())
        u = jj[0]/w
        v = 1-ii[0]/h
        dummy_point_uv = (u,v)
        #dummy_point_uv = (0,0)
        # add a point in texcoord
        texc = np.row_stack((texcoord, texcoord[0]))
        texc[-1] = dummy_point_uv
        
        # set all invisible face to use the dummy point
        texface = np.array(faces)
        mask = np.zeros(len(faces), dtype=bool)
        mask[idxface_show] = True
        texface[~mask] = len(texc)-1
    else:
        texc = texcoord
        texface = faces
        
    md = ModelData(vertices, faces, texcoord_uvw=texc, texcoord_faces=texface,
                   texture_image = texture_image, vertex_color_data=vertex_color,
                   vertex_color_faces=faces)
    if vertex_normals is not None:
        md.set_vertex_normals(vertex_normals)
    
    canvas = TightRawMeshCanvas()
    canvas.set_light_weight(w_front_light)
    canvas.set_model_data(md)
    
    if texture_image is not None:
        canvas.set_display_mode(canvas.DPMODE_TEXTURE_WITH_LIGHT)
    if vertex_color is not None:
        canvas.set_display_mode(canvas.DPMODE_VERTEX_COLOR_WITH_LIGHT)
        
    if show_backface is not None:
        canvas.set_show_backface(show_backface)
    
    #canvas.set_display_width(width)
    canvas.set_display_size(width,width)
    if view_rect_xywh is None:
        canvas.set_view_rect_tight(adjust_window_size=True)
    else:
        canvas.set_view_rect(*view_rect_xywh, adjust_window_size=True)
    canvas.set_background_color(bgcolor4f)
    canvas.update_view()
    canvas.update()
    if show_now:
        canvas.show(run=True)
    return canvas


def show_model_data_interactive(model, viewdir = None, bgcolor3f = (0,0,0),
                               show_now = True, show_axis = False, w_front_light = 0):
    ''' show a ModelData
    
    parameters
    -------------------
    w_front_light
        the weight of frontal lighting, 1.0 = use full light, 0 = no light.
    '''
    from igpy.modeling.modeldata import ModelData
    if False:
        model = ModelData()
        
    v,f,vn,vt = model.get_unstitched_model()
    v = v - v.mean(axis=0)
    teximg = model.texture_image
    
    if vt is None or teximg is None:
        vt = None
        teximg = None
        
    return show_single_mesh_interactive(v,f,texcoord=vt,texture_image=teximg,viewdir=viewdir,
                                        show_axis=show_axis,show_now=show_now,bgcolor3f=bgcolor3f,
                                        vertex_normal=vn, w_front_light = w_front_light)

def show_single_mesh_interactive(vertices, faces, texcoord = None, texture_image = None, 
                     viewdir = None, bgcolor3f = (0,0,0), vertex_color = None, vertex_normal = None,
                     show_now = True, show_axis = False, w_front_light = 0):
    ''' show a single textured mesh model
    
    return
    -------------------
    canvas
        the SceneCanvas used to draw the content
    mesh_visual
        the mesh visual
    '''
    if vertices.shape[-1] == 2:
        vertices = np.column_stack((vertices, np.zeros(len(vertices))))
    md = ModelData(vertex_local=vertices, faces=faces, texcoord_uvw=texcoord,
                   texcoord_faces=faces, texture_image = texture_image,
                   vertex_color_data = vertex_color,
                   vertex_color_faces = faces)
    if vertex_normal is not None:
        md.set_vertex_normals(vertex_normal)
        
    canvas = CanvasWithScreenshot(keys = 'interactive', show = show_now)
    view = canvas.central_widget.add_view()
    view.bgcolor = bgcolor3f
    
    x = TexturedMeshVisual()
    MyMesh = scene.visuals.create_visual_node(TexturedMeshVisual)
    mesh = MyMesh(parent=view.scene)    
    mesh.set_model_data(md)
    mesh.set_light_weight(w_front_light)
    #mesh.visible = False
    
    vertices = mesh.m_vertices
    faces = mesh.m_faces
    vertices_use = vertices
    aabb = np.row_stack((vertices_use.min(axis=0),vertices_use.max(axis=0)))
    boxsize = aabb[1] - aabb[0]
    rx,ry,rz = aabb.T    
    
    # create wireframe
    #Plot3D = scene.visuals.create_visual_node(MyLineVisual)
    #pos = vertices
    #connect = np.row_stack([faces[:,[0,1]], faces[:,[0,2]], faces[:,[1,2]]])
    #wireframe = Plot3D(pos = pos, connect = connect, color = (1,0,0,1),
                       #parent = view.scene)
    
    az, el = 0,0
    if viewdir is not None:
        az, el = direction_to_azimuth_elevation(viewdir)
        az = np.rad2deg(az)
        el = np.rad2deg(el)
    view.camera = 'turntable'
    view.camera.fov = 0
    view.camera.azimuth = az
    view.camera.elevation = el
    view.camera.distance = 0
    view.camera.set_range(x=rx, y=ry, margin = 0.)
    if show_axis:
        #axis = scene.visuals.XYZAxis(parent = view)
        axclass = scene.visuals.create_visual_node(XYZAxisVisual)
        vbox = vertices.max(axis=0) - vertices.min(axis=0)
        vlen = np.linalg.norm(vbox)
        axis = axclass(parent = view.scene,  length = vlen * 0.5)
    
    if show_now:
        app.run()
    return canvas, mesh

class MyLineVisual(LineVisual):
    def __init__(self, *args, **kwargs):
        LineVisual.__init__(self, *args, **kwargs)
    
    def _prepare_draw(self, view):
        gloo.gl.glLineWidth(3.0)
        super()._prepare_draw(view)

class CanvasWithScreenshot(scene.SceneCanvas):
    ''' A SceneCanvas which takes a screenshot every frame, and store it in
    self.m_screenshot.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfreeze()
        self.m_screenshot = None
        self.freeze()
        
    def on_draw(self, event):
        super().on_draw(event)
        self.m_screenshot = _screenshot()
        
    def draw(self):
        super().draw()
        self.m_screenshot = _screenshot()

class TexturedMeshVisual(visuals.Visual):
    ''' A vispy visual for showing a textured mesh
    '''
    
    VERTEX_SHADER = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    varying vec3 v_normal;
    varying vec3 v_eye_dir;
    
    //things we need from the caller:
    //uniform vec3 vertex_normal
    //
    
    void main()
    {
        vec4 visual_pos = vec4($position,1.0);
        vec4 pos = $visual_to_render(visual_pos);
        gl_Position = pos;
        v_texcoord = $texcoord;
        
        vec4 scene_pos = $visual_to_scene(visual_pos);
        vec4 _pos;
        _pos = $scene_to_doc(scene_pos);
        _pos.z += 0.01;
        vec4 p_front = $doc_to_scene(_pos);
        p_front /= p_front.w;
        
        _pos.z -= 0.02;
        vec4 p_back = $doc_to_scene(_pos);
        p_back /= p_back.w;
        
        v_eye_dir = normalize(p_back.xyz - p_front.xyz);
        
        vec4 _normal = $visual_to_scene(vec4($normal,0.0));
        v_normal = normalize(_normal.xyz);
    }
    """
    
    FRAGMENT_SHADER = """
    uniform sampler2D texture;    
    varying vec2 v_texcoord;
    varying vec3 v_normal;   
    varying vec3 v_eye_dir;
    void main()
    {
        float light_coef = dot(v_eye_dir, v_normal);
        light_coef = 1.0 * (1-$light_weight) + light_coef * $light_weight;
        gl_FragColor = texture2D(texture, v_texcoord);
        gl_FragColor.xyz *= light_coef;
    }
    """    
    
    def __init__(self):
        super().__init__(TexturedMeshVisual.VERTEX_SHADER, 
                         TexturedMeshVisual.FRAGMENT_SHADER)
        self.m_vertices = None
        self.m_faces = None
        self.m_normals = None
        self.m_texcoord = None
        self.m_texture_image = None
        
        # load mesh and texture
        self.m_glbuf_face = gloo.IndexBuffer()
        self.m_glbuf_vertex = gloo.VertexBuffer()
        self.m_glbuf_texcoord = gloo.VertexBuffer()
        self.m_glbuf_vnormal = gloo.VertexBuffer()
        
        self._draw_mode = 'triangles'
        self.m_light_weight = 0.0
        
    def set_light_weight(self, w):
        ''' set the weight of head light
        
        parameters
        -----------------
        w
            a number between 0 and 1, the effect of head lighting.
            1.0 = use head light completely, 0 = turn of head light
        '''
        self.m_light_weight = w
        
    def set_model_data(self, model_data):
        ''' set a ModelData into the visual
        '''
        md = model_data
        if md.is_texcoord_unique_for_vertex:
            v,f,vn,vt = md.vertices, md.faces, md.get_vertex_normals(), md._texcoord_uvw[:,:-1]
        else:
            v,f,vn,vt = md.get_unstitched_model()
        vt[:,1] = 1-vt[:,1]
        self.m_vertices = v.astype(np.float32)
        self.m_faces = f.astype(np.uint32)
        self.m_normals = vn.astype(np.float32)
        self.m_texcoord = vt.astype(np.float32)
        if md._texture_image is not None:
            self.m_texture_image = md._texture_image       
        else:
            self.m_texture_image = np.zeros((256,256,3), dtype=np.uint8)
        
    def _prepare_transforms(self,view):
        #view.view_program.vert['transform'] = view.get_transform()
        self.shared_program.vert['visual_to_render'] = view.get_transform()
        self.shared_program.vert['visual_to_scene'] = view.get_transform('visual','scene')
        self.shared_program.vert['scene_to_doc'] = view.get_transform('scene','document')
        self.shared_program.vert['doc_to_scene'] = view.get_transform('document', 'scene')
        #self.shared_program.vert['visual_to_frame'] = view.get_transform('visual','framebuffer')
        #self.shared_program.vert['doc_to_visual'] = view.get_transform('document','visual')
        pass
    
    def _prepare_draw(self, view):
        self.set_gl_state('translucent', depth_test = True, cull_face = False)
        self.shared_program.frag['light_weight'] = self.m_light_weight
        
        self.m_glbuf_vertex.set_data(self.m_vertices, convert=True)
        self.shared_program.vert['position'] = self.m_glbuf_vertex
        
        self.m_glbuf_vnormal.set_data(self.m_normals, convert=True)
        self.shared_program.vert['normal'] = self.m_glbuf_vnormal
        
        self.m_glbuf_face.set_data(self.m_faces, convert=True)
        self._index_buffer = self.m_glbuf_face
        
        self.m_glbuf_texcoord.set_data(self.m_texcoord, convert=True)
        self.shared_program.vert['texcoord'] = self.m_glbuf_texcoord
        
        self.shared_program['texture'] = self.m_texture_image
        
class TightMeshCanvas(app.Canvas):
    def __init__(self):
        super().__init__(keys = 'interactive')
        self.m_mesh_visual_list = []
        
    def add_mesh_visual(self, mesh_visual):
        self.m_mesh_visual_list.append(mesh_visual)
        
    def on_resize(self, event):
        vp = (0,0,self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for mesh in self.m_mesh_visual_list:
            mesh.transforms.configure(canvas = self, viewport = vp)
            
    def on_draw(self, ev):
        gloo.set_viewport(0,0,*self.physical_size)
        gloo.clear(color='black', depth=True)
        for mesh in self.m_mesh_visual_list:
            mesh.draw()
            
class ShaderProgramInfo:
    def __init__(self, program = None, vertex_dtype = None, vertex_buffer = None,
                 name = None):
        self.program = program
        self.vertex_dtype = vertex_dtype
        self.vertex_buffer = vertex_buffer
        self.name = name
            
class TightRawMeshCanvas(app.Canvas):
    ''' show a single mesh
    
    The axis is fixed: x left, y up, z out, origin at bottom left
    '''
    
    VERTEX_SHADER = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform sampler2D texture;
    attribute vec3 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = projection * view * model * vec4(position,1.0);
        v_texcoord = texcoord;
    }
    """
    
    FRAGMENT_SHADER = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
    """    
    
    DTYPE_VERTEX = [('position', np.float32, 3), ('texcoord', np.float32, 2)]
    
    DPMODE_VERTEX_COLOR = 'use_vertex_color'
    DPMODE_VERTEX_COLOR_WITH_LIGHT = 'use_vertex_color_with_light'
    DPMODE_TEXTURE = 'use_texture'
    DPMODE_TEXTURE_WITH_LIGHT = 'use_texture_with_light'
    
    def __init__(self):
        super().__init__(keys='interactive')
        self.m_timer = app.Timer('auto', self.on_timer)
        
        # create programs
        program_set = {}
        
        # color mesh without lighting
        vs,fs,vtype = get_shader_colored_mesh()
        pg = Program(vs,fs)
        program_set[self.DPMODE_VERTEX_COLOR] = \
            ShaderProgramInfo(program=pg, vertex_dtype=vtype, name=self.DPMODE_VERTEX_COLOR)
        
        # color mesh with lighting
        vs,fs,vtype = get_shader_colored_mesh(with_lighting=True)
        pg = Program(vs,fs)
        program_set[self.DPMODE_VERTEX_COLOR_WITH_LIGHT] = \
            ShaderProgramInfo(program=pg, vertex_dtype=vtype, name=self.DPMODE_VERTEX_COLOR_WITH_LIGHT)            
        
        # texture mesh without lighting
        vs,fs,vtype = get_shader_texture_mesh()
        pg = Program(vs,fs)
        program_set[self.DPMODE_TEXTURE] = \
            ShaderProgramInfo(program=pg, vertex_dtype=vtype, name=self.DPMODE_TEXTURE)
        
        # texture mesh with lighting
        vs,fs,vtype = get_shader_texture_mesh(with_lighting=True)
        pg = Program(vs,fs)
        program_set[self.DPMODE_TEXTURE_WITH_LIGHT] = \
            ShaderProgramInfo(program=pg, vertex_dtype=vtype, name=self.DPMODE_TEXTURE_WITH_LIGHT)        
        
        self.m_program_set = program_set
        
        self.m_display_mode = self.DPMODE_TEXTURE
        
        # current program
        self.m_program = self.m_program_set[self.m_display_mode].program
        
        #self.m_program = Program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)
        
        self.m_gl_faces = None
        self.m_mat_model = np.eye(4)
        self.m_mat_view = np.eye(4)
        self.m_mat_projection = np.eye(4)
        self.m_model_data = None        
        self.m_aabb = None #[xmin,ymin,zmin;xmax,ymax,zmax]
        self.m_view_rect_xywh = None
        self.m_show_backface = True
        self.m_light_weight = 1.0
        
        self.update_view()
        gloo.set_state(clear_color=(0.,0.,0.,1.0), depth_test = True, cull_face = True)
        
        self.m_timer.start()
        #self.show(visible=True)
        
    def set_show_backface(self, tf):
        ''' set whether the backface is shown or culled
        '''
        self.m_show_backface = tf
        gloo.set_state(cull_face = not tf)
        
    def set_display_mode(self, dp):
        self.m_display_mode = dp
        self.m_program = self.m_program_set[dp].program
        
    def set_light_weight(self, w):
        self.m_light_weight = w
        
    def set_view_rect_tight(self, adjust_window_size = False):
        ''' adjust view rect such that the content fills the entire viewport
        
        parameters
        --------------------
        adjust_window_size
            should the window size be adjusted so that its aspect ratio follows the content.
            If true, the window width is fixed and the height is adjusted
        '''
        rx, ry = self.m_aabb[:,:-1].T
        w = rx[1] - rx[0]
        h = ry[1] - ry[0]
        self.m_view_rect_xywh = np.array([rx[0], ry[0], w, h])
        
        if adjust_window_size:
            width, height = self.size
            height = int(h/w * width)
            self.set_display_size(width, height)
            
        self.update_view()
        
    def set_view_rect(self, x, y, width, height, adjust_window_size = False):
        ''' set view rectangle in world coordinate
        '''
        self.m_view_rect_xywh = np.array([x,y,width,height])
        h = height
        w = width
        
        if adjust_window_size:
            width, height = self.size
            height = int(h/w * width)
            self.set_display_size(width, height)        
        
        self.update_view()
        
    def on_timer(self, evt):
        self.update()
        
    def get_render_image(self):
        self.show()
        
        # update several times to refresh the depth buffer
        self.update()
        app.process_events()
        self.update()
        app.process_events()
        self.update()
        app.process_events()
        
        # screenshot() is more robust
        img = gloo.util._screenshot()
        #img = self.render()
        self.close()
        return img
        
    def set_background_color(self, color4f = (0.,0.,0.,1.0)):
        gloo.set_state(clear_color = color4f, depth_test = True)
        self.update()
        
    def update_view(self):
        gloo.set_viewport(0,0,*self.physical_size)
        #projmat = vistrans.perspective(45.0, self.size[0]/self.size[1],
                                       #1.0, 1000.0)
        if self.m_aabb is None:
            projmat = vistrans.ortho(-10,10,-10,10,0,10000)
        else:
            x,y,w,h = self.m_view_rect_xywh
            projmat = vistrans.ortho(x,x+w,y,y+h,0,10000)
        self.m_program['projection'] = projmat
        self.m_mat_projection = projmat
        
    def set_display_size(self, width, height):
        ''' Note that the render image dimension is constrained by MAX_RENDER_IMAGE_DIMENSION,
        if the width or height is greater than this, it will be shrinked down.
        
        Return
        ---------------
        width, height
            the width and height actually set into the window, considering the MAX_RENDER_IMAGE_DIMENSION
            constraint.
        '''
        longdim = max(width,height)
        if longdim > MAX_RENDER_IMAGE_DIMENSION:
            scale = MAX_RENDER_IMAGE_DIMENSION/longdim
            width = int(width * scale)
            height = int(height * scale)
            
        self.size = (int(width),int(height))
        return (width, height)
        
    def set_display_width(self, width):
        ''' set the width of the window, maintaining aspect ratio.
        '''
        w, h = self.size
        height = int(width * h/w)
        self.set_display_size(width, height)
        self.size = (width, height)
        
        #rx, ry = self.m_aabb[:,:-1].T
        #data_width, data_height = rx[1]-rx[0], ry[1]-ry[0]
        #height = int(width * data_height/data_width)
        #self.size = (width, height)
        
    def on_resize(self, evt):
        import pprint
        self.update_view()
        
    def set_model_data(self, model_data):
        if False:
            model_data = ModelData()    
            
        self.m_model_data = model_data
        v = model_data.vertices
        self.m_aabb = np.row_stack((v.min(axis=0), v.max(axis=0)))
        
        self.update_rendering()
        
    def update_rendering(self):
        model_data = self.m_model_data
        
        # fill in the shader
        if False:
            model_data = ModelData()        
            
        v = model_data.vertices.astype(np.float32)
        vn = model_data.get_vertex_normals()
        
        z = v[:,-1]
        z = z - z.max() - 100
        v[:,-1] = z
        
        f = model_data.faces
        
        # update data in program
        for key, val in self.m_program_set.items():
            if False:
                val = ShaderProgramInfo()
            vtype = val.vertex_dtype
            program = val.program
            v_data = np.zeros(len(v), dtype=vtype)
            v_data['position'] = v
            
            if key == self.DPMODE_TEXTURE and model_data.has_texcoord:
                vt = model_data.get_texcoord_per_vertex().astype(np.float32)
                vt[:,1] = 1-vt[:,1]
                img = model_data.texture_image                
                v_data['texcoord'] = vt
                program['texture'] = img
            
            elif key == self.DPMODE_TEXTURE_WITH_LIGHT and model_data.has_texcoord:
                vt = model_data.get_texcoord_per_vertex().astype(np.float32)
                vt[:,1] = 1-vt[:,1]
                img = model_data.texture_image                
                v_data['texcoord'] = vt
                v_data['normal'] = vn
                program['texture'] = img
                program['lightdir'] = np.array([0,0,-1], dtype=np.float32)
                program['lightweight'] = self.m_light_weight
                normal_matrix = np.linalg.inv(self.m_mat_view.dot(self.m_mat_model)).T
                program['normal_transform'] = normal_matrix                
                
            elif key == self.DPMODE_VERTEX_COLOR:# and model_data.vertex_color4f is not None:
                vcolor = model_data.get_color4f_per_vertex()
                v_data['vertex_color'] = vcolor
                
            elif key == self.DPMODE_VERTEX_COLOR_WITH_LIGHT:
                vcolor = model_data.get_color4f_per_vertex()
                v_data['vertex_color'] = vcolor
                v_data['normal'] = vn
                program['lightdir'] = np.array([0,0,-1], dtype=np.float32)
                program['lightcolor'] = np.array([1,1,1], dtype=np.float32)
                normal_matrix = np.linalg.inv(self.m_mat_view.dot(self.m_mat_model)).T
                program['normal_transform'] = normal_matrix
                
            val.vertex_buffer = VertexBuffer(v_data)
            program.bind(val.vertex_buffer)
                
        self.m_gl_faces = IndexBuffer(f.astype(np.uint32))
        
        if self.m_view_rect_xywh is None:
            self.set_view_rect_tight()
        
    def on_draw(self, evt):
        gloo.clear(color=True, depth=True)
        self.m_program['model'] = self.m_mat_model
        self.m_program['view'] = self.m_mat_view        
        self.m_program.draw('triangles', self.m_gl_faces)
        
def get_shader_colored_mesh(with_lighting = False):
    ''' get shader suitable for displaying mesh with vertex color
    
    return
    ---------------------
    vertex_shader
        the vertex shader code
    fragment_shader
        the fragment shader code
    vertex_dtype
        the numpy dtype for creating vertex array
    '''
    
    VERTEX_SHADER_NO_LIGHT = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    attribute vec3 position;
    attribute vec4 vertex_color;
    
    varying vec4 vcolor;
    void main()
    {
        gl_Position = projection * view * model * vec4(position,1.0);
        vcolor = vertex_color;
    }
    """
    
    FRAGMENT_SHADER_NO_LIGHT = """
    varying vec4 vcolor;
    void main()
    {
        gl_FragColor = vcolor;
    }
    """        
    
    VERTEX_SHADER_WITH_LIGHT = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 normal_transform;
    uniform vec3 lightdir;
    uniform vec3 lightcolor;

    attribute vec3 position;
    attribute vec4 vertex_color;
    attribute vec3 normal;

    varying vec4 vcolor;
    varying vec3 vnormal;
    void main()
    {
        gl_Position = projection * view * model * vec4(position,1.0);
        //mat4 normal_matrix = transpose(inverse(view * model));        
        vcolor = vertex_color;
        vnormal = normalize(vec3(normal_transform * vec4(normal, 0.0)));
    }
    """    
    
    FRAGMENT_SHADER_WITH_LIGHT = """
    uniform vec3 lightdir;
    uniform vec3 lightcolor;
    varying vec4 vcolor;
    varying vec3 vnormal;
    void main()
    {
        float lightcoef = abs(dot(lightdir, vnormal));
        gl_FragColor = vcolor * vec4(lightcolor,1.0)* lightcoef;
    }
    """   
    
    if with_lighting:
        DTYPE_VERTEX = [('position', np.float32, 3), ('vertex_color', np.float32, 4), 
                        ('normal', np.float32,3)]
        return VERTEX_SHADER_WITH_LIGHT, FRAGMENT_SHADER_WITH_LIGHT, DTYPE_VERTEX
    else:
        DTYPE_VERTEX = [('position', np.float32, 3), ('vertex_color', np.float32, 4)]    
        return VERTEX_SHADER_NO_LIGHT, FRAGMENT_SHADER_NO_LIGHT, DTYPE_VERTEX    
        
def get_shader_texture_mesh(with_lighting = False):
    ''' get shader suitable for displaying textured mesh
    
    return
    ---------------------
    vertex_shader
        the vertex shader code
    fragment_shader
        the fragment shader code
    vertex_dtype
        the numpy dtype for creating vertex array
    '''
    
    VERTEX_SHADER_NO_LIGHT = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform sampler2D texture;
    attribute vec3 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = projection * view * model * vec4(position,1.0);
        v_texcoord = texcoord;
    }
    """
    
    FRAGMENT_SHADER_NO_LIGHT = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
    """    
    
    VERTEX_SHADER_WITH_LIGHT = """
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform sampler2D texture;
    
    uniform mat4 normal_transform;
    
    attribute vec3 position;
    attribute vec2 texcoord;
    attribute vec3 normal;
    
    varying vec2 v_texcoord;
    varying vec3 vnormal;
    void main()
    {
        gl_Position = projection * view * model * vec4(position,1.0);
        v_texcoord = texcoord;
        vnormal = normalize(vec3(normal_transform * vec4(normal, 0.0)));
    }
    """
    
    FRAGMENT_SHADER_WITH_LIGHT = """
    uniform sampler2D texture;
    uniform float lightweight;
    uniform vec3 lightdir;
    
    varying vec2 v_texcoord;
    varying vec3 vnormal;
    void main()
    {
        float lightcoef = abs(dot(lightdir, vnormal));
        vec4 texcolor = texture2D(texture, v_texcoord);
        gl_FragColor = texcolor * lightcoef * lightweight + texcolor * (1-lightweight);
    }
    """        
    
    if with_lighting:
        DTYPE_VERTEX = [('position', np.float32, 3), 
                        ('texcoord', np.float32, 2), 
                        ('normal', np.float32,3)]
        return VERTEX_SHADER_WITH_LIGHT, FRAGMENT_SHADER_WITH_LIGHT, DTYPE_VERTEX        
    else:
        DTYPE_VERTEX = [('position', np.float32, 3), 
                        ('texcoord', np.float32, 2)]    
        return VERTEX_SHADER_NO_LIGHT, FRAGMENT_SHADER_NO_LIGHT, DTYPE_VERTEX

class XYZAxisVisual(vispy.visuals.LineVisual):
    """
    Simple 3D axis for indicating coordinate system orientation. Axes are
    x=red, y=green, z=blue.
    """
    def __init__(self, origin = None, xyzdir = None, length = None, 
                 **kwargs):
        if origin is None:
            origin = np.zeros(3)
        
        if xyzdir is None:
            xyzdir = np.eye(3)
            
        if length is None:
            length = 1
            
        verts = np.row_stack((origin, origin + xyzdir[0] * length,
                              origin, origin + xyzdir[1] * length,
                              origin, origin + xyzdir[2] * length))
            
        #verts = np.array([[0, 0, 0],
                          #[1, 0, 0],
                          #[0, 0, 0],
                          #[0, 1, 0],
                          #[0, 0, 0],
                          #[0, 0, 1]])
        color = np.array([[1, 0, 0, 1],
                          [1, 0, 0, 1],
                          [0, 1, 0, 1],
                          [0, 1, 0, 1],
                          [0, 0, 1, 1],
                          [0, 0, 1, 1]])
        super().__init__(pos=verts * length, color=color, connect='segments',
                        method='gl', **kwargs)
import numpy as np
from os import path as op
from struct import *
import logging as logger

def load_collada(fn_collada):
    ''' load all the meshes in collada file
    
    parameters
    ---------------
    fn_collada
        collada filename
    
    return
    --------------
    modelist
        a list of ModelData
    '''
    import collada
    from ..modeling.modeldata import ModelData
    from ..geometry import geometry as geom
    
    obj = collada.Collada(fn_collada)
    modelist = []
    for node in obj.scene.nodes:
        node : collada.scene.Node = node
        tmat = node.matrix.T
        submodelist = []
        for g_obj in node.objects('geometry'):
            for prim in g_obj._primitives:
                prim : collada.triangleset.BoundTriangleSet = prim
                v = prim.vertex
                f = prim.vertex_index
                
                this_model = ModelData(v,f)
                submodelist.append(this_model)
        model = geom.merge_models(submodelist)
        model.name = node.id
        model.set_transmat(tmat)
        modelist.append(model)    
    return modelist

def load_obj_single_mesh_v3(obj_filename, texture_filename = None):
    ''' load .obj file, where the file only contains one object
    
    Return
    ------------------------
    meshdata
        a ModelData object containing the mesh
    '''
    import vispy.io as vio
    from igpy.modeling.modeldata import ModelData
    
    v = []
    f = []
    
    vt = []
    vtf = []
    
    vn = []
    vnf = []
    
    #normal = []
    #texcoord = []
    with open(obj_filename, 'r') as fid:
        for line in fid:
            line = line.strip()
            
            if line.startswith('v '):
                data = [float(x) for x in line.split()[1:4]]
                v.append(data)
            elif line.startswith('vt '):
                data = [float(x) for x in line.split()[1:]]
                vt.append(data)                
            elif line.startswith('vn '):
                data = [float(x) for x in line.split()[1:]]
                vn.append(data)                
            elif line.startswith('f '):
                # do we have '/'?
                tx_faces = line.split()[1:]
                this_vf = []
                this_vtf = []
                this_vnf = []
                for x in tx_faces:
                    x_parts = x.split('/')
                    this_vf.append(int(x_parts[0]))
                    if len(x_parts) >= 2 and x_parts[1] != '':
                        this_vtf.append(int(x_parts[1]))
                    if len(x_parts) >= 3 and x_parts[2] != '':
                        this_vnf.append(int(x_parts[2]))
                
                f.append(this_vf)
                if len(this_vtf) > 0:
                    vtf.append(this_vtf)
                if len(this_vnf) > 0:
                    vnf.append(this_vnf)
            elif line.startswith('#'):
                pass  # Comment
            elif line.startswith('mtllib '):
                pass
            elif any(line.startswith(x) for x in ('g ', 's ', 'o ', 'usemtl ')):
                pass  # Ignore groups and smoothing groups, obj names, material
            elif not line.strip():
                pass
            else:
                logger.warning('Notice reading .OBJ: ignoring %s command.'
                               % line.strip())            
            
    v = np.array(v)
    f = (np.array(f)-1).astype(int)
    
    vt = np.array(vt) if len(vt)>0 else None
    vtf = np.array(vtf) if len(vtf)>0 else None
    if vtf is not None:
        vtf = (vtf - 1).astype(int)
    
    vn = np.array(vn) if len(vn)>0 else None
    vnf = np.array(vnf) if len(vnf)>0 else None
    if vnf is not None:
        vnf = (vnf- 1).astype(int)    
        
    md = ModelData(v,f,texcoord_uvw=vt, texcoord_faces=vtf)    
    
    if vn is not None and vnf is not None:
        normal = np.zeros_like(v)
        
        # use accumarray() to accumulate normals and then normalize them
        idx_vnf = vnf.flatten()
        idx_vf = f.flatten()
        for i in range(3):
            x = np.bincount(idx_vf, vn[idx_vnf,i])
            normal[:,i] = x
        normal = normal / np.linalg.norm(normal, axis=1).reshape((-1,1))    
        md._vertex_normals = normal
        md.update_face_normal_by_vertex_normal()
        
    
    #v,f,normal,texcoord = vio.read_mesh(obj_filename)
    #md = ModelData(v,f,texcoord_uvw=texcoord, texcoord_faces=f)
    
    #if normal is not None and len(normal) == len(v):
        #normal = normal / np.linalg.norm(normal, axis=1).reshape((-1,1))
        #md._vertex_normals = normal
        #md.update_face_normal_by_vertex_normal()
    
    if texture_filename is not None:
        img = imread(texture_filename)
        md._texture_image = img
    
    return md

def load_obj_single_mesh_v2(obj_filename, texture_filename = None):
    ''' load .obj file, where the file only contains one object
    
    Return
    ------------------------
    meshdata
        a ModelData object containing the mesh
    '''
    from igpy.modeling.modeldata import ModelData
    
    v = []
    f = []
    
    vt = []
    vtf = []
    
    vn = []
    vnf = []
    
    #normal = []
    #texcoord = []
    with open(obj_filename, 'r') as fid:
        for line in fid:
            line = line.strip()
            
            if line.startswith('v '):
                data = [float(x) for x in line.split()[1:]]
                v.append(data)
            elif line.startswith('vt '):
                data = [float(x) for x in line.split()[1:]]
                vt.append(data)                
            elif line.startswith('vn '):
                data = [float(x) for x in line.split()[1:]]
                vn.append(data)                
            elif line.startswith('f '):
                # do we have '/'?
                tx_faces = line.split()[1:]
                this_vf = []
                this_vtf = []
                this_vnf = []
                for x in tx_faces:
                    x_parts = x.split('/')
                    this_vf.append(int(x_parts[0]))
                    if len(x_parts) >= 2 and x_parts[1] != '':
                        this_vtf.append(int(x_parts[1]))
                    if len(x_parts) >= 3 and x_parts[2] != '':
                        this_vnf.append(int(x_parts[2]))
                
                f.append(this_vf)
                if len(this_vtf) > 0:
                    vtf.append(this_vtf)
                if len(this_vnf) > 0:
                    vnf.append(this_vnf)
            elif line.startswith('#'):
                pass  # Comment
            elif line.startswith('mtllib '):
                pass
            elif any(line.startswith(x) for x in ('g ', 's ', 'o ', 'usemtl ')):
                pass  # Ignore groups and smoothing groups, obj names, material
            elif not line.strip():
                pass
            else:
                logger.warning('Notice reading .OBJ: ignoring %s command.'
                               % line.strip())            
            
    v = np.array(v)
    f = (np.array(f)-1).astype(int)
    
    vt = np.array(vt) if len(vt)>0 else None
    vtf = np.array(vtf) if len(vtf)>0 else None
    if vtf is not None:
        vtf = (vtf - 1).astype(int)
    
    vn = np.array(vn) if len(vn)>0 else None
    vnf = np.array(vnf) if len(vnf)>0 else None
    if vnf is not None:
        vnf = (vnf- 1).astype(int)    
        
    md = ModelData(v,f,texcoord_uvw=vt, texcoord_faces=vtf)    
    
    if vn is not None and vnf is not None:
        normal = np.zeros_like(v)
        
        # use accumarray() to accumulate normals and then normalize them
        idx_vnf = vnf.flatten()
        idx_vf = f.flatten()
        for i in range(3):
            x = np.bincount(idx_vf, vn[idx_vnf,i])
            normal[:,i] = x
        normal = normal / np.linalg.norm(normal, axis=1).reshape((-1,1))    
        md._vertex_normals = normal
        md.update_face_normal_by_vertex_normal()
        
    
    #v,f,normal,texcoord = vio.read_mesh(obj_filename)
    #md = ModelData(v,f,texcoord_uvw=texcoord, texcoord_faces=f)
    
    #if normal is not None and len(normal) == len(v):
        #normal = normal / np.linalg.norm(normal, axis=1).reshape((-1,1))
        #md._vertex_normals = normal
        #md.update_face_normal_by_vertex_normal()
    
    if texture_filename is not None:
        img = imread(texture_filename)
        md._texture_image = img
    
    return md

def load_obj_single_mesh(obj_filename, texture_filename = None):
    ''' load .obj file, where the file only contains one object
    
    Return
    ------------------------
    meshdata
        a ModelData object containing the mesh
    '''
    import vispy.io as vio
    from igpy.modeling.modeldata import ModelData
    
    v,f,normal,texcoord = vio.read_mesh(obj_filename)
    md = ModelData(v,f,texcoord_uvw=texcoord, texcoord_faces=f)
    
    if normal is not None and len(normal) == len(v):
        normal = normal / np.linalg.norm(normal, axis=1).reshape((-1,1))
        md._vertex_normals = normal
        md.update_face_normal_by_vertex_normal()
    
    if texture_filename is not None:
        img = imread(texture_filename)
        md._texture_image = img
    
    return md

def save_obj(objfilename, vertices, faces, 
             vnormal_data = None, vnormal_faces = None,
             texcoord_data = None, texcoord_faces = None, 
             texture_image = None, vertex_color3f = None, 
             name = None):
    ''' save .obj file

    Parameters
    ---------------------------
    objfilename
        the .obj filename. If texture_path is provided, a .mtl file with the same name is created
    vertices
        nx3 vertex
    faces
        nx3 face array
    vnormal_data
        nx3 vertex normal data. This is the data to be indexed, 
        it may be more or less than the number of vertices
    vnormal_faces
        nx3 index into vnormal_data for each vertex in each face
    texcoord_data
        nx3 or nx2 texture coordinates, to be indexed by texcoord_faces
    texcoord_faces
        nx3 index into texcoord_data for each vertex in each face
    texture_image
        the texture image
    vertex_color3f
        vertex color for each vertex
    name
        name of this model
    '''
    
    faces = faces.astype(int)
    if vnormal_faces is not None:
        vnormal_faces = vnormal_faces.astype(int)
    if texcoord_faces is not None:
        texcoord_faces = texcoord_faces.astype(int)
        
    import os
    x, ext = os.path.splitext(objfilename)
    outdir, basename = os.path.split(x)
    if len(outdir) == 0:
        outdir ='.'
    fn_out_obj = objfilename
    if ext != '.obj':
        fn_out_obj = objfilename+'.obj'
    fn_out_mtl = outdir + '/' + basename + '.mtl'
    fn_out_texture = outdir + '/' + basename + '.png'
    
    texture_name = basename + '.png'
    mtl_name = os.path.split(fn_out_mtl)[-1]
    mtl_template = \
        '''
        newmtl lambert2
            Ns 1.0000
            Ni 1.5000
            d 1.0000
            Tr 0.0000
            Tf 1.0000 1.0000 1.0000 
            illum 2
            Ka 0.0000 0.0000 0.0000
            Kd 0.8000 0.8000 0.8000
            Ks 0.0000 0.0000 0.0000
            Ke 0.0000 0.0000 0.0000
            map_Ka {0}
            map_Kd {0}
        '''
    
    # create material file?
    if texture_image is not None:
        import skimage.io as sio
        with open(fn_out_mtl, 'w+') as fid:
            mtl_string = mtl_template.format(texture_name)
            fid.write(mtl_string)
        sio.imsave(fn_out_texture, texture_image)
    
    with open(fn_out_obj,'w+') as fid:
        # write material
        if texture_image is not None:
            fid.write('mtllib {0}\n\n'.format(mtl_name))
            
        # write vertices
        if vertex_color3f is None:
            for v in vertices:
                fid.write('v {0} {1} {2}\n'.format(v[0],v[1],v[2]))
        else:
            for i in range(len(vertices)):
                v = vertices[i]
                c = vertex_color3f[i]
                fid.write('v {0} {1} {2} {3} {4} {5}\n'.format(v[0],v[1],v[2],c[0],c[1],c[2]))
        fid.write('# {0} vertices\n\n'.format(len(vertices)))
        
        # write normals
        if vnormal_data is not None:
            for v in vnormal_data:
                fid.write('vn {0} {1} {2}\n'.format(v[0],v[1],v[2]))
            fid.write('# {0} vertex normals\n\n'.format(len(vnormal_data)))
            
        # write texture coordinates
        if texcoord_data is not None:
            #if texcoord_data.shape[-1] == 2:
                #tdata = np.column_stack((texcoord_data, np.zeros(len(texcoord_data))))
            #else:
                #tdata = texcoord_data
            tdata = texcoord_data
            for v in tdata:
                if len(v)==3:
                    fid.write('vt {0} {1} {2}\n'.format(v[0],v[1],v[2]))
                else:
                    fid.write('vt {0} {1}\n'.format(v[0],v[1]))
            fid.write('# {0} texture coordinates\n\n'.format(len(tdata)))
                
        # write faces
        if name is not None:
            fid.write('g {0}\n'.format(name))
        if texture_image is not None:
            fid.write('usemtl lambert2\n')
        for i in range(len(faces)):
            if vnormal_faces is not None:
                fmt = 'f {0}/{1}/{2} {3}/{4}/{5} {6}/{7}/{8}\n'
            elif texcoord_faces is not None:
                fmt = 'f {0}/{1}{2} {3}/{4}{5} {6}/{7}{8}\n'
            else:
                fmt = 'f {0}{1}{2} {3}{4}{5} {6}{7}{8}\n'
            f = faces[i]+1
            vt = ['','',''] if texcoord_faces is None else texcoord_faces[i]+1
            vn = ['','',''] if vnormal_faces is None else vnormal_faces[i]+1
            outstr = fmt.format(f[0],vt[0],vn[0],f[1],vt[1],vn[1],f[2],vt[2],vn[2])
            fid.write(outstr)
        fid.write('# {0} faces\n\n'.format(len(faces)))
        
# read exr file
def imread_exr(filename):
    ''' read exr file into a set of matrices
    
    return
    ---------------
    imgdict
        a dict of channels, for example imgdict['R'] is the R channel image
    '''
    import OpenEXR as exr
    import Imath
    
    xfile = exr.InputFile(filename)
    win = xfile.header()['dataWindow']
    minc, maxc = np.array([win.min.x, win.min.y]), np.array([win.max.x, win.max.y])
    width, height = maxc - minc + 1    
    
    # get all channels
    chdict= xfile.header()['channels']
    output = {}
    for key, val in chdict.items():
        raw_data = xfile.channel(key)
        if str(val.type) == 'FLOAT':
            img = np.frombuffer(raw_data, dtype=float32).reshape((height,width)).astype(float)
            output[key] = img
        elif str(val.type) == 'HALF':
            img = np.frombuffer(raw_data, dtype=float16).reshape((height,width)).astype(float)
            output[key] = img            
        else:
            assert False, 'only supports floating point images'
    return output

def imsave_exr(output_arr, filename, dtype = 'FLOAT',img_format = None):
    '''save array as .exr file
    parameters:
    --------------------
    dtype:
        FLOAT OR HALF, respons to float32 or float16 
        
    '''
    import OpenEXR
    import Imath
    import os
    import igpy.common.shortfunc as sf
    
    output_arr = np.atleast_3d(output_arr)
    n_channels = output_arr.shape[-1]
    w,h = output_arr.shape[:2]
    if img_format == None and n_channels<5:
        choices = {1:'R', 2: 'RG', 3:'RGB',4:'RGBA'}
        c_names = choices[n_channels]
    else:
        c_names = img_format
        
    n_channels = np.min((n_channels,len(c_names)))
    pixels_list = []
    for i in range(n_channels):
        if dtype == 'FLOAT':
            pixels = output_arr[:,:,i].astype(float32).tostring()
            pixels_list.append(pixels)
        elif dtype == 'HALF':
            pixels = output_arr[:,:,i].astype(float16).tostring()
            pixels_list.append(pixels)            
        else:
            return 
        
    HEADER = OpenEXR.Header(w,h)
    if dtype=='FLOAT':
        data_channel = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    else:
        data_channel = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        
    HEADER['channels'] = dict([(c,data_channel) for i,c in enumerate(c_names) if i < n_channels] ) 
    
    #check the filename
    path,name,ext = sf.fileparts(filename)
    if ext =='.exr':
        exr = OpenEXR.OutputFile(filename,HEADER)
        fin_pixels = dict([(c, pixels_list[i]) for i,c in enumerate(c_names) if i < n_channels] ) 
        exr.writePixels(fin_pixels)
        exr.close()
    else:
        print('extension is error')
        pass
        
def imread(filename, use_exif_orientation = False, is_binary_mask = False) -> np.ndarray:
    ''' read an image, return grayscale, RGB or RGBA image
    
    parameters
    ----------------
    filename
        the image file name
    use_exif_orientation
        whether we should rotate the image based on its exif data
    is_binary_mask
        is the image a binary mask? If True, the image will be converted to a binary mask where zero-value pixels are set to False. Alpha channel is ignored.
        
    return
    -----------
    img
        the read image
    '''
    import cv2
    import os
    
    assert os.path.exists(filename), '{0} does not exist'.format(filename)
    img : np.ndarray = None
    
    # use python's file reading procedure to support unicode path
    with open(filename, 'rb') as fid:
        content = fid.read()
        content = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
    if use_exif_orientation:
        from PIL import Image, ExifTags
        imgpil = Image.open(filename)
        if hasattr(imgpil,'_getexif'):
            # find orientation code
            name2code = {val:key for key, val in ExifTags.TAGS.items()}
            key_ori = name2code['Orientation']
            exdata = imgpil._getexif()
            if exdata is not None and key_ori in exdata.keys():
                ori_code = exdata[key_ori]
                
                if ori_code == 3:
                    img = np.rot90(img, 2)
                elif ori_code == 6:
                    img = np.rot90(img, 3)
                elif ori_code == 8:
                    img = np.rot90(img, 1)
                    
    if is_binary_mask:
        if np.ndim(img) == 2:
            img = img > 0
        elif np.ndim(img) == 3: #RGB or RGBA
            img = np.any(img[:,:,:3], axis=2)
    
    if np.ndim(img) == 3:
        if img.shape[-1] == 4:
            img = np.array(img[:,:,[2,1,0,3]])
        else:
            img = np.array(img[:,:,[2,1,0]])
            
    return np.ascontiguousarray(img)

def imwrite(filename, img):
    ''' write grayscale, RGB or RGBA image
    '''
    import cv2
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.dtype in (float, np.float16, np.float32):
        img = (img * 255).astype(np.uint8)
        
    if np.ndim(img) == 2:
        cv2.imwrite(filename, img)
    elif np.ndim(img) == 3:
        if img.shape[-1] == 3:
            cv2.imwrite(filename, img[:,:,::-1])
        elif img.shape[-1] == 4:
            cv2.imwrite(filename, img[:,:,[2,1,0,3]])
        else:
            assert False,'unsupported image'
    else:
        assert False,'unsupported image'
        
def read_cof(fn_cof):
    """Read  data from cof file.

    Parameters
    ----------
    fn_cof : str
        File name to read. Format will be inferred from the filename.
        Currently only '.cof' and '.cof.gz' are supported.

    Returns
    -------
    cof_data_dict: dict.'vertices','faces','normals','texcoords','BWIndext','BWValue'
        ,'vision','vertexCount','facesCount','reserved1','reserved2'
    
    
    vertices : array
        Vertices.
    faces : array 
        Triangle face definitions.
    normals : array
        Normals for the mesh.
    texcoords : array 
        Texture coordinates.
    BWIndext : array 
          
    BWValue : array 
      
    vision : float
    vertexCount : int
    facesCount: int 
    reserved1 : float 
    reserved2 : float      
    """    
    
    # Check format
    fmt = op.splitext(fn_cof)[1].lower()
    if fmt == '.gz':
        fmt = op.splitext(op.splitext(fn_cof)[0])[1].lower()

    if fmt in ('.cof'):
        cof_data_dict = {}
        
        v_list = []
        vn_list = []
        UV_list = []
        BWIndex_list = []
        BWValue_list = []
        f_list =[] 
        with open(fn_cof,"rb") as f:
            st=f.read(20)        
            vision,vertexCount,facesCount,reserved1,reserved2 = unpack('fiiff', 
                                                                           st)
        
            for i in range(vertexCount):
                st=f.read(12)
                v1,v2,v3 =  unpack('fff',st)            
                v_list.append([v1,v2,v3])
        
            for i in range(vertexCount):
                st=f.read(12)
                vn1,vn2,vn3 =  unpack('fff',st)            
                vn_list.append([vn1,vn2,vn3])       
        
            for i in range(vertexCount):
                st=f.read(8)
                v1,v2 =  unpack('ff',st)            
                UV_list.append([v1,v2])          
        
            for i in range(vertexCount):
                st=f.read(4)
                v1,v2,v3,v4 =  unpack('BBBB',st)            
                BWIndex_list.append([v1,v2,v3,v4]) 
        
                st=f.read(16)
                v1,v2,v3,v4 =  unpack('ffff',st)            
                BWValue_list.append([v1,v2,v3,v4])     
        
            for i in range(int(facesCount/3)):
                st=f.read(12)             
                f1,f2,f3 =  unpack('iii',st)  
                f_list.append([f1,f2,f3])        
        
        
            cof_data_dict['vertices'] = np.array(v_list)
            cof_data_dict['normals'] = np.array(vn_list)
            cof_data_dict['faces'] = np.array(f_list)
            cof_data_dict['texcoords'] = np.array(UV_list)
            cof_data_dict['BWIndext'] = np.array(BWIndex_list)
            cof_data_dict['BWValue'] = np.array(BWValue_list) 
            cof_data_dict['vision'] = vision
            cof_data_dict['vertexCount'] = vertexCount
            cof_data_dict['facesCount'] = facesCount
            cof_data_dict['reserved1'] = reserved1
            cof_data_dict['reserved2'] = reserved2
            return cof_data_dict
    elif not format:
        raise ValueError('read_cof needs could not determine format.')
    else:
        raise ValueError('read_cof does not understand format %s.' % fmt)   
    
    
    
def save_cof(fn_cof,cof_data_dict,is_mirror = True):
    """save_cof .cof.

    Parameters
    ----------
    fn_cof : str  
     cof_data_dict: dict.'vertices','faces','normals','texcoords','BWIndext','BWValue'
        ,'vision','vertexCount','facesCount','reserved1','reserved2'
    
    
    vertices : 
        Vertices.
    faces :  
        Triangle face definitions.
    normals : array
        Normals for the mesh.
    texcoords :  
        Texture coordinates.
    BWIndext :  
    
    BWValue :  
      
    vision : float
    vertexCount : int
    facesCount: int 
    reserved1 : float 
    reserved2 : float     

      
    """    
    
    # Check format
    fmt = op.splitext(fn_cof)[1].lower()
    if fmt == '.gz':
        fmt = op.splitext(op.splitext(fn_cof)[0])[1].lower()

    if fmt in ('.cof'):    
        vision = cof_data_dict['vision']       
        vertexCount = cof_data_dict['vertexCount']
        facesCount = cof_data_dict['facesCount']
        reserved1 = cof_data_dict['reserved1']
        reserved2 = cof_data_dict['reserved2']
        
        v= cof_data_dict['vertices'].astype(float32) 
        vn = cof_data_dict['normals'] 
        f = cof_data_dict['faces'] 
        UV = cof_data_dict['texcoords'] 
        BWIndex = cof_data_dict['BWIndext'] 
        BWValue = cof_data_dict['BWValue'] 
        #mirror cof
        if is_mirror:
            v[:,0] = -v[:,0]
            vn[:,0] = -vn[:,0] 
            
            f_temp = f[:,0].copy()
            f[:,0] = f[:,2]
            f[:,2] = f_temp            
        
        with open(fn_cof,"wb") as fo:
                st = pack('fiiff', vision,vertexCount,facesCount,reserved1,reserved2)
                fo.write(st)
            
                for i in range(vertexCount):
                    st=pack('fff', v[i][0], v[i][1], v[i][2])
                    fo.write(st)
            
                for i in range(vertexCount):
                    st=pack('fff', vn[i][0], vn[i][1], vn[i][2])
                    fo.write(st)  
            
                for i in range(vertexCount):
                    st=pack('ff', UV[i][0], UV[i][1])
                    fo.write(st)  
            
                for i in range(vertexCount):
                    st=pack('BBBB', BWIndex[i][0], BWIndex[i][1], BWIndex[i][2], BWIndex[i][3])
                    fo.write(st)                 
            
                    st=pack('ffff', BWValue[i][0], BWValue[i][1], BWValue[i][2], BWValue[i][3])
                    fo.write(st) 
            
                for i in range(int(facesCount/3)):
                    st=pack('iii', f[i][0], f[i][1], f[i][2])
                    fo.write(st)           
          
          
    elif not format:
        raise ValueError('save_cof needs could not determine format.')
    else:
        raise ValueError('save_cof does not understand format %s.' % fmt)            
    
    
def convert_json2cof(json_dir, cof_dir):   
    """
    convert a .json format file into a .cof format file
    
    Parameters:
    
    input: a json file 
    output a cof file
    """
    
    #read json file    
    import json
    from igpy.modeling.modeldata import ModelData
    with open(json_dir, 'r') as fid:
        jdata = json.load(fid)
     
    j_model_dict = {}
    bw_all_dict =  {} 
    parts_name = []
    for key, val in jdata.items():
        v = np.array(val['vertices'])
        f = np.array(val['triangles']).reshape((-1,3))
        vt = np.array(val['uv'])
        bw = np.array(val['boneWeights'])
        model = ModelData(v,f,name=key, texcoord_uvw=vt, texcoord_faces=f)
        j_model_dict[key] = model  
        bw_all_dict[key] = bw
        parts_name.append(key)
 
    #collect the total information                
    item_vert = []
    item_faces = []
    item_normal = []
    item_texcds = []
    item_bwidx = []
    item_bwval = []
    
    vtxcount = 0
    prevtxcount = 0
    facecount = 0
       
    for mname in parts_name:
        print(mname)
        prevtxcount = vtxcount
        item_vert.append( j_model_dict[mname]._vertices)
        vtxcount += j_model_dict[mname]._vertices.shape[0]
        
        curface = j_model_dict[mname]._faces + prevtxcount
        item_faces.append(curface)
        
        facecount += j_model_dict[mname]._faces.shape[0]
        
        item_normal.append(j_model_dict[mname]._vertex_normals)
        item_texcds.append(j_model_dict[mname]._texcoord_uvw)
        
        bidx_bw = bw_all_dict[mname] #array
        
        #add weights and index       
        for idx in range(len(bidx_bw)):
            widval = bidx_bw[idx]
            item_bwidx.append(widval['index'])
            item_bwval.append(widval['weight'])

              
    cof_data_dict = {}
    vision = 0.1
    reserved1 = 0.0
    reserved2 = 0.0
    
    fmt = op.splitext(cof_dir)[1].lower()
    if fmt == '.gz':
        fmt = op.splitext(op.splitext(cof_dir)[0])[1].lower()

    if fmt in ('.cof'):       
        v = np.asarray(item_vert)
        v = np.concatenate(v,axis = 0)      
        
        vn = np.asarray(item_normal)
        vn = np.concatenate(vn,axis = 0)
        
        f = np.asarray(item_faces)      
        f = np.concatenate(f,axis = 0);
        
        if False:
            import myplot.vtkplot as vp
            vp.trisurf(v, f, rendertype='wireframe',color3f = (0,1,0))  
            vp.show()
        
        UV = np.asarray(item_texcds)
        UV = np.concatenate(UV,axis = 0)
        
        BWIndex = np.asarray(item_bwidx) 
            
        BWValue = np.asarray(item_bwval)
        
        with open(cof_dir,"wb") as fo:
                st = pack('fiiff', vision,vtxcount, 3*facecount, reserved1,reserved2)
                fo.write(st)
            
                for i in range(vtxcount):
                    st=pack('fff', v[i][0], v[i][1], v[i][2])
                    fo.write(st)
            
                for i in range(vtxcount):
                    st=pack('fff', vn[i][0], vn[i][1], vn[i][2])
                    fo.write(st)  
            
                for i in range(vtxcount):
                    st=pack('ff', UV[i][0], UV[i][1])
                    fo.write(st)  
            
                for i in range(vtxcount):
                    st=pack('BBBB', BWIndex[i][0], BWIndex[i][1], BWIndex[i][2], BWIndex[i][3])
                    fo.write(st)                 
            
                    st=pack('ffff', BWValue[i][0], BWValue[i][1], BWValue[i][2], BWValue[i][3])
                    fo.write(st) 
            
                for i in range(int(facecount)):
                    st=pack('iii', f[i][0], f[i][1], f[i][2])
                    fo.write(st)           
          
          
    elif not format:
        raise ValueError('save_cof needs could not determine format.')
    else:
        raise ValueError('save_cof does not understand format %s.' % fmt)    
 
    
def convert_json2cof_haed(json_dir, cof_dir):   
    """
    convert a .json format file into a .cof format file
    
    Parameters:
    
    input: a json file 
    output a cof file
    """
    
    #read json file    
    import json
    from igpy.modeling.modeldata import ModelData
    with open(json_dir, 'r') as fid:
        jdata = json.load(fid)
     
    j_model_dict = {}
    bw_all_dict =  {} 
    parts_name = []
    for key, val in jdata.items():
        v = np.array(val['vertices'])
        f = np.array(val['triangles']).reshape((-1,3))
        vt = np.array(val['uv'])
        bw = np.array(val['boneWeights'])
        model = ModelData(v,f,name=key, texcoord_uvw=vt, texcoord_faces=f)
        j_model_dict[key] = model  
        bw_all_dict[key] = bw
        parts_name.append(key)
 
    #collect the total information                
    item_vert = []
    item_faces = []
    item_normal = []
    item_texcds = []
    item_bwidx = []
    item_bwval = []
    
    vtxcount = 0
    prevtxcount = 0
    facecount = 0
      
    parts_name = parts_name[0:1]   
    for mname in parts_name:
        print(mname)
        prevtxcount = vtxcount
        item_vert.append( j_model_dict[mname]._vertices)
        vtxcount += j_model_dict[mname]._vertices.shape[0]
        
        curface = j_model_dict[mname]._faces + prevtxcount
        item_faces.append(curface)
        
        facecount += j_model_dict[mname]._faces.shape[0]
        
        item_normal.append(j_model_dict[mname]._vertex_normals)
        item_texcds.append(j_model_dict[mname]._texcoord_uvw)
        
        bidx_bw = bw_all_dict[mname] #array
        
        #add weights and index       
        for idx in range(len(bidx_bw)):
            widval = bidx_bw[idx]
            item_bwidx.append(widval['index'])
            item_bwval.append(widval['weight'])

              
    cof_data_dict = {}
    vision = 0.1
    reserved1 = 0.0
    reserved2 = 0.0
    
    fmt = op.splitext(cof_dir)[1].lower()
    if fmt == '.gz':
        fmt = op.splitext(op.splitext(cof_dir)[0])[1].lower()

    if fmt in ('.cof'):       
        v = np.asarray(item_vert)
        v = np.concatenate(v,axis = 0)      
        
        vn = np.asarray(item_normal)
        vn = np.concatenate(vn,axis = 0)
        
        f = np.asarray(item_faces)      
        f = np.concatenate(f,axis = 0);
        
        if False:
            import myplot.vtkplot as vp
            vp.trisurf(v, f, rendertype='wireframe',color3f = (0,1,0))  
            vp.show()
        
        UV = np.asarray(item_texcds)
        UV = np.concatenate(UV,axis = 0)
        
        BWIndex = np.asarray(item_bwidx) 
            
        BWValue = np.asarray(item_bwval)
        
        with open(cof_dir,"wb") as fo:
                st = pack('fiiff', vision,vtxcount, 3*facecount, reserved1,reserved2)
                fo.write(st)
            
                for i in range(vtxcount):
                    st=pack('fff', v[i][0], v[i][1], v[i][2])
                    fo.write(st)
            
                for i in range(vtxcount):
                    st=pack('fff', vn[i][0], vn[i][1], vn[i][2])
                    fo.write(st)  
            
                for i in range(vtxcount):
                    st=pack('ff', UV[i][0], UV[i][1])
                    fo.write(st)  
            
                for i in range(vtxcount):
                    st=pack('BBBB', BWIndex[i][0], BWIndex[i][1], BWIndex[i][2], BWIndex[i][3])
                    fo.write(st)                 
            
                    st=pack('ffff', BWValue[i][0], BWValue[i][1], BWValue[i][2], BWValue[i][3])
                    fo.write(st) 
            
                for i in range(int(facecount)):
                    st=pack('iii', f[i][0], f[i][1], f[i][2])
                    fo.write(st)           
          
          
    elif not format:
        raise ValueError('save_cof needs could not determine format.')
    else:
        raise ValueError('save_cof does not understand format %s.' % fmt)    
    
def readPLY(filename):
    ''' load off file and return vertices and faces
    '''    
    import trimesh
    with open(filename,'rb') as fid:
        meshdata = trimesh.io.ply.load_ply(fid)
        meshdata['process'] = False
        meshobj = trimesh.Trimesh(**meshdata)
    return (meshobj.vertices, meshobj.faces)    

def load_ctm_single_model(fn_ctm, fn_texture = None):
    ''' load a ctm mesh as a single model
    
    Parameters
    -------------------
    fn_ctm
        filename of the ctm model
    fn_texture
        filename of the texture
        
    return
    ------------------
    model
        A ModelData of the ctm model
    '''
    import ctypes
    from igpy.modeling.modeldata import ModelData
    from . import openctm as ctm
    import os
    assert os.path.exists(fn_ctm), 'ctm file {0} does not exsit'.format(fn_ctm)
    
    # load ctm file
    fn = ctypes.c_char_p(fn_ctm.encode())
    con = ctm.ctmNewContext(ctm.CTM_IMPORT)
    ctm.ctmLoad(con, fn)
    
    # check format error
    err = ctm.ctmGetError(con)
    assert err == ctm.CTM_NONE, 'ctm file found but error occurs during loading'
    
    # get vertices
    n_vert = ctm.ctmGetInteger(con, ctm.CTM_VERTEX_COUNT)
    vertices = np.zeros((n_vert, 3))
    ctm_vert = ctm.ctmGetFloatArray(con, ctm.CTM_VERTICES)
    for i in range(n_vert*3):
        vertices.flat[i] = ctm_vert[i]
        
    # get faces
    n_face = ctm.ctmGetInteger(con, ctm.CTM_TRIANGLE_COUNT)
    faces = np.zeros((n_face,3), dtype=int)
    ctm_face = ctm.ctmGetIntegerArray(con, ctm.CTM_INDICES)
    for i in range(n_face*3):
        faces.flat[i] = ctm_face[i]
        
    # get uv
    n_uvmap = ctm.ctmGetInteger(con, ctm.CTM_UV_MAP_COUNT)
    if n_uvmap >=1:
        uv = np.zeros((n_vert,2))
        ctm_uv = ctm.ctmGetFloatArray(con, ctm.CTM_UV_MAP_1)
        for i in range(n_vert*2):
            uv.flat[i] = ctm_uv[i]
        uvface = faces
    else:
        uv = None
        uvface = None
        
    # get normal
    has_normal = bool(ctm.ctmGetInteger(con, ctm.CTM_HAS_NORMALS))
    if has_normal:
        normals = np.zeros_like(vertices)
        ctm_normals = ctm.ctmGetFloatArray(con, ctm.CTM_NORMALS)
        for i in range(n_vert*3):
            normals.flat[i] = ctm_normals[i]
    else:
        normals = None
        
    # create model
    if fn_texture is not None:
        img = imread(fn_texture)
    else:
        img = None
        
    model = ModelData(vertices, faces, texcoord_uvw=uv, texcoord_faces=uvface, texture_image=img)
    if normals is not None:
        model.set_vertex_normals(normals)    
        
    name = ctm.ctmGetString(con, ctm.CTM_NAME)
    if name is not None:
        model.name = name
    
    ctm.ctmFreeContext(con)
    return model

def load_single_mesh(fn_model, fn_texture = None):
    ''' load a file containing a single mesh into a ModelData
    
    Parameters
    --------------------
    fn_model
        the model file, can be .obj, .ctm
    fn_texture
        the texture image file
        
    Return
    -----------------
    model
        A ModelData containing the model
    '''
    from . import shortfunc as sf
    import os
    assert os.path.exists(fn_model), '{0} does not exist'.format(fn_model)
    if fn_texture is not None:
        os.path.exists(fn_texture), '{0} does not exist'.format(fn_texture)
        
    p, fn, ext = sf.fileparts(fn_model)
    if ext == '.ctm':
        model = load_ctm_single_model(fn_model, fn_texture)
    elif ext == '.obj':
        model = load_obj_single_mesh_v2(fn_model, fn_texture)
    else:
        assert False, 'model filename {0} is in wrong format'.format(fn_model) 
        
    return model

def copy_and_convert_image(fn_src, fn_dst):
    ''' copy image fn_src to fn_dst, with file type conversion
    
    Parameters
    -----------------------
    fn_src
        source file
    fn_dst
        destination file
    '''
    img = imread(fn_src)
    imwrite(fn_dst, img)
    
    
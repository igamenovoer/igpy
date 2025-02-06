# a class representing a 3D model for algorithmic processing only
import numpy as np
import trimesh
import igpy.common.shortfunc as sf

class ModelData(object):
    def __init__(self, vertex_local = None, faces = None, 
                 transmat = None, name = '', 
                 texcoord_uvw = None,
                 texcoord_faces = None,
                 texture_image = None,
                 vertex_color_data = None,
                 vertex_color_faces = None):
        ''' initialize the 3d model with nx3 vertex array and face array
        
        Parameters
        -------------------------
        vertex_local
            local coordinates of the vertices
        faces
            face array of the vertices
        transmat
            4x4 transformation matrix
        name
            name of the mesh
        texcoord_uvw
            uvw of texture coordinates, it is NOT related to vertex or face
        texcoord_faces
            index into texcoord_uvw, giving the texture coordinate of the vertices
            of each face. Let texcoord_faces[i]=(a,b,c), then the vertices
            v[faces[i],:] have texture coordinates texcoord_uvw[texcoord_faces[i],:].
            Note that texture coordinate is face-specific, a single vertex could 
            have multiple texture coordinates depending on the face in question.
        texture_image
            the texture image
        vertex_color_data
            nx3 or nx4 vertex color array, each row is a color
        vertex_color_faces
            nx3 index array, vertex_color_faces[i]=(a,b,c) iff the colors for the vertices of
            this triangle is vertex_color_data[a], [b], [c].
        '''
        self._vertices : np.ndarray = None
        self._faces : np.ndarray = None
        
        if vertex_local is not None:
            self._vertices = np.array(vertex_local)
            if faces is not None:
                self._faces = np.array(faces,dtype=int)
        
        if texcoord_uvw is not None:
            if texcoord_uvw.shape[-1] == 2:
                texcoord_uvw = np.column_stack((texcoord_uvw, np.zeros(len(texcoord_uvw))))
        self._texcoord_uvw : np.ndarray = texcoord_uvw
        self._texcoord_faces : np.ndarray = texcoord_faces
        self._texture_image : np.ndarray = texture_image

        self._vertex_color4f_data : np.ndarray = None #RGBA color data
        self._vertex_color_faces : np.ndarray = None
        self.set_vertex_colors(vertex_color_data, vertex_color_faces)
        
        # additional info attached to the texture
        # e.g., the texture might have several masks to define different areas
        self._texture_info : dict = {}
        
        if transmat is None:
            transmat = np.eye(4)
        else:
            assert np.allclose(transmat.shape,[4,4]), 'transmat must be 4x4'
            transmat = np.array(transmat)
        self._transmat : np.ndarray = transmat
        self.name = name
        
        # a networkx vertex graph
        self._vertex_graph = None
        if False:
            import networkx as nx
            self._vertex_graph = nx.Graph()
        
        # compute other things
        self._vertex_normals = None
        self._face_normals = None
        if vertex_local is not None:
            self._update_normals()
            
        # user data stored in a dict
        self._user_data = {}
        
    @property
    def transmat(self) -> np.ndarray:
        return self._transmat
    
    def set_user_data(self, name, data):
        self._user_data[name] = data
    
    def get_user_data(self, name):
        return self._user_data.get(name)
            
    def update_face_normal_by_vertex_normal(self):
        meshobj = trimesh.Trimesh(self._vertices, self._faces, vertex_normals=self._vertex_normals,
                                  process=False)
        self._face_normals = np.array(meshobj.face_normals)
        
    def unify_normals(self, is_outward = True):
        ''' unify all normals, so that they all points outward
        '''
        meshobj = trimesh.Trimesh(self._vertices, self._faces, process=False)
        meshobj.fix_normals()
        self._vertices = np.array(meshobj.vertices)
        self._faces = np.array(meshobj.faces)
        self._vertex_normals = np.array(meshobj.vertex_normals)
        self._face_normals = np.array(meshobj.face_normals)
        
    def __deepcopy__(self,memo):
        import copy
        cls = self.__class__
        newone = cls.__new__(cls)
        memo[id(self)] = newone
        for k,v in list(self.__dict__.items()):
            if k == 'q_mesh':
                newone.q_mesh = self.q_mesh
            else:
                setattr(newone, k, copy.deepcopy(v, memo = memo))
        return newone
        
    # build vertex graph
    def build_vertex_graph(self, edge_weight = 'hop'):
        ''' build vertex graph.
        
        parameter
        ----------------------
        edge_weight
            define what is used as edge weight, can be:
            
            None, no edge weight is defined
            
            'hop', each edge is weighted as 1.
            
            'L2', edge weight is the L2 distance between vertices.
            Distance is computed in global coordinate
            
        return
        -----------------------
        vertex_graph
            a networkx graph object
        '''
        import networkx as nx
        import igpy.common.shortfunc as sf
        faces = self.faces
        g = nx.Graph()
        g.add_nodes_from(np.arange(len(self._vertices)))
        edges = np.row_stack((faces[:,[0,1]], faces[:,[0,2]], faces[:,[1,2]]))
        edges = np.sort(edges, axis=1)
        edges = sf.unique_rows(edges)[0]
        
        if edge_weight is not None:
            if edge_weight.lower() == 'l2':
                vertices = self.vertices
                dist = np.linalg.norm(vertices[edges[:,0]] - vertices[edges[:,1]], axis=1)
            elif edge_weight.lower() == 'hop':
                dist = np.ones(len(edges))
                
            for i in range(len(edges)):
                u,v,d = edges[i,0], edges[i,1], dist[i]
                g.add_edge(u,v,weight=d)
        else:
            g.add_edges_from(edges)
        self._vertex_graph = g
        return g
        
    @property
    def vertex_graph(self):
        return self._vertex_graph
    
    def has_normal(self):
        return self._vertex_normals is not None
        
    # set 4x4 transformation matrix
    def set_transmat(self, transmat):
        self._transmat = transmat
        
    # apply a transmat over the existing transmat
    def apply_transmat(self, transmat):
        self._transmat = self._transmat.dot(transmat)
        
    # get information, prefer these instead of properties
    def get_vertex_normals(self, coordinate = 'global'):
        ''' get vertex normals
        
        coordinate = 'global' or 'local', for obtaining normals in different coordinate system
        '''
        normals = np.array(self._vertex_normals)
        if len(normals) == 0:
            return normals
        
        if coordinate == 'global':
            transmat = self._transmat
            normals = sf.transform_vectors(normals, transmat)
            normals = sf.normalize_rows(normals)[0]
        return normals
    
    def set_vertex_normals(self, vn, coordinate = 'global'):
        ''' set normal for each vertex
        
        parameter
        -------------------
        vn
            nx3 matrix, vn[i] is the normal for vertex i
        coordinate
            'global' or 'local', specify the coordinate system of vn
        '''
        normals = np.ascontiguousarray(np.atleast_2d(vn))
        
        if coordinate == 'global':
            transmat = self._transmat
            normals = sf.transform_vectors(normals, np.linalg.inv(transmat))
            
        nlens = np.linalg.norm(normals, axis=1)
        nlens[nlens == 0] = 1
        normals = normals / nlens.reshape((-1,1))        
        self._vertex_normals = normals
    
    def set_vertex_colors(self, vertex_colors, color_faces = None):
        ''' set vertex color for each vertex, the color can be 3f or 4f.
        
        parameter
        ---------------------
        vertex_colors
            nx3 or nx4 vertex color array, each row is a color, can be RGB or RGBA.
        color_faces
            nx3 index array, each row for a face, such that for face[i], its vertex colors
            are vertex_colors[color_faces[i]]. If not specified, color_faces will be
            the same as geometry faces, so the vertex_colors specify the color for each
            vertex.
        '''
        if vertex_colors is None:
            self._vertex_color4f_data = None
            self._vertex_color_faces = None
            return
        
        if color_faces is None:
            color_faces = np.array(self._faces)
        else:
            color_faces = np.array(color_faces)
        
        assert len(vertex_colors) == len(self._vertices),'should have the same number of colors as the vertices'
        assert vertex_colors.dtype == np.float or vertex_colors.dtype == np.float32, 'color should be floating point in [0,1]'
        if vertex_colors.shape[-1] == 3:
            vcolor = np.column_stack((vertex_colors, np.ones(len(vertex_colors))))
        elif vertex_colors.shape[-1] == 4:
            vcolor = np.array(vertex_colors)
        else:
            assert False,'color should be 3f or 4f'
        
        assert len(color_faces) == len(self._faces), 'each color face should correspond to one geometry face'
            
        self._vertex_color4f_data = vcolor
        self._vertex_color_faces = color_faces
    
    def set_texture_image(self, teximg):
        if teximg is not None:
            self._texture_image = np.array(teximg)
        else:
            self._texture_image = None
        
    def set_texture_mask(self, name, mask):
        ''' set a texture mask
        
        parameter
        ----------------
        name
            unique name of the texture mask
        mask
            the mask. If None, the mask of the given name will be erased
        '''
        assert type(name) == str, 'name must be a string'
        if mask is not None:
            self._texture_info[name] = mask
        else:
            self._texture_info.pop(name)
        
    def set_texture_info(self, name, info):
        ''' set a texture info
        
        parameter
        ----------------
        name
            unique name of the texture info
        info
            the info. If None, the info of the given name will be erased
        '''
        
        if info is not None:
            self._texture_info[name] = info
        else:
            self._texture_info.pop(name)
        
    def get_texture_info(self, name):
        ''' get texture infomation already set in the model.
        If the info does not exist, return None
        '''
        return self._texture_info.get(name)
        
    def get_texture_mask(self, name):
        return self._texture_info.get(name)
        
    @property
    def texture_maskset(self):
        return dict(self._texture_info)
    
    def set_texture_coordinate(self, texcoord_uvw, texcoord_faces):
        if texcoord_uvw.shape[-1] == 2:
            texcoord_uvw = np.column_stack((texcoord_uvw, np.zeros(len(texcoord_uvw))))
        self._texcoord_uvw = np.array(texcoord_uvw)
        self._texcoord_faces = np.array(texcoord_faces)
    
    
    def get_face_centers(self, coorindate = 'global'):
        ''' face centers in global or local coordinate
        '''
        if coorindate == 'global':
            v = self.vertices
        elif coorindate == 'local':
            v = self._vertices
        else:
            assert False, 'coorindate must be global or local'
            
        f = self._faces
        centers = np.squeeze(v[f].mean(axis=1))
        return centers
    
    def get_face_normals(self, coordinate = 'global'):
        ''' get face normals
        
        coordinate = 'global' or 'local', for obtaining normals in different coordinate system
        '''        
        normals = np.array(self._face_normals)
        if len(normals) == 0:
            return normals
        
        if coordinate == 'global':
            transmat = self._transmat
            normals = sf.transform_vectors(normals, transmat)
            normals = sf.normalize_rows(normals)[0]
        return normals        
        
    @property
    def vertices(self):
        ''' get global coordinate of the vertices
        '''
        if self._vertices is None:
            return None
        
        x = np.ones((len(self._vertices),4))
        x[:,:-1] = self._vertices
        y = x.dot(self._transmat)
        return y[:,:-1]
    
    @property
    def vertices_local(self):
        ''' get local coordinate of the vertices
        '''
        return np.array(self._vertices)
    
    @property
    def faces(self):
        ''' get the faces
        '''
        return np.array(self._faces)
    
    @property
    def edges(self):
        ''' get edges as nx2 matrix, pairs of vertex indices
        '''
        return np.row_stack((self._faces[:,[0,1]], self._faces[:,[0,2]], self._faces[:,[1,2]]))
    
    @property
    def aabb(self):
        ''' get the AABB bounding box of this mesh, return as (mincorner, maxcorner) in global coordinate
        '''
        vts = self.vertices
        minval, maxval = vts.min(axis=0), vts.max(axis=0)
        return np.array([minval, maxval])
        #center = (minval + maxval)/2
        #size = maxval - minval
        #return (center, size)

    def merge_edges(self, edges, in_place = True):
        ''' merge vertices on specified edges
        
        If in_place = True, the current mesh gets modified, otherwise a new mesh is returned
        '''        
        vts = self._vertices
        edges = np.array(edges)
        import igpy.common.shortfunc as sf
        edges = sf.vec1n_to_mat1n(edges)
        
        # make a graph out of the vertices
        import networkx as nx
        g= nx.Graph()
        g.add_edges_from(edges)
        
        # find connected vertices as subgraphs, each subgraph will be merged into one vertex
        node_merge_list = [list(x) for x in nx.connected_components(g)]
        
        # merge the nodes in a single connected component
        replace_dict = {} #replace_dictp[i]=j if node i will be replaced with j
        pts_append = [] #new vertices
        for ns in node_merge_list:
            # compute mean position of old nodes
            newpos = vts[ns].mean(axis=0)
            
            # append to current vertex list
            pts_append.append(newpos)
            
            # replace vertex
            for i in ns:
                replace_dict[i] = len(vts) + len(pts_append)-1
        
        # apply modification to the mesh
        if len(pts_append)>0:
            vts = np.row_stack((vts, pts_append)) #add vertices
            newfaces = sf.replace_values(self._faces, replace_dict) #replace vertices
                
            # remove useless vertices
            maskremove = np.zeros(len(vts), dtype=bool)
            for ns in node_merge_list:
                maskremove[ns]=True        
            idx_from = np.arange(len(vts)) #we need to update vertex index again, replacing idxfrom[i] with idxto[i]
            idx_from = idx_from[~maskremove]
            idx_to = np.arange(len(idx_from))
            tmp = {}
            for k in range(len(idx_from)):
                tmp[idx_from[k]] = idx_to[k]
            newfaces = sf.replace_values(newfaces, tmp)
            
            # remove vertices
            vts = vts[~maskremove]
            
            # validate the faces, remove invalid ones, which contains duplicated vertices
            maskinvalid = (newfaces[:,0]==newfaces[:,1]) | (newfaces[:,0]==newfaces[:,2]) | (newfaces[:,1]==newfaces[:,2])
            newfaces = newfaces[~maskinvalid]
        else:
            vts = self._vertices
            newfaces = self._faces
        
        # done, we have new mesh now
        if in_place:
            self._vertices = vts
            self._faces = newfaces
            return None
        else:
            obj = ModelData(vertex_local=vts, faces=newfaces, transmat=self._transmat,name=self.name)
            return obj        
    
    def merge_vertices_by_distance(self, distance, is_local = False, in_place = True):
        ''' merge neighboring vertices that are closer than a specified distance. 
        The distance is by default measured in global coordinate, 
        you can specify it in local coordinate by setting is_local=True
        
        If in_place = True, the current mesh gets modified, otherwise a new mesh is returned
        '''
        edges = [self._faces[:,[0,1]], self._faces[:,[0,2]], self._faces[:,[1,2]]]
        edges = np.row_stack(edges)        
        
        # compute edge lengths        
        if is_local:
            v = self._vertices            
        else:
            v = self.vertices
        edgelens = np.linalg.norm(v[edges[:,0]] - v[edges[:,1]], axis=1)
        
        # select edges that are going to be kept
        mask_keep = edgelens <= distance
        edges_use = edges[mask_keep]        
        return self.merge_edges(edges_use, in_place = in_place)
    
    def convert_distance(self, dist, input_coordinate = 'global'):
        ''' convert distance from global to local coordinate, and vice versa
        
        dist = a single value or a list of distance values, which
        will be transformed into distance in another coordinate system
        
        input_coordinate = 'global' if dist is measured in global coordinate, and
        output will be in local coordinate, otherwise just reverse
        '''
        p0 = np.array([0,0,0,1.])
        p1 = np.array([1,0,0,1.])
        if input_coordinate == 'global':
            transmat = np.linalg.inv(self._transmat)
        else:
            transmat = self._transmat
        p0 = p0.dot(transmat)
        p1 = p1.dot(transmat)
        dist_unit = np.linalg.norm(p0-p1)
        return np.array(dist)*dist_unit
    
    def _update_normals(self):
        obj = trimesh.base.Trimesh(self._vertices, self._faces, process=False)
        self._vertex_normals = np.array(obj.vertex_normals)
        self._face_normals = np.array(obj.face_normals)
    
    def _set_local_vertices(self, newvts):
        self._vertices = np.array(newvts)
        self._update_normals()
        
    @property
    def texture_image(self):
        if self._texture_image is not None:
            return np.array(self._texture_image)
        else:
            return None
        
    @property
    def is_texcoord_unique_for_vertex(self):
        ''' is each vertex only has one texture coordinate ?
        '''
        if self._texcoord_faces is None:
            return None
        
        idx_v = self._faces.flatten()
        idx_t = self._texcoord_faces.flatten()
        vtpair = np.column_stack((idx_v, idx_t))
        
        # uniqueness criteria: if idx_v[i] == idx_v[j], then idx_t[i]==idx_t[j]
        # we can create (idx_v[i],idx_t[i]) pairs and check uniqueness
        from igpy.common.shortfunc import unique_rows
        u = unique_rows(vtpair)[0]
        is_tex_unique = len(u[:,0]) == len(np.unique(u[:,0])) #idx_v should not have duplication
        
        return is_tex_unique
        
        #return np.allclose(self._faces.shape, self._texcoord_faces.shape) \
               #and np.allclose(self._faces, self._texcoord_faces)
    @property
    def has_texcoord(self):
        return self._texcoord_uvw is not None
    
    def get_color4f_per_vertex(self):
        if self._vertex_color4f_data is None:
            return None
        
        texf = self._vertex_color_faces.flatten()
        vt = self._vertex_color4f_data[texf]
        idxvts = self.faces.flatten()
        
        uvts, uidx = np.unique(idxvts, return_index=True)
        vt_output = np.zeros((len(self._vertices), 4))
        vt_output[uvts] = vt[uidx, :]
        return vt_output    
    
    @property
    def texcoord_uv(self):
        if self._texcoord_uvw is not None:
            return self._texcoord_uvw[:,:-1]
        else:
            return None
        
    @property
    def texcoord_faces(self):
        if self._texcoord_faces is not None:
            return self._texcoord_faces
        else:
            return None
    
    def get_texcoord_per_vertex(self, ndim = 2):
        ''' get the texture coordinate for each vertex.
        This assumes each vertex has a unique texture coordinate.
        Otherwise, an arbitrary one of the texture coordinate is returned
        
        parameter
        -------------------------
        ndim
            can be 2 or 3, for UV nad UVW coordinates respectively
        
        return
        -----------------------------
        texcoord
            texture coordinate for each vertex. None if texture coordinate
            does not exist.
        '''
        if self._texcoord_uvw is None:
            return None
        
        texf = self._texcoord_faces.flatten()
        vt = self._texcoord_uvw[texf]
        idxvts = self.faces.flatten()
        
        uvts, uidx = np.unique(idxvts, return_index=True)
        vt_output = np.zeros((len(self._vertices), ndim))
        vt_output[uvts] = vt[uidx, :ndim]
        return vt_output
    
    def save_as_obj(self, filename, 
                    with_normal = True, with_texture = True,
                    centerize = False):
        ''' save object into a .obj file and texture
        '''
        from igpy.common.inout import save_obj
        if with_normal:
            vnormal = self.get_vertex_normals()
            vnormal_face = self.faces
        else:
            vnormal = None
            vnormal_face = None
            
        if with_texture:
            teximg = self.texture_image
        else:
            teximg = None
            
        v = self.vertices
        if centerize:
            v = v - v.mean(axis=0)
        save_obj(filename, vertices=v, faces=self.faces,
                 vnormal_data=vnormal,
                 vnormal_faces=vnormal_face,
                 texcoord_data=self.texcoord_uv,
                 texcoord_faces=self.texcoord_faces,
                 texture_image=teximg,
                 name = self.name)
    
    def submesh_by_vertices(self, idxvert, outdict = None):
        ''' create a submesh by selecting some vertex.
        If the vertices of a face are ALL included in idxvert,
        then the face will exist in the submesh. Otherwise,
        the face will be discarded. As such, not all vertices
        listed in idxvert will exist in the submesh, only 
        those the fully specify a face will remain.
        
        parameter
        ------------------
        idxvert
            indices of vertices to exist in submesh
        outdict
            a dict() additional output
            
        return
        ------------------
        model_data
            a new ModelData containing only the selected vertices
        '''
        mask_face = np.in1d(self._faces, idxvert).reshape((-1,3))
        mask_face = mask_face.all(axis=1)
        idxface = np.nonzero(mask_face)[0]
        return self.submesh_by_faces(idxface, outdict=outdict)
    
    def set_vertices(self, vertices, coordinate = 'global'):
        ''' modify the vertex positions
        
        parameter
        ------------------
        vertices
            the new positions of the vertices
        coordinate
            the specified vertices is in which coordinate system? can be 'local' or 'global'            
        '''
        if coordinate == 'local':
            self._vertices = np.array(vertices)
        else:
            tmat = self._transmat
            v_aug = np.column_stack((vertices, np.ones(len(vertices))))
            v_local = sf.mldivide(tmat.T, v_aug.T).T[:,:-1]
            v_local = np.ascontiguousarray(v_local)
            self._vertices = v_local
    
    def submesh_by_faces(self, idxface, outdict = None):
        ''' create a submesh by selecting some face
        
        parameter
        -----------------------
        idxface
            indices of faces in selection
        outdict
            a dict() for storing additional output
            
        return
        -----------------------
        model_data
            a new ModelData containing only the selected faces
        '''
        from igpy.geometry.geometry import submesh_by_faces
        if outdict is None:
            outdict = {}
        
        # geometry vertex and face
        x = self._faces[idxface]
        idxvts, sub_face = np.unique(x, return_inverse=True)
        sub_face = sub_face.reshape((-1,3))
        sub_vertex = self._vertices[idxvts]
        outdict['idx_vertex'] = idxvts
        outdict['idx_face'] = idxface
        
        # texture vertex and face
        if self._texcoord_uvw is not None:
            x = self._texcoord_faces[idxface]
            idxvts, sub_tex_face = np.unique(x, return_inverse=True)
            sub_tex_face = sub_tex_face.reshape((-1,3))
            sub_tex_vertex = self._texcoord_uvw[idxvts]
            outdict['idx_texcoord_vertex'] = idxvts
        else:
            sub_tex_face = None
            sub_tex_vertex = None
            
        # vertex color
        if self._vertex_color4f_data is not None:
            x = self._vertex_color_faces[idxface]
            idxvts, sub_color_face = np.unique(x, return_inverse=True)
            sub_color_face = sub_color_face.reshape((-1,3))
            sub_color_vertex = self._vertex_color4f_data[idxvts]
            outdict['idx_color_vertex'] = idxvts
        else:
            sub_color_face = None
            sub_color_vertex = None
            
        # create mesh
        md = ModelData(sub_vertex, sub_face, transmat=self._transmat, 
                       name = self.name, 
                       texcoord_uvw=sub_tex_vertex,
                       texcoord_faces=sub_tex_face,
                       vertex_color_data=sub_color_vertex,
                       vertex_color_faces=sub_color_face,
                       texture_image=self._texture_image)
        md._texture_info = self._texture_info
        return md
            
    def get_unique_texcoord_mesh(self, outdict = None):
        ''' unstitch the mesh by texture coordinates, so that each piece of the mesh
        has unique texture coordinate for each vertex.
        
        return
        ---------------------
        list of model_data
            a list of ModelData, where each ModelData contains vertices that have unique
            texture coordinates
        '''
        
        from igpy.geometry.geometry import split_mesh
        if outdict is None:
            outdict= {}
        
        # split the mesh based on texture coordinate
        v = self._texcoord_uvw
        f = self._texcoord_faces
        idxfacelist = split_mesh(f)
        
        # get each submesh
        submeshlist = []
        outdict['submesh_info_list'] = []
        for idx_t_face in idxfacelist:
            _outdict = {}
            md = self.submesh_by_faces(idx_t_face, outdict = _outdict)
            outdict['submesh_info_list'].append(_outdict)
            submeshlist.append(md)
        return submeshlist
        
        
    def get_unstitched_model(self):
        ''' get vertices, faces, normals, texture coordinates such that
        the faces are all unstitched.
        
        Returns
        -----------------
        vertices
            the unstitched vertices
        faces
            the unstitched faces
        vertex_normals
            the normal of each vertex
        texture_coordinates
            UV texture coordinate for each vertex
        '''
        v_all = self.vertices
        f_all = self.faces
        
        f1, f2, f3 = f_all.T
        v1, v2, v3 = v_all[f1], v_all[f2], v_all[f3]
        vertices = np.row_stack((v1,v2,v3))
        ii = np.arange(len(f1))
        faces = np.column_stack((ii,ii + len(ii),ii + len(ii)*2))
        
        if self._texcoord_faces is not None:
            tf1, tf2, tf3 = self._texcoord_faces.T
            uvs = self._texcoord_uvw[:,:-1]
            vt1, vt2, vt3 = uvs[tf1], uvs[tf2], uvs[tf3]
            texcoord = np.row_stack((vt1,vt2,vt3))
        else:
            texcoord = None
            
        vn_all = self.get_vertex_normals()
        vn1,vn2,vn3 = vn_all[f1],vn_all[f2],vn_all[f3]
        normals = np.row_stack((vn1,vn2,vn3))
        
        #import myplot.vtkplot as vp
        #vp.trisurf(vertices, faces)
        #vp.show()
        
        return vertices, faces, normals, texcoord
        
        
    def move_vertex_along_normal(self, distance, idxvertex = None, coordinate = 'global', in_place = True):
        ''' extrude the vertices along normals by distance
        
        coordinate = the coordinate system where distance is measured
        
        in_place = True if this model will be modified, False if a new model should be returned
        
        idxvertex = indices of the vertices that will be moved. If not set, all vertices will be moved
        '''
        
        # convert distance to local
        if coordinate == 'local':
            dist_local = distance
        else:
            dist_local = self.convert_distance(distance, input_coordinate = 'global')
            
        # select vertex
        if idxvertex is None:
            idxvertex = np.arange(len(self._vertices))
        else:
            idxvertex = np.array(idxvertex).flatten()            
            
        # translate the vertices
        vnormal = self.get_vertex_normals(coordinate='local')
        vts = np.array(self._vertices)
        vts[idxvertex] = self._vertices[idxvertex] + vnormal[idxvertex] * dist_local
        
        if in_place:
            # update vertices
            self._set_local_vertices(vts)
            return None
        else:
            # make new object
            obj = ModelData(vts, self._faces, self._transmat, name = self.name)
            return obj
        
    def clone(self):
        ''' return a copy of this mesh.
        '''
        return ModelData.copy(self)
        
    @staticmethod
    def copy(meshobj):
        ''' copy a modeldata object
        '''
        import copy
        props_not_copy = set(['q_mesh','_vertex_graph'])
        obj = ModelData()
        for key, val in vars(meshobj).items():
            if key not in props_not_copy:
                x = copy.deepcopy(val)
                setattr(obj, key, x)
        return obj
        #obj = ModelData(meshobj._vertices, meshobj._faces, meshobj.transmat, name = meshobj.name)
        #return obj
        
   
        
        
        
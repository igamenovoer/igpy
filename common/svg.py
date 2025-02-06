import numpy as np
import lxml.etree as et
from . import shortfunc as sf

def hex_to_rgb(hex_string:str) -> np.ndarray:
    ''' convert hex color code into (r,g,b) format
    '''
    h = hex_string.lstrip('#')
    color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return np.array(color)

class ShapeBase:
    def __init__(self):
        # RGB color in (r,g,b), uint8 format
        self.color3u : np.ndarray = np.zeros(3, dtype=np.uint8)
        self.name : str = None
        
        self.m_transmat : np.ndarray = np.eye(3)
        
    def set_transmat(self, tmat):
        assert np.allclose(tmat.shape, (3,3)), 'transmat must be 3x3'
        self.m_transmat = np.array(tmat)        

class Rectangle(ShapeBase):
    def __init__(self, x=0, y=0, width=0, height=0):
        super().__init__()
        self.m_x = x
        self.m_y = y
        self.m_width = width
        self.m_height = height
        
    def set_by_xywh(self, x,y, width, height):
        self.m_x = x
        self.m_y = y
        self.m_width = width
        self.m_height = height
        
    def set_scale_wrt_origin(self, x_scale, y_scale):
        self.m_transmat[0,0] = x_scale
        self.m_transmat[1,1] = y_scale
        
    def set_translate(self, dx, dy):
        self.m_transmat[-1,:-1] = (dx, dy)
        
    def is_empty(self):
        if self.m_x is None or self.m_y is None or self.m_width is None or self.m_height is None:
            return True
        else:
            return False
        
    @property
    def xywh(self):
        ''' get (x,y,width,height)
        '''
        if self.is_empty():
            return None
        else:
            p = np.array([self.m_x, self.m_y])
            p = sf.transform_points(p, self.m_transmat).flatten()
            width = self.m_width * self.m_transmat[0,0]
            height = self.m_height * self.m_transmat[1,1]
            return np.array((p[0],p[1],width,height))
        
class Polyline(ShapeBase):
    def __init__(self, points : np.ndarray = None):
        super().__init__()
        self.m_points = np.atleast_2d(points)
        self.m_transmat : np.ndarray = np.eye(3)
        
    def is_empty(self):
        return self.m_points is None or len(self.m_points)==0
    
    def set_by_points(self, points : np.ndarray):
        self.m_points = np.atleast_2d(points)
        
    @property
    def num_points(self):
        ''' number of points in the polyline
        '''
        if self.m_points is None:
            return 0
        else:
            return len(self.m_points)
        
    @property
    def points(self):
        if self.is_empty():
            return None
        else:
            return sf.transform_points(self.m_points, self.m_transmat)
        
class Line(ShapeBase):
    def __init__(self):
        super().__init__()
        self.m_p0 : np.ndarray = None
        self.m_direction : np.ndarray = None
        self.m_transmat : np.ndarray = np.eye(3)
        
    def set_by_point_direction(self, p0, direction):
        self.m_p0 = np.array(p0).flatten()
        self.m_direction = np.array(direction).flatten()
        self.m_direction /= np.linalg.norm(self.m_direction)
        
    def set_by_points(self, p0, p1):
        self.m_p0 = np.array(p0).flatten()
        direction = np.array(p0).flatten() - np.array(p1).flatten()
        direction /= np.linalg.norm(direction)
        self.m_direction = direction
        
    def is_empty(self):
        return self.m_p0 is None or self.m_direction is None
        
    @property
    def p0(self):
        if self.is_empty():
            return None
        else:
            return sf.transform_points(self.m_p0, self.m_transmat).flatten()
        
    @property
    def direction(self):
        if self.is_empty():
            return None
        else:
            d = self.direction.dot(self.m_transmat[:2,:2])
            d /= np.linalg.norm(d)
            return d
        
class Polygon(ShapeBase):
    ''' polygon defined by a sequence of points.
    The polygon is automatically closed, that is, points[0] will connect to points[-1] by assumption.
    '''
    def __init__(self, points : np.ndarray = None):
        super().__init__()
        self.m_points : np.ndarray = points
        self.m_transmat : np.ndarray = np.eye(3)
        
    def set_by_points(self, points : np.ndarray):
        self.m_points = points
        
    def is_empty(self):
        return self.m_points is None or len(self.m_points) == 0
        
    @property
    def points(self):
        return sf.transform_points(self.m_points, self.m_transmat)

class SvgData:
    ''' svg objects read from svg file
    '''
    def __init__(self):
        self.polylines : list = None
        self.polygons : list = None
        self.rectangles : list = None
        
        # represent the canvas, that is, artboard in illustrator
        self.canvas_box : Rectangle = None
        
    def clear(self):
        self.polylines = []
        self.polygons = []
        self.rectangles = []
        self.canvas_box = None
        
    def set_canvas_box_and_transform(self, rect : Rectangle):
        ''' set a new canvas box, translate and scale the primitives so that
        their position and scale relative to the canvas is preserved.
        In order to use this function, self.canvas_box must not be None
        '''
        assert self.canvas_box is not None, 'no existing canvas box'
        import copy
        
        xywh_new = rect.xywh
        xywh_old = self.canvas_box.xywh
        
        # translate first and then scale
        dvec = xywh_new[:2] - xywh_old[:2]
        scale = xywh_new[2:]/xywh_old[2:]
        tmat_translate = np.eye(3)
        tmat_translate[-1,:2] = dvec
        tmat_scale = np.eye(3)
        tmat_scale[0,0] = scale[0]
        tmat_scale[1,1] = scale[1]
        tmat = tmat_translate.dot(tmat_scale)
        
        for obj in self.polygons:
            obj.set_transmat(tmat)
            
        for obj in self.polylines:
            obj.set_transmat(tmat)
            
        for obj in self.rectangles:
            obj.set_transmat(tmat)
            
        self.canvas_box = copy.deepcopy(rect)
    
    @staticmethod
    def init_with_svgfile(svg_filename, canvas_color3u = None):
        ''' parse svg to get geometry objects
        
        parameters
        ---------------
        svg_filename
            the name of the svg file
        canvas_color
            (r,g,b) of the rectangle that represents the canvas. If set, then you can identify
            the canvas box using this color and set canvas_box accordingly.
            If set, the canvas box will NOT be added to self.rectangles.
        '''
        import lxml.etree as et
        import webcolors
        with open(svg_filename) as fid:
            root : et._Element = et.fromstring(fid.read())
        remove_xml_namespace(root)
        
        output = SvgData()
        output.clear()
        
        def parse_color(color_desc : str) -> tuple:
            if '#' in color_desc:
                color = tuple(webcolors.hex_to_rgb(color_desc))
            else:
                color = tuple(webcolors.name_to_rgb(color_desc))  
            return color
        
        def parse_polylines(xml_root:et._Element):
            lines = xml_root.findall('polyline')
            output = []
            for obj in lines:
                _pts : str = obj.get('points')
                pts = np.fromstring(_pts, dtype=float, sep=' ').reshape((-1,2))
                pline = Polyline(points = pts)
                color = parse_color(obj.get('stroke'))
                pline.color3u = color
                output.append(pline)
            return output
        
        def parse_lines(xml_root:et._Element):
            # find the lines
            lines = xml_root.findall('line')
            output = []
            for obj in lines:
                coords = [obj.get('x1'), obj.get('y1'), obj.get('x2'), obj.get('y2')]
                x1, y1, x2, y2 = [float(x) for x in coords]
                color = parse_color(obj.get('stroke'))
                
                pline = Polyline(points = np.array([[x1,y1],[x2,y2]]))
                pline.color3u = color
                output.append(pline)
            return output
        
        def parse_polygons(xml_root:et._Element):
            # find polygons
            svg_polys = xml_root.findall('polygon')
            output = []
            for obj in svg_polys:
                pts = np.fromstring(obj.get('points'), sep=' ').reshape((-1,2))
                color = parse_color(obj.get('stroke'))
                polygon = Polygon(points = pts)
                polygon.color3u = color
                output.append(polygon)
            return output
        
        def parse_rectangles(xml_root:et._Element):
            # find rectangles
            svg_rects = xml_root.findall('rect')
            output = []
            for obj in svg_rects:
                x = float(obj.get('x'))
                y = float(obj.get('y'))
                width = float(obj.get('width'))
                height = float(obj.get('height'))
                color = parse_color(obj.get('stroke'))
                box = Rectangle(x,y,width,height)
                box.color3u = color
                output.append(box)
            return output
        
        def parse_group(xml_root:et._Element, output):
            # output is SvgData
            g = xml_root
            lines = parse_lines(g)
            output.polylines.extend(lines)
            
            polylines = parse_polylines(g)
            output.polylines.extend(polylines)            
            
            polygons = parse_polygons(g)
            output.polygons.extend(polygons)
            
            rects = parse_rectangles(g)
            output.rectangles.extend(rects)
            
        groups = root.findall('g')
        for g in groups:
            parse_group(g, output)
        parse_group(root, output)
        
        # find viewbox
        if canvas_color3u is not None:
            canvas_box : Rectangle = None
            for box in output.rectangles:
                if np.allclose(canvas_color3u, box.color3u):
                    canvas_box = box
                    
            if canvas_box is not None:
                output.canvas_box = box
                output.rectangles.remove(canvas_box)
                
        return output
        
def remove_xml_namespace(root : et.Element):
    ''' remove namespace in xml, nodes like {domain}ns:tag will now be tag only
    '''
    import lxml.objectify as objectify
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'): continue  # (1)
        i = elem.tag.find('}')
        if i >= 0:
            elem.tag = elem.tag[i+1:]
    objectify.deannotate(root, cleanup_namespaces=True)   
    
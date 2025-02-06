import numpy as np

def texture_points_to_uv(image_points_xy, image_shape, invert_y = True):
    ''' convert texture image points (x,y) into texture coordinates uv
    
    parameter
    ----------------------
    image_points_xy
        points in texture image, nx2 points in (x,y) format
    image_shape
        shape of the image, got directly by image.shape, in (height,width,channel) format
    invert_y
        should we invert y axis? Some software uses image upper-left corner as origin,
        others use lower-left as origin, so sometimes you may need to invert y axis if
        later is the case.
        
    return
    --------------------
    uv
        uv texture coordinates
    '''
    width, height = image_shape[1], image_shape[0]
    xs = image_points_xy[:,0]
    u = xs/(width-1)
    
    ys = image_points_xy[:,1]
    if invert_y:
        v = 1-ys/(height-1)
    else:
        v = ys/(height-1)
    
    uv = np.column_stack((u,v))
    return uv

def uv_to_texture_points(uv, image_shape, invert_y = True):
    ''' convert uv to points in texture image
    
    parameter
    ----------------------
    uv
        nx2 points of uv coordinates
    image_shape
        shape of the texture
    invert_y
        should we invert y axis? Some software uses image upper-left corner as origin,
        others use lower-left as origin, so sometimes you may need to invert y axis
        
    return
    ----------------------
    image_points_xy
        (x,y) coordinates in texture image, for each point in uv
    '''
    width, height = image_shape[1], image_shape[0]
    xs = uv[:,0] * (width-1)
    if invert_y:
        ys = (1-uv[:,1])* (height-1)
    else:
        ys = uv[:,1] * (height-1)
    pts = np.column_stack((xs,ys))
    return pts
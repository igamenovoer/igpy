import numpy as np

def bresenham_3(p1 : np.ndarray, p2 : np.ndarray) -> np.ndarray:
    ''' Bresenham's line drawing algorithm in 3D, from p1 to p2
    
    parameters
    ---------------
    p1 : np.ndarray[int]
        source point
    p2 : np.ndarray[int]
        target point
        
    return 
    ------------
    pts : np.ndarray
        points on the line from p1 to p2
    '''
    x0,y0,z0 = p1.astype(int)
    x1,y1,z1 = p2.astype(int)
    dx = abs(x1-x0)
    sx = 1 if x0<x1 else -1
    dy = abs(y1-y0)
    sy = 1 if y0<y1 else -1
    dz = abs(z1-z0)
    sz = 1 if z0<z1 else -1
    dm = max(dx, dy, dz)
    i = dm
    x1 = y1 = z1 = i // 2
    
    output = np.zeros((dm+1, 3), dtype=int)

    while i >= 0:
        output[dm-i] = [x0, y0, z0]
        x1 -= dx
        if x1 < 0:
            x1 += dm
            x0 += sx
        y1 -= dy
        if y1 < 0:
            y1 += dm
            y0 += sy
        z1 -= dz
        if z1 < 0:
            z1 += dm
            z0 += sz
        i -= 1
    
    return output

def bresenham_2(p1 : np.ndarray, p2 : np.ndarray) -> np.ndarray:
    ''' Bresenham's line drawing algorithm in 3D, from p1 to p2
    
    parameters
    ---------------
    p1 : np.ndarray[int]
        source point
    p2 : np.ndarray[int]
        target point
        
    return 
    ------------
    pts : np.ndarray
        points on the line from p1 to p2
    '''
    x0,y0 = p1.astype(int)
    x1,y1 = p2.astype(int)
    
    dx = abs(x1-x0)
    sx = 1 if x0<x1 else -1
    dy = -abs(y1-y0)
    sy = 1 if y0<y1 else -1
    err = dx + dy

    dmax = np.max([abs(dx), abs(dy)])+1
    output = np.zeros((dmax, 2), dtype=int)
    i = 0
    while True:
        output[i] = (x0, y0)
        i += 1
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return output
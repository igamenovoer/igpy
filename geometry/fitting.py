# functions related to fitting geometry
#from .calcbase import *
import numpy as np
import cv2
from subprocess import Popen, PIPE
#from . import closestPoint
from sklearn.decomposition import PCA
import scipy.interpolate as itp

def _fitCircle(pts2d):
    pts2d = pts2d.astype('float32')
    return cv2.minEnclosingCircle(pts2d)

def fitCircle(pts2d, outdict = None):
    ret = cv2.fitEllipse(pts2d.astype('float32'))
    center = ret[0]
    ab = ret[1]
    radius = np.mean(ab)/2
    
    if outdict is not None:
        dvec = pts2d - np.array(center)
        distlist = np.linalg.norm(dvec,axis=1) - radius
        outdict['distlist'] = distlist
        
    return (center, radius)

def fitEllipse_AxisAligned(pts2d, outdict = None):
    '''fit axis aligned ellipse
    return (center,ab)
    
    if outdict is provided, outdict['distlist'] = distance from each point to the ellipse
    '''
    
    # guess the ellipse center using opencv's fitellipse
    #ret = cv2.fitEllipse(pts2d.astype('float32'))
    #center = np.array(ret[0])
    center = (pts2d.max(axis=0) + pts2d.min(axis=0))/2
    
    # formulate a least squares problem
    # (x-x0)^2/a^2+(y-y0)^2/b^2=1 for all (x,y) in pts2d
    dx2 = (pts2d[:,0] - center[0])**2
    dy2 = (pts2d[:,1] - center[1])**2
    A = np.column_stack((dx2,dy2))
    b = np.ones((len(pts2d),1))
    
    #sol = np.linalg.lstsq(A,b)[0]
    sol = scipy.optimize.nnls(A,b.flatten())[0]
    sol[sol<1e-3] = 1e-3
    ab = 1/np.sqrt(sol)
    assert np.any(np.isnan(ab)) == False,'ab has nan'
    
    # compute point to ellipse distance
    if outdict is not None:
        tt = np.linspace(0,2*np.pi,NPTS_POLY)
        xs = np.cos(tt) * ab[0] + center[0]
        ys = np.sin(tt) * ab[1] + center[1]
        ptspoly = np.column_stack((xs,ys))
        distlist = closestPoint.dist2poly2d(pts2d, ptspoly)
        outdict['distlist'] = distlist
    
    return (center,ab)

def fitLine(pts, line_dir = None, outdict = None):
    ''' fit a line to a list of points
    line_dir = specified line direction, it will be determined automatically if not provided
    outdict = a dict() containing additional output
    outdict['distlist'] = distance from pts[i] to the line
    outdict['segment'] = (p1,p2), the finite length line segment
    
    return (p0, line_direction)
    '''
    line_dir = np.array(line_dir).flatten()
    if line_dir is None:
        pts = np.array(pts)
        ptsmean = pts - pts.mean(axis=0)
        [u,s,v] = np.linalg.svd(ptsmean.T.dot(ptsmean))
        line_dir = v[0]
    line_dir = np.array(line_dir)
    p0 = pts.mean(axis=0)
    
    if outdict is not None:
        ptsproj = projectPointsToLine(pts,p0,line_dir)
        dvec = pts - ptsproj
        distlist = np.linalg.norm(dvec,axis=0)
        outdict['distlist'] = distlist
        
        dotvals = (pts - p0).dot(line_dir)
        idxmin = np.argmin(dotvals)
        idxmax = np.argmax(dotvals)
        p1 = p0 + dotvals[idxmin] * line_dir
        p2 = p0 + dotvals[idxmax] * line_dir
        outdict['segment'] = (p1,p2)
        
    return (p0,line_dir)

def fitEllipseOnPlane_AxisAligned(pts3d, a_dir, b_dir, outdict = None):
    ''' fit an ellipse whose a,b axis directions are given by a_dir and b_dir
        the points should be on a plane
        return (center, ab)

        outdict is a dict() for storing additional output,
        if provided, outdict['distlist'] = distance from each point to the ellipse
    '''
    
    # make a frame first
    a_dir = np.array(a_dir)
    b_dir = np.array(b_dir)
    normal = np.cross(a_dir,b_dir)
    
    a_dir /= np.linalg.norm(a_dir)
    b_dir /= np.linalg.norm(b_dir)
    normal /= np.linalg.norm(normal)
    rotmat = np.row_stack((a_dir,b_dir,normal))
    position = pts3d[0]
    
    # transform points to local frame
    pts2d = (pts3d - position).dot(np.linalg.inv(rotmat))
    pts2d = pts2d[:,:-1]
    
    # fit ellipse
    c2,ab = fitEllipse_AxisAligned(pts2d, outdict = outdict)
    
    # transform back to original
    center = np.append(c2,0).dot(rotmat) + position
    return (center,ab)

# fit circle to 2d points, given as nx2 numpy matrix
# return (center,radius)
def __fitCircle(pts2d):
    x = pts2d[:,0].flatten()
    y = pts2d[:,1].flatten()
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m
    
    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)
    
    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)
    
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    
    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    
    return ((xc_1,yc_1),R_1)

# fit a rectangle to a list of points
# return ((center_x,center_y), (width, height), angle(rad)), angle is CCW rotation from x+ axis
def _fitRectangle(pts2d):
    '''
    fit a rectangle to a list of points
    return ((center_x,center_y), (width, height), angle(rad)), angle is CCW rotation from x+ axis
    '''
    pts = pts2d.astype('float32')
    center, size, angle = cv2.fitEllipse(pts) #angle is degree
    angle = np.deg2rad(angle)
    return (center, size, angle)

def fitPlane(pts3d):
    ''' fit a plane to a set of 3d points
    
    return (p0, normal)
    '''
    p0 = pts3d.mean(axis=0)
    ptsmean = pts3d - p0
    [u,s,v] = np.linalg.svd(ptsmean.T.dot(ptsmean))
    normal = v[-1]
    return (p0, normal)

def fit_rectangle_min_area_2(pts2d):
    ''' fit a rectangle to a list of points by minimum area
    return ((center_x,center_y), (width, height), (width_dir, height_dir))
    '''
    pts = pts2d.astype('float32')
    center, size, angle = cv2.minAreaRect(pts)
    angle = np.deg2rad(angle)
    wdir = np.array([np.cos(angle), np.sin(angle)])
    hdir = np.array([-np.sin(angle), np.cos(angle)])
    return (center, size, (wdir, hdir))

def fit_rectangle_min_area_3(pts3d):
    ''' fit a rectangle to a list of points by minimum area
    return ((center_x,center_y), (width, height), (width_dir, height_dir))
    '''
    _, normal = fitPlane(pts3d)
    p0 = pts3d.mean(axis=0)
    pltrans = PlaneTransform(p0,normal)
    pts2d = pltrans.to_2d(pts3d)
    
    center_2, size, whdir_2 = fit_rectangle_min_area_2(pts2d)
    center = pltrans.to_3d(center_2)
    whdir = (pltrans.to_dir_3d(whdir_2[0]).flatten(), pltrans.to_dir_3d(whdir_2[1]).flatten())
    return (center, size, whdir)
    

def fitRectangle(pts2d, outdict = None):
    '''
    fit a rectangle to a list of points
    return ((center_x,center_y), (width, height), angle(rad)), angle is CCW rotation from x+ axis
    
    outdict['distlist'] = distance from pts2d[i] to the nearest poin
    '''
    pts = pts2d.astype('float32')
    center, size, angle = cv2.minAreaRect(pts)
    angle = np.deg2rad(angle)
    
    if outdict is not None:
        wdir = np.array([np.cos(angle), np.sin(angle)])
        hdir = np.array([-np.sin(angle), np.cos(angle)])
        
        dw = wdir * size[0]
        dh = hdir * size[1]
        c = np.array(center)
        p0 = c + dw/2 + dh/2
        p1 = p0 - dh
        p2 = p1 - dw
        p3 = p2 + dh
        ptsrect = np.array([p0,p1,p2,p3,p0])
        distlist = closestPoint.dist2poly2d(pts2d, ptsrect)
        outdict['distlist'] = distlist
    
    return (center, size, angle)

def fitRectangle_AxisAligned(pts2d, outdict = None):
    '''
    fit an axis-aligned rectangle to a set of points
    return ((center_x, center_y), (width,height))
    
    outdict is a dict() to store additional output.
    if provided, outdict['distlist'] = distance from each point to the rectangle
    '''
    
    # use iterative method to find the 4 edges
    n_iter = 10
    
    # initialize 4 lines, two horizontal, two vertical
    x0, y0 = pts2d.min(axis=0) #min corner
    x1, y1 = pts2d.max(axis=0) #max corner
    for i in range(n_iter):
        # soft assign points to lines
        
        # distance to vertical lines
        d0 = pts2d - np.array([x0,y0]).reshape((1,2))
        
        # distance to horizontal lines
        d1 = pts2d - np.array([x1,y1]).reshape((1,2))
        
        # concatenate distance
        dxy = np.column_stack((d0,d1))
        dxy = np.abs(dxy)
        
        # compute hard assignment
        # for each row, select the min and assign 1 to the place
        wxy = (dxy == dxy.min(axis=1).reshape((-1,1))).astype('float')
        
        # for each column, select the min and assign 1 to the place
        # this makes sure that each line has at least one point assigned
        wxy_sure = (dxy == dxy.min(axis=0).reshape((1,-1))).astype('float')
        wxy += wxy_sure
        
        # normalize for all points assigned to each line
        wxy = wxy / wxy.sum(axis=0).reshape((1,-1))
        
        # recompute lines
        x0,y0,x1,y1 = (wxy * np.column_stack((pts2d,pts2d))).sum(axis=0)
        
    # just to make sure we have a valid rectangle
    xmin = min(x0,x1)
    xmax = max(x0,x1)
    ymin = min(y0,y1)
    ymax = max(y0,y1)
    center = np.array([xmax+xmin, ymax+ymin])/2
    size = np.array([xmax-xmin,ymax-ymin])

    if outdict is not None:
        ptspoly = [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)]
        distlist = closestPoint.dist2poly2d(pts2d, ptspoly)
        outdict['distlist'] = distlist
    
    return (center,size)

def fitRectangle_AutoAxis(pts2d, outdict = None):
    ''' fit a 2d rectangle using axis-aligned rectangle fitting,
    but will first automatically find the axis directions
    
    return (center, (width, height), (width_dir, height_dir))
    
    outdict['distlist'] = distance from pts3d[i] to the fitted rectangle
    '''
    _,_,angle = fitRectangle(pts2d)
    
    # width and height direction
    w_dir = np.array([np.cos(angle), np.sin(angle)])
    h_dir = np.array([-np.sin(angle), np.cos(angle)])
    
    # use w_dir and h_dir as new frame
    rotmat = np.row_stack((w_dir, h_dir))
    p0 = pts2d.mean(axis=0)
    transmat = makeTransmat(rotmat, p0)
    pts2d_local = transformPoints(pts2d, np.linalg.inv(transmat))
    center, size = fitRectangle_AxisAligned(pts2d_local, outdict=outdict)
    
    return (center, size, (w_dir, h_dir))

def fitRectangleOnPlane_AutoAxis(pts3d, normal = None, outdict = None):
    '''fit a rectangle on a plane using axis-aligned rectangle fitting,
    but will first automatically find the axis directions
    
    return (center, (width, height), (width_dir, height_dir))
    
    outdict['distlist'] = distance from pts3d[i] to the fitted rectangle
    '''
    # find plane normal if not provided
    if normal is None:
        _, normal = fitPlane(pts3d)
        
    p0 = pts3d.mean(axis=0)
    pltrans = PlaneTransform(p0, normal)
    pts2d = pltrans.to_2d(pts3d)
    
    # fit rectangle
    center2d, size, whdir2d = fitRectangle_AutoAxis(pts2d, outdict = outdict)
    whdir3d = (pltrans.to_dir_3d(whdir2d[0]).flatten(), pltrans.to_dir_3d(whdir2d[1]).flatten())
    center3d = pltrans.to_3d(center2d)
    
    return (center3d, size, (whdir3d[0], whdir3d[1]))

def fitRectangleOnPlane_AxisAligned(pts3d, width_dir, height_dir, outdict = None):
    '''
    fit a rectangle to a set of points which all lie on a plane
    width_dir and height_dir are directions of width and height of the rectangle    
    
    return (center,size) where center is (x,y,z) and size is (width,height)
    if outdict is provided, outdict['distlist'] = distance from each point to the rectangle
    '''
    # make a local frame and convert the points into local coordinate
    pts3d = np.array(pts3d)
    x = np.array(width_dir)
    x = x/np.linalg.norm(x)
    y = height_dir
    y = y/np.linalg.norm(y)
    z = np.cross(x,y)
    z = z/np.linalg.norm(z)
    
    rotmat = np.row_stack((x,y,z))
    position = pts3d[0]
    pts3d -= position
    pts2d = pts3d.dot(np.linalg.inv(rotmat))[:,:-1]
    center, size = fitRectangle_AxisAligned(pts2d, outdict = outdict)
    
    # convert back to 3d coordinate
    center = np.append(center,0).dot(rotmat) + position
    return (center,size)

# fit rectangle on a 3d plane, given points on the plane
def fitRectangleOnPlane(pts3d, normal, outdict = None):
    '''fit rectangle on a 3d plane, given points on the plane
    
    normal = the normal of the plane
    
    return ((center_x, center_y, center_z), (width, height), (width_dir, height_dir))
    
    outdict['distlist'] = distance from pts3d[i] to the rectangle
    '''
    # 2d coordinates of the 3d points on the plane
    (z,udir,vdir) = makeFrame(normal)
    u = pts3d.dot(udir)
    v = pts3d.dot(vdir)
    ptslocal = np.vstack((u,v)).transpose()    
    
    center, size, angle = fitRectangle(ptslocal, outdict = outdict)
    
    # convert back to 3d
    zdist = pts3d[0].dot(normal)
    c3d = center[0] * udir + center[1] * vdir + zdist * normal
    width_dir = np.cos(angle) * udir + np.sin(angle) * vdir
    height_dir = - np.sin(angle) * udir + np.cos(angle) * vdir
    return (c3d, size, (width_dir, height_dir))

# fit a circle to a list of 3d points on a plane
def fitCircleOnPlane(pts3d, normal, outdict = None):
    '''fit a circle to a list of 3d points on a plane
    
    pts3d = nx3 numpy matrix of 3d points, the must be on a plane
    
    normal = normal of the plane that contains pts3d
    
    outdict = a dict() containing additional outputs
    outdict['distlist'] = distance from pts3d[i] to the nearest point on the circle
    
    return (center_3d, radius)
    '''

    # 2d coordinates of the 3d points on the plane
    (z,udir,vdir) = makeFrame(normal)
    u = pts3d.dot(udir)
    v = pts3d.dot(vdir)
    ptslocal = np.vstack((u,v)).transpose()
    
    # fit circle
    (center,radius) = fitCircle(ptslocal, outdict = outdict)
    
    # recover center of circle
    zdist = pts3d[0].dot(normal)
    c3d = center[0]*udir + center[1]*vdir + zdist * normal
    
    return (c3d, radius)

def fitSplineAdaptive(pts, tt, tnew, ndegree = 3, 
                      min_error_reduction_ratio = 1e-2, 
                      max_control_pts = None,
                      outdict = None):
    ''' adaptively fit a spline
    
    tt[i] is the parameter value of pts[i]
    
    tnew[i] is the new parameter value for sampling
    
    outdict['n_control_pts'] = number of control points used
    outdict['error'] = the final error
    
    return the newly sampled points from the fitted curve
    '''
    import meshproc
    from . import curveproc as cp
    n_minctrl = ndegree + 2
    
    if max_control_pts is None:
        max_control_pts = max(len(pts)/2, ndegree+2)
    
    n_ctrl = n_minctrl    
    prev_err = -1
    while True:
        # control point within range?
        if n_ctrl >= max_control_pts:
            break
        
        # fit a spline
        ptsnew = meshproc.fit_spline(pts, tt, tt, ndegree, n_ctrl)
        
        # find rms error
        err = np.mean((pts - ptsnew)**2)
        
        # error reduction ok?
        if prev_err >= 0:
            ratio = (prev_err - err)/prev_err
            if ratio >=0 and ratio < min_error_reduction_ratio:
                # cannot reduce error further, just stop here
                break
            
        # move on
        n_ctrl += 1
        prev_err = err
        
    if outdict is not None:
        outdict['error'] = prev_err
        outdict['n_control_pts'] = n_ctrl
        
    # sample again
    output = meshproc.fit_spline(pts, tt, tnew, ndegree, n_ctrl)
    return output
            

# fit a spline to pts, and return n samples from the fitted curve
def fitSpline(pts, nsample):
    # compute parameter value for the points
    dvec = pts[1:] - pts[:-1]
    seglens = np.linalg.norm(dvec,axis=1)
    tt = np.cumsum(seglens)
    tt = np.insert(tt,0,0) / tt[-1]
    
    #prepare to call exe
    npts = len(pts)
    ndim = pts.shape[1]
    
    strlist = [str(ndim),str(npts)]
    
    # convert points [t x y z ...] to string
    ptdata = np.column_stack((tt,pts)).flatten()
    ptdata_str = ' '.join(list(ptdata.astype('|S8')))
    strlist.append(ptdata_str)
    
    # output parameters
    tout = np.linspace(0,1,nsample)
    tout_str = ' '.join(list(tout.astype('|S8')))
    strlist.append(tout_str)
    
    # make command and execute
    cmdin = ' '.join(strlist)
    exefile = 'curvefit/curvefit.exe'
    proc = Popen(exefile,stdin=PIPE, stdout=PIPE)
    cmdout = proc.communicate(cmdin)[0]
    cmdout = cmdout.replace('\r','').replace('\n','')
    
    # convert data back
    ptsout = np.fromstring(cmdout,dtype=float,sep=' ')
    ptsout = ptsout.reshape((-1,ndim))
    return ptsout

def findOOB(pts2d, method = 'ellipse'):
    ''' find an oriented bounding box for a set of 2d points.
    
    method = which method is used to compute the bounding box, which can be:
    
    'ellipse' = fit an ellipse to the points
    
    'pca' = fit PCA to the points
    
    'minarea' = use minimum area bounding box
    
    return (center, size, angle), where size is the (width,height) of the box, angle is the
    angle of width edge in rad
    '''
    
    # compute angle of x axis
    if method == 'ellipse':
        _, _, angle = cv2.fitEllipse(pts2d.astype('float32')) #angle is degree
        angle = np.deg2rad(angle)
    elif method == 'pca':
        pca = PCA()
        pca.fit(pts2d)
        xdir = pca.components_[0]
        cosval = xdir[1]/np.linalg.norm(xdir)
        angle = np.arccos(np.clip(cosval,-1,1))
    elif method == 'minarea':
        pts = pts2d.astype('float32')
        _, _, angle = cv2.minAreaRect(pts)
        angle = np.deg2rad(angle)
    
    # compute center and size
    c = np.cos(angle)
    s = np.sin(angle)
    rotmat = np.array([[c,s],[-s,c]])
    pts = pts2d.dot(np.linalg.inv(rotmat)) #use rotated frame as reference frame        
    size = pts.max(axis=0) - pts.min(axis=0)
    center = (pts.max(axis=0) + pts.min(axis=0))/2
    center = center.dot(rotmat) #back to the original frame
    return (center, size, angle)
        
# fit 1d polynomial with derivative constraint
def fit_polynomial_1d(x, y, order, ws = None, dervs = None):
    ''' fit a polynomial, possibly with derivative constraint
    
    fit y=f(x) where f(x) is a <order>-degree polynomial.
    For example, if order=3, then y=a_3*x^3+a_2*x^2+a_1*x+a_0
    
    dervs = a list of derivatives. dervs[i] is the (i+1)-order derivative of each point.
    We have len(dervs[i])==len(x), if dervs[i][j]=nan, then jth point has no derivative constraint
    at order i+1. If dervs[i]=None or [], then there is no i+1 order derivative constraint
    
    return a poly1d object
    '''
    
    #assert ws is None, 'weighting not implemented yet'
    if ws is None:
        ws = np.ones(len(x))
    ws = np.array(ws)
    assert len(ws) == len(x)
    
    # need to sqrt the weight because the cost is sum of squares
    ws_sqrt = np.sqrt(ws)

    x = np.array(x)
    y = np.array(y)
    xpower = [] #xpower[i] = x**i
    for i in range(order+1):
        v = x**i
        xpower.append(v)
    xpower = np.column_stack(xpower)
    
    coeflist = []
    blist = []
    
    # fit position
    coef_0dg = xpower * ws_sqrt.reshape((-1,1))
    b_0dg = y * ws_sqrt
    coeflist.append(coef_0dg)
    blist.append(y)
    
    # fit derivatives
    pfunc = np.poly1d(np.ones(order+1))
    if dervs is not None:
        for i in range(len(dervs)):
            ds = dervs[i]
            if ds is None or len(ds)==0:
                continue
            
            maskuse = ~np.isnan(ds)
            dfunc = pfunc.deriv(i+1) #a fast way to get coefficients of ith derivative
            if np.all(dfunc.coeffs == 0):
                break
            
            coef_idg = np.zeros((maskuse.sum(), order+1))
            n = len(dfunc.coeffs)
            coef_idg[:,i+1:] = xpower[maskuse,:order-i] * dfunc.coeffs[::-1]
            b_idg = ds[maskuse]
            
            coef_idg *= ws_sqrt[maskuse].reshape((-1,1))
            b_idg *= ws_sqrt[maskuse]
            
            coeflist.append(coef_idg)
            blist.append(b_idg)
    
    # formulate and solve
    coef = np.row_stack(coeflist)
    b = np.concatenate(blist)
    
    import igpy.common.shortfunc as sf
    cs = sf.mldivide(coef, b)
    output = np.poly1d(cs[::-1])
    return output
            
# fit a spline to a set of points
# tol = tolerance, between 0 and 1
# pts can be nx2 or nx3 matrix
# return a spline parameter spl, where you can sample its points using splev(tt,splout)
# tt is in (0,1) range
def _fitSpline(pts,kdegree = 5, tol = 0.02):
    '''
    # example:
    tt = np.linspace(0,5*np.pi,100)
    x = np.cos(tt)*3 + tt + np.random.random(tt.shape)
    y = np.sin(tt)*3
    pts = np.c_[x,y]
    spl = fitSpline(pts)
    
    u = np.linspace(0,1,100)
    newpts = itp.splev(u,spl)
    
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.plot(newpts[0],newpts[1],'r-')
    plt.axis('equal')
    '''    
    
    # compute error metric
    pts = np.array(pts)
    pca = PCA()
    pca.fit(pts)
    sval = np.sqrt(np.mean(pca.explained_variance_))
    tt = np.linspace(0,1,len(pts))
    ws = np.zeros(tt.shape) + 1/sval
    #ws[0] = ws[0]*100
    #ws[-1] = ws[-1]*100
    
    # fit spline
    spladv,tmp = itp.splprep(pts.T, u=tt, w=ws, k = kdegree, s=tol * len(pts))
    return spladv
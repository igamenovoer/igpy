import numpy as np
import scipy.interpolate as itp

# interpolate polyline and get intermediate points
class PolylineEquidistantSampler:
    def __init__(self):
        self.m_interp : itp.BSpline = None
        
    def init_by_pts(self, pts):
        ''' initialize the interpolator by a list of D-dimensional points, given in NxD matrix
        '''
        
        # parameterize the polyline by arclen in (0,1)
        seglen = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        ts = np.zeros(len(pts))
        ts[1:] = np.cumsum(seglen)
        ts /= ts[-1]
        
        self.m_interp = itp.make_interp_spline(ts, pts, 1)
        
    def sample(self, t):
        ''' sample points based on arclen parameter, which is in unit range
        
        return
        ----------
        pts
            NxK points where N=len(t)
        '''
        pts = self.m_interp(t)
        return pts
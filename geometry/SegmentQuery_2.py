# perform 2d segment query
import meshproc
import numpy as np

class SegmentQuery_2(object):
    def __init__(self, pts1 = None, pts2 = None):
        self._sq = meshproc.SegmentQuery()
        
        if (not pts1 is None) and (not pts2 is None):
            self.set_data(pts1,pts2)
        
    def set_data(self,pts1,pts2):
        ''' initialize query structure with segments defined as (pts1[i],pts2[i])
        '''
        pts1 = self._check_and_mod_pts2d(pts1)
        pts2 = self._check_and_mod_pts2d(pts2)
        assert len(pts1) == len(pts2)
            
        # append 0 to 3d
        pts1 = np.column_stack((pts1, np.zeros(len(pts1))))
        pts2 = np.column_stack((pts2, np.zeros(len(pts2))))
        self._sq.set_data(pts1,pts2)
        
    def set_data_polyline(self, pts):
        ''' initialize query structure with a polyline, where each segment of the polyline
        is used as a segment in the query structure
        '''
        pts1 = pts[:-1]
        pts2 = pts[1:]
        self.set_data(pts1,pts2)
        
    def _check_and_mod_pts2d(self, pts):
        ''' check dimension of points, and make sure the points are defined as row vectors
        '''
        pts = np.array(pts)
        if len(pts.shape)==1:
            pts = pts.reshape((-1,2))
        else:
            assert pts.shape[1]==2, 'only accept 2d points'
        return pts
    
    def closest_points(self, pts):
        ''' find closest points for pts
        
        return (nnpts, idxsegs), where nnpts[i] is the closest point of pts[i]
        to a segment whose index is idxsegs[i]
        '''
        pts = self._check_and_mod_pts2d(pts)
        pts = np.column_stack((pts, np.zeros(len(pts))))
        nnpts, idxsegs = self._sq.closest_points(pts)
        
        # remove 3d
        nnpts = nnpts[:,:-1]
        return (nnpts, idxsegs)
    
    def get_segments(self):
        ''' return the original segments
        '''
        pts1, pts2 = self._sq.get_segments()
        pts1 = pts1[:,:-1]
        pts2 = pts2[:,:-1]
        return (pts1,pts2)
    
    def intersect_by_line(self, p0, ldir):
        ''' intersect the segments with a line
        
        return (pts, idxsegs) where pts[i] is an intersection point on segment idxsegs[i]
        '''
        p0 = self._check_and_mod_pts2d(p0)
        ldir = self._check_and_mod_pts2d(ldir)
        
        # make to 3d
        p0 = np.append(p0,0)
        ldir = np.append(ldir,0)
        
        nnpts, idxsegs = self._sq.intersect_by_line(p0, ldir)
        nnpts = nnpts[:,:-1]
        return (nnpts, idxsegs)
    
    def intersect_by_segment(self, p1, p2):
        ''' intersect the segments with a 2d segment
        
        return (pts, idxsegs) where pts[i] is an intersection point on segment idxsegs[i]
        '''
        p1 = self._check_and_mod_pts2d(p1)
        p2 = self._check_and_mod_pts2d(p2)
        
        p1 = np.append(p1,0)
        p2 = np.append(p2,0)
        
        nnpts, idxsegs = self._sq.intersect_by_segment(p1,p2)
        nnpts = nnpts[:,:-1]
        return (nnpts, idxsegs)
    
        
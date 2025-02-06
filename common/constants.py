# default values
# objects for the algoritm
#from panda3d import core
import numpy as np
LINE_COLOR = (1,0,0,1)
LINE_THICKNESS = 3

# given a polyline, make a line segment L using its two end points, 
# if the distance between all other points and L, divided by |L|, is smaller than this ratio
# then the polyline is considered a straight line
THRES_LINE_MIDIST_TO_ENDIST_RATIO = 0.02

# number of points to sample when converting a shape into a polygon for distance computation
NPTS_POLY = 30

# if two vectors' incident angle is smaller than this, they are considered parallel
THRES_ANGLE_VECTOR_PARALLEL = np.deg2rad(10)

# when clustering normals to identify majority, how many clusters to create
NCLUST_MAJORITY_DIRECTION = 5

# default parameters for SAM06 corner detection method
SAM06_NPTS_TEST_RATIO = 0.1
SAM06_NPTS_TEST_MAX = 20
SAM06_NPTS_HALF_MERGE_RATIO = SAM06_NPTS_TEST_RATIO/2
SAM06_NPTS_HALF_MERGE_MAX = int(SAM06_NPTS_TEST_MAX/2)
SAM06_DIST_RATIO = 0.02 #peak-to-line distance / diaglen

# sharpness detection parameter
SHARPNESS_NPTS_MIN_RATIO = 0.02
SHARPNESS_NPTS_MAX_RATIO = SAM06_NPTS_TEST_RATIO
SHARPNESS_NPTS_MIN = 3
SHARPNESS_NPTS_MAX = 10
SHARPNESS_MAX_ANGLE = np.deg2rad(120)

# line/curve segmentation
SEGCURVE_THRES_SHARPNESS = 0.75 #a keypoint must be over this sharpness to be considered as a sharp corner
SEGCURVE_CORNER_DIST_RATIO = 0.07 #used to detect corner within a segment of curve

# number of proposals generated after merging them
N_PROPOSAL_OUTPUT = 3

# thresholds
THRES_MAX_NORMCOST_INPLANE_DISTORT = 0.15 # cost is considered small if below this
THRES_MAX_COST_SHARE_CENTER_WITH_FC =  0.15 # less than this, the new shape share center with feature curve

# how many points to sample from the constraint feature curve, when evaluting curve-to-curve distance?
NPTS_SAMPLE_FROM_FC = 30
NPTS_SAMPLE_FROM_STROKE = 30 #how many points to sample from stroke?

THRES_UPPERCOST_SHAPE_IS_THE_SAME = 0.1 # upper bound of the cost to consider shapes as the same
THRES_SHAPE_SIMILAR_SCALE_RANGE = 0.05 # when the stroke is determined to be similar to a shape, the scale can be (fixscale * (1 +- val))
THRES_UPPERCOST_SHAPE_AT_POSITION = 0.2 # if shape's center distance to a ponit if close enough, shape is considered positioned there
THRES_UPPERCOST_SHAPE_ON_A_LINE =  0.2 # if shape's center distance to a line is within this thres, it is considered OK
THRES_UPPERCOST_SHAPE_INPLANE_DISTROTION = 0.8 # if inplane distortion is within this cost, it is considered OK

# if the angle of two planes are within this range, they are considered parallel
THRES_ANGLE_PLANE_PARALLEL = np.deg2rad(10.0)

#if two parallel planes are this close, they are considered the same
THRES_DIST2DIAG_PLANE_IDENTICAL = 0.05

# normalized region size
LINE_REGION_WIDTH = 0.25
LINE_REGION_LENGTH = 0.25

# error code return by position solving from proposals
ERR_OK = 0
ERR_UNDER_CONSTRAINED = 1
ERR_INCOMPATIBLE_CONSTRAINT = 2

# prority of proposals
PRO_FIXED_CURVE = 1 # this is a general curve
PRO_ON_PLANE = 1
PRO_ON_LINE = 1
PRO_FIXED_POSITION = 1 #the shape is at a fixed position
PRO_SIMILAR_SHAPE = 2
PRO_SAME_SHAPE = 3

# maximum number of solutions derived from constraints
N_MAX_TRY = 20
N_TRY_PER_PROPOSAL = 3

# shape type strings
SPNAME_ELLIPSE = 'ellipse > circle' # > is 'generalized type of'
SPNAME_CIRCLE = 'circle'
SPNAME_RECTANGLE = 'rectangle > square'
SPNAME_SQUARE = 'rectangle'
SPNAME_LINE = 'line'
SPNAME_POLYLINE = 'polyline' # general curve
SPNAME_NOT_SET = 'none'

# sketch segment type
SKTYPE_LINE_SEGMENT = 'line segment'
SKTYPE_CURVE = 'curve'

# keypoint distance type
KPTYPE_SECTION = 'section distance'
KPTYPE_MODEL = 'model distance'

# number of continous contour in order to be considered as stable group
N_CONTOUR_AS_STABLE = 4

# maximum voxel grid length
MAX_VOXEL_DIM = 1000
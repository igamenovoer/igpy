# responsible for plotting image-related things
import cv2
import numpy as np


def to_uint8(img: np.ndarray):
    """convert float image to uint8 image"""
    if img.dtype in (np.float64, np.float32, np.float16):
        return (img * 255).astype(np.uint8)
    elif img.dtype == np.uint8:
        return np.array(img)
    elif img.dtype == int:
        return img.astype(np.uint8)
    elif img.dtype in [np.uint16, np.uint32]:
        x = img.astype(float) / np.iinfo(img.dtype).max * 255
        return x.astype(np.uint8)
    elif img.dtype == bool:
        return img.astype(np.uint8) * 255
    else:
        raise "error type"


def imshow(img, winname=None, allow_resize=True, wait_key=True, destroy_on_exit=True):
    import uuid

    if winname is None:
        winname = str(uuid.uuid4())

    if img.dtype == bool:
        img = img.astype("uint8") * 255

    # only display RGB
    if np.ndim(img) == 3 and img.shape[-1] > 3:
        img = img[:, :, :3]

    if allow_resize:
        # import skimage.viewer as skview
        # skview.ImageViewer(img).show()
        cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    else:
        cv2.namedWindow(winname)

    # convert rgb into bgr for opencv display
    if len(img.shape) == 3:
        img = img[:, :, ::-1]

    cv2.imshow(winname, img)
    if wait_key:
        kval = cv2.waitKey(0)

    if destroy_on_exit:
        cv2.destroyAllWindows()


def imtool(img):
    import skimage.viewer as skview

    skview.ImageViewer(img).show()


def draw_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    line_width: int = 1,
    color3f=None,
    is_closed=False,
    return_copy=False,
):

    if img.dtype != np.uint8 or return_copy:
        img = to_uint8(img)

    if color3f is None:
        color3f = np.zeros(3)
    # color = tuple([int(x*255) for x in color3f[::-1]])  # opencv assumes BGR image
    color = tuple([int(x * 255) for x in color3f])
    cv2.polylines(img, [pts.astype(int)], is_closed, color, int(line_width))
    return img


def draw_rectangle(
    img: np.ndarray,
    xywh: np.ndarray,
    line_width: int = 1,
    color3f=None,
    return_copy=False,
):
    """draw a single rectangle

    parameters
    ----------------
    img
        the canvas to be drawn, in uint8 format
    xywh
        x,y,width,height of the rectangle, in image coordinate
    line_width
        number of pixels, the width of the line
    color3f
        color of the rectangle
    return_copy
        if True, do not modify the input image, but return a copy. Otherwise,
        modify the input image and return a copy
    """
    if color3f is None:
        color3f = np.zeros(3)
    # color = tuple([int(x*255) for x in color3f[::-1]])  # opencv assumes BGR image
    color = tuple([int(x * 255) for x in color3f])

    if return_copy:
        img = img.copy()

    xywh = np.atleast_1d(xywh).flatten().astype(int)
    x, y, w, h = xywh

    # cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), color, line_width)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, line_width)
    return img


def draw_mask_overlay(img: np.ndarray, mask: np.ndarray):
    """overlay a mask onto rgb or gray image

    return
    ---------
    img_ov
        the overlayed image
    """
    h, w = img.shape[:2]

    if img.ndim == 2:
        output: np.ndarray = np.dstack([img] * 3)
    else:
        output: np.ndarray = np.array(img)

    mask = mask > 0
    if output.dtype == np.uint8:
        output[:, :, 0] = mask.astype(np.uint8) * 255
    elif output.dtype in (np.float64, np.float32, np.float16):
        output[:, :, 0] = mask.astype(output.dtype)

    return output


def draw_calibration_grid(img, pts, pattern_size_wh):
    """draw calibration grid points (opencv chessboard or symmetric circle pattern) onto the image

    parameters
    --------------
    img
        the image over which the calibration pattern is to be drawn
    pts
        xy points of the calibration pattern, can be Nx1x2 (opencv format) or Nx2 points
    pattern_size_wh
        (n_points_per_row, n_points_per_col)

    return
    ------------
    imgnew
        image with pattern drawn
    """
    canvas: np.ndarray = to_uint8(img)
    if canvas.ndim == 2:  # gray scale image, convert to rgb
        canvas = np.dstack([canvas] * 3)
    output = cv2.drawChessboardCorners(
        canvas, tuple(pattern_size_wh), np.reshape(pts, (-1, 1, 2)), True
    )
    return output


def draw_matches(img_left, pts_left, img_right, pts_right):
    """draw key points matches between two images

    parameters
    ---------------
    img_left, img_right
        left and right images
    pts_left, pts_right
        xy points in left and right images, one-to-one correspondence

    return
    -------------
    img
        an image with matches drawn on it
    """
    pts_1 = []
    pts_2 = []
    for p1, p2 in zip(pts_left, pts_right):
        q1 = cv2.KeyPoint()
        q1.pt = tuple(p1)
        pts_1.append(q1)

        q2 = cv2.KeyPoint()
        q2.pt = tuple(p2)
        pts_2.append(q2)

    dmatches = []
    for i in range(len(pts_1)):
        d = cv2.DMatch(i, i, 0)
        dmatches.append(d)

    canvas = cv2.drawMatches(img_left, pts_1, img_right, pts_2, dmatches, None)
    return canvas


def draw_keypoints(img, cv_keypoints: list, color3f=None, return_copy=False):
    """draw opencv keypoints

    parameters
    ----------------
    img
        the canvas
    cv_keypoints
        a list of cv2.KeyPoint
    color3f
        color of the key points
    return_copy
        if True, return a copy of the image, otherwise draw on the image directly
    """
    canvas = to_uint8(img)
    if color3f is None:
        color = (-1, -1, -1)
    else:
        color = np.atleast_1d(color3f) * 255
        color = tuple(color.astype(int))
    cv2.drawKeypoints(
        canvas, cv_keypoints, canvas, color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return canvas


def draw_markers(
    img,
    pts_xy,
    color3f=None,
    scale=1.0,
    marker_type="+",
    return_copy=False,
    line_width=1.0,
):
    """draw markers in an image

    parameters
    -----------------
    img
        the image onto which the marker is drawn,
        can be RGB, RGBA or single channel image, can be uint8 or float
    pts_xy
        marker centers, points in xy coordinate, should be integers
    color3f
        the marker color in rgb format, range in (0,1)
    return_copy
        if True, a copy of img will be made with the markers drawn on it.
        if False, the markers will be drawn on img, which must be rgb image
    marker_type
        a character representing the marker type, see marker types section

    marker types
    --------------------
    'v'
        triangle down
    '+'
        cross
    's'
        square
    'd'
        diamond
    'x'
        tilted cross

    return
    ---------------
    img
        the image which has the markers drawn on it
    """
    from ..common import image_processing as ip
    import cv2

    if color3f is None:
        color3f = (0, 0, 0)

    canvas = img
    if return_copy:
        canvas = np.array(img)

    assert np.ndim(canvas) == 3, "canvas must be RGB image"
    if canvas.dtype != np.uint8:
        canvas = ip.to_uint8(canvas)

    pts_xy = np.atleast_2d(pts_xy)
    pts_xy = pts_xy.astype(int)
    line_width = int(line_width)

    mktype_str2cv = {
        "+": cv2.MARKER_CROSS,
        "d": cv2.MARKER_DIAMOND,
        "s": cv2.MARKER_SQUARE,
        "v": cv2.MARKER_TRIANGLE_DOWN,
        "x": cv2.MARKER_TILTED_CROSS,
    }

    mkcolor = (np.array(color3f) * 255).astype(int).tolist()
    mktype = mktype_str2cv[marker_type]
    diaglen = np.linalg.norm(img.shape[:2])
    mksize = diaglen / 50
    mksize = int(mksize * scale)
    _canvas = np.ascontiguousarray(canvas[:, :, :3])

    for p in pts_xy:
        cv2.drawMarker(_canvas, tuple(p), mkcolor, mktype, mksize, line_width)
    canvas[:, :, :3] = _canvas

    if canvas.dtype != np.uint8:
        output = ip.unify_image_dtype(canvas, img)
    else:
        output = canvas
    return output

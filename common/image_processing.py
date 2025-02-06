import numpy as np

# from ..geometry import geometry as geom
import igpy.common.shortfunc as sf
from igpy.common.shortfunc import to_float, to_uint8

# from . import shortfunc as sf
# from .shortfunc import to_float, to_uint8


class ExifInfo:
    """useful information from exif data"""

    def __init__(self):
        import datetime

        self.datetime_original: datetime.datetime = None

    @staticmethod
    def init_from_image_file(imgfilename):
        import PIL.ExifTags
        import PIL.Image
        import datetime

        obj = ExifInfo()

        img = PIL.Image.open(imgfilename)
        ex: dict = img._getexif()
        ex = {
            PIL.ExifTags.TAGS[key]: val
            for key, val in ex.items()
            if key in PIL.ExifTags.TAGS
        }
        obj.datetime_original = datetime.datetime.strptime(
            ex["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
        )

        return obj


class OpenCVConst:
    """some opencv constants"""

    _L_max = None

    @staticmethod
    def get_L_max_in_lab():
        """get the max value of L in LAB decomposition"""
        if OpenCVConst._L_max is not None:
            return OpenCVConst._L_max

        import cv2
        import numpy as np

        x = np.atleast_2d(np.arange(0, 256))
        img = np.dstack([x, x, x]).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L_max = lab[:, :, 0].max()
        if OpenCVConst._L_max is None:
            OpenCVConst._L_max = L_max
        return L_max


def poisson_blend(
    img,
    mask,
    chg_ij,
    chg_vals,
    outdict=None,
    w_fix=1.0,
    add_connection=None,
    formulate_only=False,
    solver_cache=None,
):
    """perform poisson blending over an image. You specify some pixel value changes,
    and other pixels are optimized using poinsson blending.

    parameters
    ----------------------------
    img
        the image, can be grayscale or multi channel.
    mask
        boolean mask over the image, indicating which pixels are to be optimized by poisson blending
    chg_ij
        nx2 image coordinates, specify which pixel values to change
    chg_vals
        nxd vector, specify for each point in chg_ij, the target value to be changed into
    add_connection
        manually specified additional pixel connections.
        nx4 matrix, each row is (a,b,c,d) meaning the pixel
        mask[a,b] and mask[c,d] are connected.
    formulate_only
        If True, the poisson equation only formulated but not solved, the A,b can be
        retrieved from outdict
    solver_cache
        a SparseSolverDecomp object, which can be used to solve A\b quickly

    return
    -----------------------------
    img_out
        the image with poisson blending performed
    """
    from . import cmconst
    import scipy.sparse as sparse
    from igpy.geometry import geometry as geom

    if cmconst.DBG_SAVE_POISSON_MATRIX_TO_DIR is not None:
        import scipy.io as sio

        fn_dir = cmconst.DBG_SAVE_POISSON_MATRIX_TO_DIR
        sf.mkdir(fn_dir)

        import uuid

        basename = str(uuid.uuid4()).split("-")[0]
        fn_dir = fn_dir + "/" + basename
        sf.mkdir(fn_dir)

    if outdict is None:
        outdict = dict()

    if False:
        solver_cache = sf.SparseSolverDecomp()

    img_shape_org = img.shape
    # unify format
    img = np.atleast_3d(img)
    chg_ij = np.atleast_2d(chg_ij)
    if np.ndim(chg_vals) == 1:
        chg_vals = chg_vals.reshape((-1, 1))
    n_fix = len(chg_ij)

    # create pixel graph, compute its laplacian
    nodes_ij, adjmat = make_pixel_mesh(mask, add_connection=add_connection)
    lpmat = geom.laplacian_from_adjmat(adjmat)
    n_var = len(nodes_ij)

    # variable label map
    label_map = np.zeros(mask.shape, dtype=int)
    label_map[nodes_ij[:, 0], nodes_ij[:, 1]] = np.arange(n_var)

    outdict["coeflist"] = []
    # output image
    img_out = np.array(img)

    # create and factorize matrix A
    print("factorizing poisson matrix")
    A_lp = lpmat
    idxvar_chg = label_map[chg_ij[:, 0], chg_ij[:, 1]]
    ii = np.arange(len(idxvar_chg))
    data = np.ones(len(ii))
    A_fix = sparse.csc_matrix((data, (ii, idxvar_chg)), (len(ii), n_var))
    A_fix *= w_fix
    A = sparse.vstack((A_lp, A_fix))

    if not formulate_only and solver_cache is None:
        ata = A.transpose().dot(A).tocsc()
        solver = sparse.linalg.splu(ata)

    for ich in range(img.shape[-1]):
        imgch = img[:, :, ich]
        vals = imgch[nodes_ij[:, 0], nodes_ij[:, 1]]
        lpval = lpmat.dot(vals)

        # formulate laplacian
        # A_lp = lpmat
        b_lp = lpval

        # formulate fix points
        # idxvar_chg = label_map[chg_ij[:,0], chg_ij[:,1]]
        # ii = np.arange(len(idxvar_chg))
        # data = np.ones(len(ii))
        # A_fix = sparse.csc_matrix((data, (ii, idxvar_chg)), (len(ii), n_var))
        b_fix = np.array(chg_vals[:, ich]).flatten()

        # scale the fixed points so that it has comparable weights relative to laplacian term
        # s = np.sqrt(n_var/n_fix)
        # A_fix *= s
        # b_fix *= s
        # A_fix *= w_fix
        b_fix *= w_fix

        # solve for this channel
        print("poisson blending channel {0} ...".format(ich))
        # A = sparse.vstack((A_lp, A_fix))
        b = np.concatenate((b_lp, b_fix))

        if formulate_only:
            sol = np.zeros(mask.sum())
        else:
            if solver_cache is not None:
                print("solve poisson using cache LU ...")
                assert (
                    solver_cache.A - A
                ).nnz == 0, "solver cache has a different Ax=b"
                sol = solver_cache.solve(b)
            else:
                atb = A.transpose().dot(b)
                sol = solver.solve(atb)
        # sol = sf.mldivide(A,b)

        if cmconst.DBG_SAVE_POISSON_MATRIX_TO_DIR is not None:
            savedata = {"A": A, "b": b, "img": imgch, "mask": mask}
            fn_output = "{0}/{1}.mat".format(fn_dir, ich)
            print("saving poisson matrix to {0}".format(fn_output))
            sio.savemat(fn_output, savedata)

            import pickle

            fn_output = "{0}/{1}.pkl".format(fn_dir, ich)
            with open(fn_output, "wb+") as fid:
                pickle.dump(savedata, fid, protocol=pickle.HIGHEST_PROTOCOL)

        # output
        x = img_out[:, :, ich]
        x[mask] = sol
        img_out[:, :, ich] = x

        # save
        outdict["coeflist"].append((A, b))

    img_out = img_out.reshape(img_shape_org)
    return img_out


def make_pixel_mesh(mask, nb=4, add_connection=None):
    """create connections among pixels to form a graph

    parameters
    ---------------------------
    mask
        boolean mask where mask[i,j]==True iff pixel (i,j) will be a node in the pixel graph
    nb
        4-connect or 8-connect pixel connection.
    add_connection
        manually specified additional pixel connections.
        nx4 matrix, each row is (a,b,c,d) meaning the pixel
        mask[a,b] and mask[c,d] are connected.

    return
    ---------------------------
    ij
        nx2 points in (i,j) format, nodes of the graph, whcih are pixels of mask.
        The ordering is the same as mask[:]
    adjmat
        adjacency matrix of the nodes
    """
    assert nb == 4, "only 4-connect is supported right now"
    import scipy.sparse as sparse
    import numba

    # assign an index to pixels
    lbmap = np.zeros(mask.shape, dtype=int)
    ii, jj = np.nonzero(mask)
    lbmap[ii, jj] = np.arange(len(ii))

    @numba.jit
    def make_edges(ii, jj, mask, lbmap):
        # create edges
        height, width = mask.shape
        npix = mask.sum()
        edges = np.zeros((npix * 2, 2), dtype=int)
        write = 0

        for i, j in zip(ii, jj):
            # check right
            u = lbmap[i, j]
            if j + 1 < width and mask[i, j + 1]:
                v = lbmap[i, j + 1]
                # edges[write,0] = u
                # edges[write,1] = v
                edges[write] = (u, v)
                write += 1

            # check down
            if i + 1 < height and mask[i + 1, j]:
                v = lbmap[i + 1, j]
                # edges[write,0] = u
                # edges[write,1] = v
                edges[write] = (u, v)
                write += 1
        return edges[:write]

    edges = make_edges(ii, jj, mask, lbmap)

    # additional edges
    if add_connection is not None:
        # remove invalid connection
        a, b, c, d = add_connection.T
        validmask = mask[a, b] & mask[c, d]
        conn = add_connection[validmask, :]

        a, b, c, d = conn.T
        indfrom = lbmap[a, b]
        indto = lbmap[c, d]

        add_edges = np.column_stack((indfrom, indto))
        edges = np.row_stack((edges, add_edges))

    # create adjmat
    data = np.ones(len(edges))
    adjmat = sparse.csr_matrix((data, (edges[:, 0], edges[:, 1])), (len(ii), len(ii)))
    adjmat = (adjmat + adjmat.T) > 0
    ijs = np.column_stack((ii, jj))
    return ijs, adjmat.tocsr()


def mph_dilate(mask, kernel):
    """dilate the mask by kernel

    parameters
    ---------------
    mask
        a binary mask to be processed, 1=foreground
    kernel
        a small matrix representing the structural element for the operation, where non-zero is treated as 1

    return
    ---------------
    outmask
        a dilated mask
    """
    import cv2

    mask = np.array(mask) > 0
    kernel = np.array(kernel) > 0
    res = cv2.dilate(mask.astype(np.uint8), kernel.astype(np.uint8))
    return res > 0


def mph_erode(mask, kernel):
    """erode the mask by kernel

    parameters
    ---------------
    mask
        a binary mask to be eroded, 1=foreground
    kernel
        a small matrix representing the structural element for
        erosion, where non-zero is treated as 1

    return
    ---------------
    outmask
        an eroded mask
    """
    import cv2

    mask = np.array(mask) > 0
    kernel = np.array(kernel) > 0

    res = cv2.erode(mask.astype(np.uint8), kernel.astype(np.uint8))
    return res > 0


def find_inner_boundary(mask):
    """find boundary of a mask, whose pixels are in this mask.

    parameters
    ------------------
    mask
        a binary mask, where 1=foreground and 0=background, the boundary
        is found relative to the foreground

    return
    ----------------
    boundary_mask
        a mask of the boundary, such that
        boundary_mask[i,j] = True iff pixel (i,j) is a boundary pixel.
        For any boundary pixel (i,j), we must have mask[i,j]=True
    """
    er_mask = mph_erode(mask, np.ones((3, 3)))
    bdmask = np.logical_xor(mask, er_mask)
    return bdmask


def find_contours(mask):
    """find contours of a binary mask by tracing the boundaries of foreground areas.
    Foreground areas are designated by 1 in mask. Note that for a boundary pixel p,
    mask[p]==True, i.e., the boundary pixels are part of the foreground object.

    parameters
    ----------------------
    mask
        a binary mask where 1 denotes foreground and 0 for background

    return
    ----------------------
    ptslist_xy
        a list of contours, each is nx2 list of points xy in the mask
    """
    import cv2

    mask = mask > 0
    res = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    pathlist = []
    for p in res[1]:
        p = np.squeeze(p).astype(int)
        pathlist.append(p)
    return pathlist


def find_contour_longest(mask):
    """find longest contour of a binary mask by tracing the boundaries of foreground areas.
    Foreground areas are designated by 1 in mask. Note that for a boundary pixel p,
    mask[p]==True, i.e., the boundary pixels are part of the foreground object.

    parameters
    ----------------------
    mask
        a binary mask where 1 denotes foreground and 0 for background

    return
    ----------------------
    pts_xy
        nx2 points xy in the mask
    """
    ptslist = find_contours(mask)
    counts = [len(x) for x in ptslist]
    idx = np.argmax(counts)
    return ptslist[idx]


def imresize_maxlen(img, maxlen, interp_method="nearest"):
    """resize the image given the maximum dimension of resized image.

    parameters
    -----------------
    img
        the image to be resized
    maxlen
        the new value foir max(width, height)
    interp_method
        interpolation method, can be 'nearest' for nearest neighbor, 'bilinear' for bilinear interpolation,
        'cubic' for cubic interpolation

    return
    -------------------
    img_scaled
        a scaled image
    """
    h, w = img.shape[:2]
    scale = maxlen / max(h, w)
    imgnew = imresize(img, scale, interp_method=interp_method)
    return imgnew


def imresize(img, scale_or_size, interp_method="nearest"):
    """resize the image.

    parameters
    ----------------------
    img
        the image to be resized
    scale_or_size
        a scalar, or floating point pair (sh.sw) where sw and sh are scale along width and height,
        or integer pair (h,w) to directly specify the width and height
    interp_method
        interpolation method, can be 'nearest' for nearest neighbor, 'bilinear' for bilinear interpolation,
        'cubic' for cubic interpolation

    return
    -------------------
    img_scaled
        a scaled image
    """
    img_org = img
    if img_org.dtype == bool:
        img = img_org.astype(np.uint8) * 255
    height, width = img.shape[:2]

    scale = np.array(scale_or_size)
    if np.isscalar(scale) or scale.ndim == 0:
        scale = np.array([scale, scale]).astype(float)

    float_type = [np.float64, np.float32]
    int_type = [int, np.int32, np.int64]
    if scale.dtype in float_type:
        h_new = int(height * scale[0])
        w_new = int(width * scale[1])
    elif scale.dtype in int_type:
        h_new, w_new = scale
    else:
        assert False, "unsupported scale dtype"

    if w_new == width and h_new == height:
        img_out = np.array(img_org)
    else:
        import cv2

        method = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
        }
        img_out = cv2.resize(img, (w_new, h_new), None, 0, 0, method[interp_method])

    if img_org.dtype == bool:
        img_out = img_out > 0
    return img_out


def imresize_by_scale(img, scale, interp_method="nearest"):
    """scale image.

    parameters
    ----------------------
    img
        the image to be resized
    scale
        a scalar or (sw,sh), where sw and sh are scale along width and height
    interp_method
        interpolation method, can be 'nearest' for nearest neighbor, 'bilinear' for bilinear interpolation,
        'cubic' for cubic interpolation

    return
    -------------------
    img_scaled
        a scaled image
    """
    if scale == 1.0:
        return np.array(img)

    img_org = img
    if img_org.dtype == bool:
        img = img_org.astype(np.uint8) * 255

    if np.isscalar(scale):
        scale = [scale, scale]

    import cv2

    method = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    height, width = img.shape[:2]
    w = int(width * scale[0])
    h = int(height * scale[1])
    img_out = cv2.resize(img, (0, 0), None, scale[0], scale[1], method[interp_method])

    if img_org.dtype == bool:
        img_out = img_out > 0
    return img_out


def find_connected_components(mask, nb=4):
    """find connected components in the binary mask

    parameters
    -------------------
    mask
        a binary mask, where 1=foreground and 0=background, 1 is considered object
    nb
        4 or 8, denoting 4-connected or 8-connnected

    return
    --------------------
    num_comp
        number of connected components
    lbmap
        label map of connected components, where lbmap==i mask the i-th component (beginning from 1)
    """
    import cv2

    mask = mask > 0

    n, lbmap = cv2.connectedComponents(mask.astype(np.uint8) * 255, None, nb)
    return n, lbmap


def mask_by_distance(mask: np.ndarray, pixel_distance: float):
    """expand/contract a mask by marking pixels within a distance threshold.

    parameters
    --------------
    mask
        foreground mask where mask[i,j]==True iff pixel (i,j) belongs to foreground
    pixel_distance
        distance to the object pixels. If positive, the mask is expanded by including pixels no further than the specified distance. If negative, the mask is contracted, by excluding pixels within the specified distance from the border of foreground/background intersection region.

    return
    -----------
    output
        the expanded/contracted mask
    """
    if pixel_distance == 0:
        return mask

    mask = mask.astype(bool)
    if pixel_distance > 0:  # expansion
        distmat = distance_transform(~mask)
        output = distmat <= pixel_distance
    else:  # contraction
        distmat = distance_transform(mask)
        output = distmat >= np.abs(pixel_distance)
    return output


def distance_transform(mask):
    """perform distance transform, where 0 is obstacle and 1 is empty space.

    parameters
    --------------------
    mask
        input mask, 0=obstacle and 1=empty space, we find for each point in empty space,
        the closest point to obstacle.

    return
    ---------------------
    distmat
        distmat[i,j] = distance value at (i,j)
    """
    import cv2

    dmat = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return dmat.astype(float)


def multi_band_blending(img_src, mask_src, img_dst, n_band, sigma=None):
    """paste img_src[mask_src] into img_dst using multiband blending

    parameters
    -------------------------------
    img_src
        the source RGB or gray image from which a patch will be taken to blend into another image
    mask_src
        the mask of img_src to indicate the pasting patch
    img_dst
        image the same size as img_src, where img_src will be pasted into
    n_band
        number of bands to use. Can be np.inf, to indicate to use as many bands as possible

    return
    --------------------------
    img_output
        a blended image
    """
    import cv2

    assert np.allclose(
        img_src.shape, img_dst.shape
    ), "src and dst image must be of the same shape"
    assert np.allclose(
        img_src.shape[:2], mask_src.shape
    ), "src and mask must be of the same dimension"

    if not np.isfinite(n_band):
        n_band = np.ceil(np.log2(np.max(img_src.shape[:2]))).astype(int)

    imgsize_min = np.array([5, 5])
    n_chn = np.ndim(img_src)

    dtype = img_src.dtype
    if dtype == np.uint8:
        img_src = img_src.astype(float) / 255

    if img_dst.dtype == np.uint8:
        img_dst = img_dst.astype(float) / 255

    if False:
        dbg_dir = r"D:\code\hmodel_cpp\fast_recon\fast_recon\data/multi_band_blending"
        sf.mkdir(dbg_dir)

        np.save(dbg_dir + "/img_src", img_src.astype(float))
        np.save(dbg_dir + "/img_dst", img_dst.astype(float))
        np.save(dbg_dir + "/mask", mask_src.astype(float))

    from skimage.transform import pyramid_expand, pyramid_reduce

    def pyr_down(img, dst_size):
        output = pyramid_reduce(img, sigma=sigma)
        output = cv2.resize(output, tuple(dst_size), None, 0, 0, cv2.INTER_NEAREST)
        assert not np.isnan(output).any()
        # assert output.max()<=1
        assert np.isnan(output).sum() == 0
        return output

    def pyr_up(img, dst_size):
        output = pyramid_expand(img, sigma=sigma)
        output = cv2.resize(output, tuple(dst_size), None, 0, 0, cv2.INTER_NEAREST)
        assert not np.isnan(output).any()
        # assert output.max()<=1
        assert np.isnan(output).sum() == 0
        return output

    # build gaussian pyramid
    gp_src = [img_src]
    gp_dst = [img_dst]
    weight_map = mask_src.astype(float)
    gp_wmap = [weight_map]

    x_src = img_src
    x_dst = img_dst
    wmap = weight_map
    for i in range(n_band):
        imgsize = np.array(x_src.shape[:2])
        imgsize = (imgsize / 2).astype(int)

        if np.any(imgsize < imgsize_min):
            break
        # x_src = cv2.pyrDown(x_src, None, tuple(imgsize))
        x_src = pyr_down(x_src, imgsize)
        gp_src.append(x_src)

        # x_dst = cv2.pyrDown(x_dst, None, tuple(imgsize))
        x_dst = pyr_down(x_dst, imgsize)
        gp_dst.append(x_dst)

        # wmap = cv2.pyrDown(wmap, None, tuple(imgsize))
        wmap = pyr_down(wmap, imgsize)
        wmap[wmap > 1] = 1.0
        gp_wmap.append(wmap)

    # build laplacian pyramid
    lp_src = [None] * len(gp_src)
    lp_dst = [None] * len(gp_dst)
    for i in range(len(gp_src) - 1):
        imgsize = gp_src[i].shape[:2]
        # recon_src = cv2.pyrUp(gp_src[i+1], None, tuple(imgsize))
        recon_src = pyr_up(gp_src[i + 1], imgsize)
        diff_src = gp_src[i] - recon_src
        lp_src[i] = diff_src

        # recon_dst = cv2.pyrUp(gp_dst[i+1], None, tuple(imgsize))
        recon_dst = pyr_up(gp_dst[i + 1], imgsize)
        diff_dst = gp_dst[i] - recon_dst
        lp_dst[i] = diff_dst
    lp_src[-1] = gp_src[-1]
    lp_dst[-1] = gp_dst[-1]

    # blend
    gp_blend = [None] * len(gp_src)
    for i in range(len(lp_src)):
        x = lp_src[i]
        y = lp_dst[i]
        wmap = gp_wmap[i]
        if n_chn == 3:
            wmap = wmap[:, :, np.newaxis]
        # import igpy.myplot.implot as imp
        # imp.imshow(wmap[:,:,0])

        z = x * wmap + y * (1 - wmap)
        assert np.abs(z.max()) <= 1
        gp_blend[i] = z

    # reconstruct
    for i in reversed(range(len(gp_blend) - 1)):
        x_lower = gp_blend[i + 1]
        x_upper = gp_blend[i]
        imgsize = x_upper.shape[:2]
        # x_lower_enlarge = cv2.pyrUp(x_lower, None, imgsize)
        x_lower_enlarge = pyr_up(x_lower, imgsize)
        x_lower_enlarge[np.isnan(x_lower_enlarge)] = 0
        x_final = x_upper + x_lower_enlarge
        gp_blend[i] = x_final

    img_final = gp_blend[0]
    if dtype == np.uint8:
        img_final = (img_final * 255).astype(np.uint8)

    return img_final


def get_largest_component(mask):
    """return the largest connected component in a binary mask."""
    import cv2

    ncomp, lbmap = cv2.connectedComponents(mask.astype(np.uint8), None, 4)
    nnz_per_lb = np.bincount(lbmap.flatten())
    if (lbmap == 0).any():  # lbmap has 0, which should be excluded
        nnz_per_lb[0] = 0
    maxlb = np.argmax(nnz_per_lb)
    outmask = lbmap == maxlb
    return outmask


def fill_image_by_nearest_neighbor(img, mask, mask_fill=None, knn_interp=None):
    """assign colors to the image pixels using nearest neighbor

    parameters
    -------------------
    img
        single channel or multi channel image
    mask
        mask of the valid pixels
    knn_interp
        an integer, if specified, the filled pixels will be interpolated using k nearest neighbors
    mask_fill
        the mask of pixels to be filled

    return
    ----------------
    img_new
        new image with pixels in ~mask filled
    """
    import scipy.ndimage as ndimg

    if mask_fill is None:
        mask_fill = ~mask

    if knn_interp is None or knn_interp == 1:
        res = ndimg.distance_transform_edt(~mask, return_indices=True)
        imat, jmat = res[1]
        # dist, imat, jmat = ndimg.distance_transform_edt(~mask, return_indices=True)
        ii = imat[~mask]
        jj = jmat[~mask]
        imgnew = np.array(img)
        imgnew[~mask] = img[ii, jj]
        imgnew[~mask_fill] = img[~mask_fill]
    else:
        import scipy.spatial as spatial

        ii, jj = np.nonzero(mask)
        pts = np.column_stack((ii, jj)).astype(float)
        kd = spatial.cKDTree(pts)

        ii, jj = np.nonzero(mask_fill)
        pts_query = np.column_stack((ii, jj)).astype(float)
        dist, idxmat = kd.query(pts_query, knn_interp)

        colors = img[mask]
        final_color = colors[idxmat].mean(axis=1)
        imgnew = np.array(img)
        imgnew[mask_fill] = final_color

    return imgnew


def unify_image_dtype(img, img_ref):
    """convert img.dtype to img_ref.dtype

    return
    -------------
    img
        a copy of img whose dtype is the same as img_ref
    """
    if img.dtype != img_ref.dtype:
        if img_ref.dtype in [np.float64, np.float32]:
            return to_float(img)
        elif img_ref.dtype == np.uint8:
            return to_uint8(img)
        else:
            raise NotImplementedError("only supports float and uint8")
    else:
        return np.array(img)


def gray_to_rgb(img):
    """convert single channel image into rgb image.
    The returned image has the same dtype as the input image, except for bool,
    where the returned image is uint8
    """
    assert np.ndim(img) == 2, "input must be single channel image"

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    img = np.dstack([img, img, img])
    return img


def rgb2lab(img):
    """convert RGB image to LAB image.

    parameters
    ---------------------
    img
        RGB image, can be float (range in 0,1) or uint8 (range in 0,255)

    return
    ----------------
    img_lab
        the lab image,  the range of the channels are L in [0,1], a in [0,1], b in [0,1].
    """
    import cv2

    assert np.ndim(img) == 3 and img.shape[-1] == 3

    L_max = OpenCVConst.get_L_max_in_lab()

    img = to_uint8(img)
    imglab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # opencv range, L in [0,100], a and b in [0,255]
    imglab = imglab.astype(float)
    imglab[:, :, 0] /= L_max
    imglab[:, :, 1:] /= 255

    return imglab


def lab2rgb(img):
    """convert LAB image to RGB image.

    parameters
    -----------------
    img
        LAB image, must be float, the range of the channels are L in [0,1], a in [0,1], b in [0,1]

    return
    ---------------
    img_rgb
        RGB image, in float, the range is in [0,1]
    """
    import cv2

    assert np.ndim(img) == 3 and img.shape[-1] == 3
    assert img.dtype == np.float64

    L_max = OpenCVConst.get_L_max_in_lab()

    imglab = np.array(img)
    imglab[:, :, 0] = np.clip(imglab[:, :, 0] * L_max, 0, L_max)
    imglab[:, :, 1:] = np.clip(imglab[:, :, 1:] * 255, 0, 255)
    imglab = imglab.astype(np.uint8)
    imgrgb = cv2.cvtColor(imglab, cv2.COLOR_LAB2RGB)

    return to_float(imgrgb)


def clip_images_by_mask(mask, imglist, set_unmask_value_to=None):
    """given a binary mask and a set of images, clip the images so that they only contain contents inside the mask.

    parameters
    -------------------
    mask
        a binary mask
    imglist
        a list of single channel for multi channel images
    set_unmask_value_to
        if set, the unmasked pixels will be set to this value.
        If the input imglist are multichannel images, then this value can be a vector (e.g., RGB)

    return
    -----------------
    imglist_clip
        the clipped images
    """
    ii, jj = np.nonzero(mask)
    imin, imax = ii.min(), ii.max() + 1
    jmin, jmax = jj.min(), jj.max() + 1

    output = []
    for img in imglist:
        imgnew = np.array(img)

        if set_unmask_value_to is not None:
            imgnew[~mask] = set_unmask_value_to

        if np.ndim(img) == 2:
            imgnew = imgnew[imin:imax, jmin:jmax]
        elif np.ndim(img) == 3:
            imgnew = imgnew[imin:imax, jmin:jmax, :]
        else:
            assert False, "does not support images of more than 3 dimensions"

        output.append(imgnew)

    return output


def unclip_and_resize_image(
    img, mask_original, img_original=None, use_original_size=True
):
    """suppose img is cliped from img_original using mask_original, now recover
    an image whose size is the same as a scaled img_original, but with the masked region replaced by img.

    parameters
    --------------------
    img
        the current image, containing the clipped region of img_original
    mask_original
        the mask used to clip img from img_original
    img_original
        the original image. If None, it will be a zero-filled image of the same size as mask_original and with the same number of channels as img.
    use_original_size
        if True, the img is scaled so that the returned image has the same size as img_original.
        if False, the img is not scaled, and img_original is scaled accordingly.

    return
    -------------------
    imgnew
        an image whose size is scaled img_original.shape, with the contents filled by img
    """
    if img_original is None:
        if np.ndim(img) == 3:
            img_original = np.zeros(
                [*mask_original.shape, img.shape[-1]], dtype=img.dtype
            )
        else:
            img_original = np.zeros([*mask_original.shape], dtype=img.dtype)

    # find clipped region
    ii, jj = np.nonzero(mask_original)
    imin, imax = ii.min(), ii.max() + 1
    jmin, jmax = jj.min(), jj.max() + 1

    out_width = jmax - jmin
    out_height = imax - imin

    # create canvas
    if use_original_size:
        canvas = np.array(img_original)
        content = imresize(img, [out_height, out_width])
        if np.ndim(canvas) == 3:
            canvas[imin:imax, jmin:jmax, :] = content
        else:
            canvas[imin:imax, jmin:jmax] = content
        canvas[~mask_original] = img_original[~mask_original]
    else:
        scale = img.shape[0] / out_height
        org_scale = imresize(img_original, scale)
        canvas = np.array(org_scale)
        mask = imresize(mask_original, scale)
        ii, jj = np.nonzero(mask)
        imin, imax = ii.min(), ii.min() + img.shape[0]
        jmin, jmax = jj.min(), jj.min() + img.shape[1]

        if np.ndim(canvas) == 3:
            canvas[imin:imax, jmin:jmax, :] = img
        else:
            canvas[imin:imax, jmin:jmax] = img
        canvas[~mask] = org_scale[~mask]

    return canvas


def make_gaussian_image(ksize, sigma=None, imgsize_hw=None, center_xy=None):
    """create an image that contains a guassian kernel

    parameters
    ----------------
    ksize
        an odd integer, the kernel size
    sigma
        a scalar, sigma of the gaussian. If not set, we use opencv default value,
        which is sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    imgsize_hw
        the returned image size in (height, width).
        If not set, the size is the kernel size
    center_xy
        a point in (x,y), which defines the center of the kernel.
        By default, it is the image center
    """
    import cv2

    ksize = np.atleast_1d(ksize)
    assert len(ksize) == 1, "ksize must be a scalar"
    ksize = int(ksize[0])

    assert ksize % 2 == 1, "ksize must be odd"

    if sigma is None:
        sigma = -1

    if imgsize_hw is None:
        imgsize_hw = np.array([ksize, ksize])
    else:
        imgsize_hw = np.atleast_1d(imgsize_hw).flatten()

    if center_xy is None:
        center_xy = (imgsize_hw / 2).astype(int)
    else:
        center_xy = np.atleast_1d(center_xy).flatten()

    kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
    kernel = np.outer(kernel, kernel)
    output = np.zeros(imgsize_hw)

    dlen = int(ksize / 2)
    minc_xy = np.clip(center_xy - dlen, 0, np.inf).astype(int)
    maxc_xy = np.clip(center_xy + dlen, 0, imgsize_hw[::-1] - 1).astype(int)

    dx, dy = maxc_xy - minc_xy + 1
    x0, y0 = minc_xy
    output[y0 : y0 + dy, x0 : x0 + dx] = kernel[:dy, :dx]

    return output


def polygon_to_mask(pts_xy, imgsize_hw):
    """rasterize a polygon into binary mask

    parameters
    ---------------
    pts_xy
        2d point list defining the polygon
    imgsize_hw
        the image size in (height, width)

    return
    -------------
    mask
        a binary mask with points inside the polygon being flagged as 1, and 0 elsewhere
    """
    import cv2

    mask = np.zeros(imgsize_hw, dtype=np.uint8)
    pts_xy = np.atleast_2d(pts_xy)

    pts = pts_xy.astype(int).reshape((-1, 1, 2))
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), cv2.FILLED)
    return mask > 0

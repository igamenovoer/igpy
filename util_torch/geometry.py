# some geometry operations using torch tensors
import torch
import kornia  # required kornia
import numpy as np

# TODO: need to test all these functions


def transform_points_homogeneous(
    pts: torch.Tensor,
    tmat: torch.Tensor,
) -> torch.Tensor:
    """
    Efficiently transform each point by its corresponding homogeneous transformation matrix.

    parameters
    ----------------
    pts : torch.Tensor
        Points tensor of shape (N, D) where N is number of points
        and D is dimension of each point
    tmat : torch.Tensor
        Transformation matrices of shape (N, D+1, D+1) where
        each matrix is for the corresponding point

    return
    ---------------
    transformed_pts : torch.Tensor
        Transformed points of shape (N, D)
    """
    # Make points homogeneous by adding 1 as last coordinate
    N, D = pts.shape
    pts_homo = torch.ones((N, D + 1), dtype=pts.dtype, device=pts.device)
    pts_homo[:, :D] = pts

    # Reshape for batch matrix multiplication
    pts_homo = pts_homo.unsqueeze(1)  # Shape: (N, 1, D+1)

    # Perform batch matrix multiplication (P@tmat)
    transformed_pts_homo = torch.bmm(pts_homo, tmat)  # Shape: (N, 1, D+1)

    # Convert back from homogeneous coordinates
    transformed_pts_homo = transformed_pts_homo.squeeze(1)  # Shape: (N, D+1)
    transformed_pts = transformed_pts_homo[:, :D] / transformed_pts_homo[:, -1].unsqueeze(-1)

    return transformed_pts


def transform_points_linear(
    pts: torch.Tensor,
    tmat: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Efficiently transform each point by its corresponding linear transformation matrix,
    and optionally add a bias vector. Let P=pts[i], one of the input points, and let A=tmat[i], and b=bias[i],
    one of the transformation matrices and bias vectors. Then, the transformed point is given by:
    P_new = P @ A + b

    parameters
    ----------------
    pts : torch.Tensor
        Points tensor of shape (N, D) where N is number of points
        and D is dimension of each point
    tmat : torch.Tensor
        Transformation matrices of shape (N, D, C) where
        each matrix is for the corresponding point
    bias : torch.Tensor | None
        Bias vector of shape (N, C), or (C,) with auto broadcasting, or None

    return
    ---------------
    transformed_pts : torch.Tensor
        Transformed points of shape (N, C)
    """
    # Reshape points for batch matrix multiplication
    # For P @ A transformation (xA instead of Ax)
    # pts shape is (N, D), tmat shape is (N, D, C)

    # Apply transformation: P @ A
    transformed_pts = torch.bmm(pts.unsqueeze(1), tmat).squeeze(1)  # Shape: (N, C)

    # Add bias if provided: P @ A + b
    if bias is not None:
        # Handle bias auto broadcasting if needed
        if bias.dim() == 1:
            # If bias is (C,), reshape to (1, C) for broadcasting
            bias = bias.unsqueeze(0)
        transformed_pts = transformed_pts + bias

    return transformed_pts


def transform_points_linear_all_to_all_with_reduce(
    pts: torch.Tensor,
    tmat: torch.Tensor,
    reduce_method: str = "squared_sum",
) -> torch.Tensor:
    """
    Transform points by a linear transformation matrix, and optionally add a bias vector.
    This function is designed for the case where you have N points to be transformed by M different linear transformations,
    along with M bias vectors. Each point will be transformed by each of the M transformations, leading to M points.

    Specifically, for point[k] in (N,D), and transformation[m] in (M,D,C),
    the intemediate output will be a tensor of shape (N,M,C), where output[i,j] = point[i] @ transformation[j].
    This function will then reduce the output over the all points, leading to a tensor of shape (N,M).

    parameters
    ----------------
    pts : torch.Tensor, shape (N, D)
        The points to transform.
    tmat : torch.Tensor, shape (M, D, C)
        The transformation matrices to apply to the points.
    reduce_method : str, 'squared_sum' | 'sum' | 'l2_norm'
        The method to reduce the output over the all points.

    returns
    ---------------
    transformed_and_reduced : torch.Tensor, shape (N, M)
        The reduced output for each point after transformation.
    """
    if reduce_method == "squared_sum":
        # Compute transformation and squared sum reduction in one step
        # First compute the transformed points: mdj,ni->nmj
        # Then square and sum along j dimension: nmj->nm
        transformed_and_reduced = torch.einsum("ni,mij,nk,mkj->nm", pts, tmat, pts, tmat)
    elif reduce_method == "sum":
        # Compute transformation and sum reduction in one step
        # nd,mdc->nm directly sums over the c dimension
        transformed_and_reduced = torch.einsum("nd,mdc->nm", pts, tmat)
    elif reduce_method == "l2_norm":
        # Compute transformation and l2 norm reduction in one step
        # nd,mdc->nm directly sums over the c dimension
        sum_of_squares = torch.einsum("ni,mij,nk,mkj->nm", pts, tmat, pts, tmat)
        transformed_and_reduced = torch.sqrt(sum_of_squares)
    else:
        raise ValueError(f"Invalid reduce_method: {reduce_method}")

    return transformed_and_reduced


def transform_points_linear_all_to_all(
    pts: torch.Tensor,
    tmat: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Transform points by a linear transformation matrix, and optionally add a bias vector.
    This function is designed for the case where you have N points to be transformed by M different linear transformations,
    along with M bias vectors. Each point will be transformed by each of the M transformations, leading to M points.

    Specifically, for point[k] in (N,D), and transformation[m] in (M,D,C), and bias[m] in (M,C),
    the output will be a tensor of shape (N,M,C), where output[i,j] = point[i] @ transformation[j] + bias[j].

    parameters
    ----------------
    pts : torch.Tensor, shape (N, D)
        The points to transform.
    tmat : torch.Tensor, shape (M, D, C)
        The transformation matrices to apply to the points.
    bias : torch.Tensor, shape (M, C), or (C,) with auto broadcasting, or None
        The bias vectors to add to the points.

    returns
    ---------------
    transformed_pts : torch.Tensor, shape (N, M, C)
        The transformed points.
    """
    # Use einsum for efficient matrix multiplication across all points and transformations
    # 'mdc,nd->nmc': m transformations, d input dimension, c output dimension, n points
    transformed_pts = torch.einsum("nd,mdc->nmc", pts, tmat)

    # Add bias if provided
    if bias is not None:
        # Handle bias auto broadcasting if needed
        if bias.dim() == 1:
            # If bias is (C,), reshape to (1, 1, C) for broadcasting to (N, M, C)
            bias = bias.view(1, 1, -1)
        elif bias.dim() == 2:
            # If bias is (M, C), reshape to (1, M, C) for broadcasting to (N, M, C)
            bias = bias.unsqueeze(0)

        transformed_pts = transformed_pts + bias

    return transformed_pts


def create_similarity_transform_4x4(
    quaternions: torch.Tensor | None = None,
    translations: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create 4x4 similarity transformation matrices from batched rotation quaternions, translation vectors and scales.
    The created matrices are on the device as the input tensors.

    IMPORTANT: The returned matrices are designed for right multiplication with points,
    i.e., point @ transformation (where point is a row vector). This is consistent
    with the convention used in transform_points_linear_all_to_all.

    The transformation is applied in the order of scaling, rotation, and translation.
    The transformation matrix is constructed as follows:
    new_point = point @ R @ S + translation
    where R is the rotation matrix, S is the axis-aligned scaling matrix, and translation is the translation vector.

    parameters
    ----------------
    quaternions : torch.Tensor
        Rotation quaternions of shape (N, 4), or None. If None, the rotation
        matrix is the identity matrix.
    translations : torch.Tensor
        Translation vectors of shape (N, 3), or (3,) with auto broadcasting,
        or None. If None, the translation vector is the zero vector.
    scales : torch.Tensor
        Scale factors of shape (N, 3), or (3,) with auto broadcasting,
        or None. If None, the scale is the identity matrix.
        Scaling is applied before rotation and translation.

    return
    ---------------
    transform_mats : torch.Tensor
        Rigid transformation matrices of shape (N, 4, 4) for right multiplication
        with points (point @ transform_mats).
    """
    # Handle None inputs
    device = None
    dtype = None
    batch_size = None

    if quaternions is not None:
        device = quaternions.device
        dtype = quaternions.dtype
        batch_size = quaternions.shape[0]

        # Convert quaternions to rotation matrices
        import kornia

        rotation_matrices = kornia.geometry.quaternion_to_rotation_matrix(quaternions)
    else:
        if translations is not None:
            device = translations.device
            dtype = translations.dtype
            if translations.dim() == 1:
                batch_size = 1
            else:
                batch_size = translations.shape[0]
        else:
            # Both inputs are None, return a single identity matrix
            return torch.eye(4).unsqueeze(0)

    # Create transformation matrices
    transform_mats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    # Set rotation part if provided
    if quaternions is not None:
        # For right multiplication (point @ transform), we use the rotation matrix directly
        transform_mats[:, :3, :3] = rotation_matrices

    # Set scale part if provided
    if scales is not None:
        # Create a diagonal scaling matrix
        if scales.dim() == 1:  # (3,) shape
            scale_matrix = torch.diag(scales).unsqueeze(0)
            scale_matrix = scale_matrix.repeat(batch_size, 1, 1)
        else:  # (N, 3) shape
            scale_matrix = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
            for i in range(batch_size):
                scale_matrix[i] = torch.diag(scales[i])

        # Apply scaling to the rotation part of the transformation matrix
        transform_mats[:, :3, :3] = torch.matmul(rotation_matrices, scale_matrix)

    # Set translation part if provided
    if translations is not None:
        # For right multiplication (point @ transform), translation goes in the last row
        if translations.dim() == 1:  # (3,) shape
            transform_mats[:, 3, :3] = translations.unsqueeze(0)
        else:  # (N, 3) shape
            transform_mats[:, 3, :3] = translations

    return transform_mats

# some geometry operations using torch tensors
import torch
import kornia  # required kornia
import numpy as np


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
    pts_homo = pts_homo.unsqueeze(-1)  # Shape: (N, D+1, 1)

    # Perform batch matrix multiplication
    transformed_pts_homo = torch.bmm(tmat, pts_homo)  # Shape: (N, D+1, 1)

    # Convert back from homogeneous coordinates
    transformed_pts_homo = transformed_pts_homo.squeeze(-1)  # Shape: (N, D+1)
    transformed_pts = transformed_pts_homo[:, :D] / transformed_pts_homo[:, -1].unsqueeze(-1)

    return transformed_pts


def transform_points_linear(
    pts: torch.Tensor,
    tmat: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Efficiently transform each point by its corresponding linear transformation matrix,
    and optionally add a bias vector.

    parameters
    ----------------
    pts : torch.Tensor
        Points tensor of shape (N, D) where N is number of points
        and D is dimension of each point
    tmat : torch.Tensor
        Transformation matrices of shape (N, D, D) where
        each matrix is for the corresponding point
    bias : torch.Tensor | None
        Bias vector of shape (N, D), or (D,) with auto broadcasting, or None

    return
    ---------------
    transformed_pts : torch.Tensor
        Transformed points of shape (N, D)
    """
    # Apply transformation: Ax
    transformed_pts = torch.bmm(tmat, pts.unsqueeze(-1)).squeeze(-1)

    # Add bias if provided: Ax + b
    if bias is not None:
        # Handle bias auto broadcasting if needed
        if bias.dim() == 1:
            # If bias is (D,), reshape to (1, D) for broadcasting
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

    Specifically, for point[k] in (N,D), and transformation[m] in (M,C,D),
    the intemediate output will be a tensor of shape (N,M,C), where output[i,j] = transformation[j] @ point[i].
    This function will then reduce the output over the all points, leading to a tensor of shape (N,M).

    parameters
    ----------------
    pts : torch.Tensor, shape (N, D)
        The points to transform.
    tmat : torch.Tensor, shape (M, C, D)
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
        # First compute the transformed points: mcd,nd->nmc
        # Then square and sum along c dimension: nmc->nm
        transformed_and_reduced = torch.einsum("mij,nj,mik,nk->nm", tmat, pts, tmat, pts)
    elif reduce_method == "sum":
        # Compute transformation and sum reduction in one step
        # mcd,nd->nm directly sums over the c dimension
        transformed_and_reduced = torch.einsum("mcd,nd->nm", tmat, pts)
    elif reduce_method == "l2_norm":
        # Compute transformation and l2 norm reduction in one step
        # mcd,nd->nm directly sums over the c dimension
        sum_of_squares = torch.einsum("mij,nj,mik,nk->nm", tmat, pts, tmat, pts)
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

    Specifically, for point[k] in (N,D), and transformation[m] in (M,C,D), and bias[m] in (M,C),
    the output will be a tensor of shape (N,M,C), where output[i,j] = transformation[j] @ point[i] + bias[j].

    parameters
    ----------------
    pts : torch.Tensor, shape (N, D)
        The points to transform.
    tmat : torch.Tensor, shape (M, C, D)
        The transformation matrices to apply to the points.
    bias : torch.Tensor, shape (M, C), or (C,) with auto broadcasting, or None
        The bias vectors to add to the points.

    returns
    ---------------
    transformed_pts : torch.Tensor, shape (N, M, C)
        The transformed points.
    """
    # Use einsum for efficient matrix multiplication across all points and transformations
    # 'mcd,nd->nmc': m transformations, c output dimension, d input dimension, n points
    transformed_pts = torch.einsum("mcd,nd->nmc", tmat, pts)

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


def create_rigid_transform_4x4(
    quaternions: torch.Tensor | None = None,
    translations: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a 4x4 rigid transformation matrices from batched rotation quaternions
    and translation vectors. The created matrices are on the device as the input
    tensors.

    parameters
    ----------------
    quaternions : torch.Tensor
        Rotation quaternions of shape (N, 4), or None. If None, the rotation
        matrix is the identity matrix.
    translations : torch.Tensor
        Translation vectors of shape (N, 3), or (3,) with auto broadcasting,
        or None. If None, the translation vector is the zero vector.

    return
    ---------------
    transform_mats : torch.Tensor
        Rigid transformation matrices of shape (N, 4, 4)
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
        transform_mats[:, :3, :3] = rotation_matrices

    # Set translation part if provided
    if translations is not None:
        # Handle broadcasting if needed
        if translations.dim() == 1:  # (3,) shape
            transform_mats[:, :3, 3] = translations.unsqueeze(0)
        else:  # (N, 3) shape
            transform_mats[:, :3, 3] = translations

    return transform_mats

# radial basis functions
# from torchrbf package, see https://github.com/ArmanMaesumi/torchrbf

import torch
from typing import Callable
from enum import Enum

eps = 1e-7


def linear(r):
    """
    Linear radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    return
    ---------------
    torch.Tensor: Negative of the input tensor.
    """
    return -r


def thin_plate_spline(r):
    """
    Thin plate spline radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    return
    ---------------
    torch.Tensor: r^2 * log(r) with r clamped to a minimum value to avoid log(0).
    """
    r = torch.clamp(r, min=eps)
    return r**2 * torch.log(r)


def cubic(r):
    """
    Cubic radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    return
    ---------------
    torch.Tensor: r^3
    """
    return r**3


def quintic(r):
    """
    Quintic radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    return
    ---------------
    torch.Tensor: -r^5
    """
    return -(r**5)


def multiquadric(r):
    """
    Multiquadric radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    returns
    ---------------
    torch.Tensor: -sqrt(r^2 + 1)
    """
    return -torch.sqrt(r**2 + 1)


def inverse_multiquadric(r):
    """
    Inverse multiquadric radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    returns
    ---------------
    torch.Tensor: 1/sqrt(r^2 + 1)
    """
    return 1 / torch.sqrt(r**2 + 1)


def inverse_quadratic(r):
    """
    Inverse quadratic radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    returns
    ---------------
    torch.Tensor: 1/(r^2 + 1)
    """
    return 1 / (r**2 + 1)


def gaussian(r):
    """
    Gaussian radial basis function.

    parameters
    ----------------
    r : torch.Tensor
        Input tensor representing radial distances.

    returns
    ---------------
    torch.Tensor: exp(-r^2)
    """
    return torch.exp(-(r**2))


class RadialBasisFunction(Enum):
    LINEAR = "linear"
    THIN_PLATE_SPLINE = "thin_plate_spline"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    INVERSE_QUADRATIC = "inverse_quadratic"
    GAUSSIAN = "gaussian"


RADIAL_FUNCS = {
    RadialBasisFunction.LINEAR.value: linear,
    RadialBasisFunction.THIN_PLATE_SPLINE.value: thin_plate_spline,
    RadialBasisFunction.CUBIC.value: cubic,
    RadialBasisFunction.QUINTIC.value: quintic,
    RadialBasisFunction.MULTIQUADRIC.value: multiquadric,
    RadialBasisFunction.INVERSE_MULTIQUADRIC.value: inverse_multiquadric,
    RadialBasisFunction.INVERSE_QUADRATIC.value: inverse_quadratic,
    RadialBasisFunction.GAUSSIAN.value: gaussian,
}


def get_radial_function(rbf_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get the radial basis function based on the type.

    parameters
    ----------------
    rbf_type : str
        The type of radial basis function to get.

    returns
    ---------------
    rbf_fn : Callable[[torch.Tensor], torch.Tensor]
        The radial basis function.
    """
    return RADIAL_FUNCS[rbf_type]

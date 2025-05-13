# radial basis functions
# from torchrbf package, see https://github.com/ArmanMaesumi/torchrbf

import torch
from typing import Callable, Dict, Union
from enum import Enum

eps: float = 1e-7


def identity(r: torch.Tensor) -> torch.Tensor:
    """
    Identity radial basis function.
    """
    return r


def thin_plate_spline(r: torch.Tensor) -> torch.Tensor:
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


def cubic(r: torch.Tensor) -> torch.Tensor:
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


def quintic(r: torch.Tensor) -> torch.Tensor:
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


def multiquadric(r: torch.Tensor) -> torch.Tensor:
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


def inverse_multiquadric(r: torch.Tensor) -> torch.Tensor:
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


def inverse_quadratic(r: torch.Tensor) -> torch.Tensor:
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


def gaussian(r: torch.Tensor) -> torch.Tensor:
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
    THIN_PLATE_SPLINE = "thin_plate_spline"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    INVERSE_QUADRATIC = "inverse_quadratic"
    GAUSSIAN = "gaussian"
    IDENTITY = "identity"


RADIAL_FUNCS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    RadialBasisFunction.IDENTITY.value: identity,
    RadialBasisFunction.THIN_PLATE_SPLINE.value: thin_plate_spline,
    RadialBasisFunction.CUBIC.value: cubic,
    RadialBasisFunction.QUINTIC.value: quintic,
    RadialBasisFunction.MULTIQUADRIC.value: multiquadric,
    RadialBasisFunction.INVERSE_MULTIQUADRIC.value: inverse_multiquadric,
    RadialBasisFunction.INVERSE_QUADRATIC.value: inverse_quadratic,
    RadialBasisFunction.GAUSSIAN.value: gaussian,
}


def get_radial_function(rbf_type: str | RadialBasisFunction) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get the radial basis function based on the type.

    parameters
    ----------------
    rbf_type : str | RadialBasisFunction
        The type of radial basis function to get.

    returns
    ---------------
    rbf_fn : Callable[[torch.Tensor], torch.Tensor]
        The radial basis function.
    """
    if isinstance(rbf_type, str):
        return RADIAL_FUNCS[rbf_type]
    else:
        return RADIAL_FUNCS[rbf_type.value]

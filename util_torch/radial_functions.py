# radial basis functions
# for definitions, see https://rbf.readthedocs.io/en/latest/basis.html#rbf.basis.RBF
# note that, to scale the radial distance, we use the radial_scale parameter,
# and we ALWAYS apply it like as r = r * radial_scale,
# this DIFFERS from the above article, where the eps is applied sometimes as r = eps * r
# and sometimes as r = r / eps

import torch
import torch.nn as nn
from typing import Callable, Dict, Union

# from enum import Enum


class RBFType:
    THIN_PLATE_SPLINE = "thin_plate_spline"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    INVERSE_QUADRATIC = "inverse_quadratic"
    GAUSSIAN = "gaussian"
    IDENTITY = "identity"
    EXPONENTIAL = "exponential"


class RBFBase(nn.Module):
    """
    Base class for radial basis functions (RBFs).
    """

    def __init__(self, rbf_type: str):
        super().__init__()
        self._rbf_type = rbf_type
        self.register_buffer("rbf_type", torch.tensor([ord(c) for c in rbf_type], dtype=torch.int32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the radial basis function.

        parameters
        ----------------
        r : torch.Tensor
            Input tensor representing radial distances.

        return
        ---------------
        torch.Tensor: Output of the radial basis function.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class RBF_Identity(RBFBase):
    """
    Identity radial basis function, just return the input.
    """

    def __init__(self):
        super().__init__(RBFType.IDENTITY)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return r


class RBF_ThinPlateSpline(RBFBase):
    """
    Thin plate spline radial basis function.
    It is also named 2nd-order polyharmonic spline, see https://rbf.readthedocs.io/en/latest/basis.html
    """

    def __init__(self, radial_scale: float = 1.0, eps_guard: float = 1e-6):
        """
        Initialize the thin plate spline radial basis function.
        Range is (0, inf).

        parameters
        ----------------
        radial_scale : float
            Scale factor for the radial distance.
        eps_guard : float
            Small value to avoid log(0).
        """
        super().__init__(RBFType.THIN_PLATE_SPLINE)
        assert eps_guard > 0, "eps_guard must be positive"
        self.register_buffer("eps_guard", torch.tensor(eps_guard, dtype=torch.float32))
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = torch.clamp(r * self.radial_scale, min=self.eps_guard)
        return r**2 * torch.log(r)


class RBF_Multiquadric(RBFBase):
    """
    Multiquadric radial basis function.
    It is always negative, range is (-inf, 0).

    Definition:
    s=scale, r=radial distance
    phi(r) = -sqrt((r*s)^2 + 1)
    """

    def __init__(self, radial_scale: float = 1.0):
        super().__init__(RBFType.MULTIQUADRIC)
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        scale = self.radial_scale
        r = r * scale
        output = -torch.sqrt(r**2 + 1)
        return output


class RBF_InverseMultiquadric(RBFBase):
    """
    Inverse multiquadric radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    s=scale, r=radial distance
    phi(r) = 1/sqrt((r*s)^2 + 1)
    """

    def __init__(self, radial_scale: float = 1.0):
        super().__init__(RBFType.INVERSE_MULTIQUADRIC)
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        scale = self.radial_scale
        r = r * scale
        output = 1 / torch.sqrt(r**2 + 1)
        return output


class RBF_InverseQuadratic(RBFBase):
    """Inverse quadratic radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    s=scale, r=radial distance
    phi(r) = 1/((r*s)^2 + 1)
    """

    def __init__(self, radial_scale: float = 1.0):
        super().__init__(RBFType.INVERSE_QUADRATIC)
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        scale = self.radial_scale
        r = r * scale
        output = 1 / (r**2 + 1)
        return output


class RBF_Gaussian(RBFBase):
    """Gaussian radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    s=scale, r=radial distance
    phi(r) = exp(-((r*s)^2))
    """

    def __init__(self, radial_scale: float = 1.0):
        super().__init__(RBFType.GAUSSIAN)
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        scale = self.radial_scale
        r = r * scale
        output = torch.exp(-(r**2))
        return output


class RBF_Exponential(RBFBase):
    """Exponential radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    s=scale, r=radial distance
    phi(r) = exp(-r*s)
    """

    def __init__(self, radial_scale: float = 1.0):
        super().__init__(RBFType.EXPONENTIAL)
        self.register_buffer("radial_scale", torch.tensor(radial_scale, dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        scale = self.radial_scale
        r = r * scale
        output = torch.exp(-(r * scale))
        return output


class RBFFactory:
    @classmethod
    def create_rbf(cls, rbf_type: str, radial_scale: float = 1.0) -> RBFBase:
        """
        Factory method to create a radial basis function instance.

        parameters
        ----------------
        rbf_type : RBFType
            Type of the radial basis function.
        radial_scale : float
            Scale factor for the radial distance.

        return
        ---------------
        RBFBase: Instance of the specified radial basis function.
        """
        if rbf_type == RBFType.THIN_PLATE_SPLINE:
            return RBF_ThinPlateSpline(radial_scale=radial_scale)
        elif rbf_type == RBFType.MULTIQUADRIC:
            return RBF_Multiquadric(radial_scale=radial_scale)
        elif rbf_type == RBFType.INVERSE_MULTIQUADRIC:
            return RBF_InverseMultiquadric(radial_scale=radial_scale)
        elif rbf_type == RBFType.INVERSE_QUADRATIC:
            return RBF_InverseQuadratic(radial_scale=radial_scale)
        elif rbf_type == RBFType.GAUSSIAN:
            return RBF_Gaussian(radial_scale=radial_scale)
        elif rbf_type == RBFType.EXPONENTIAL:
            return RBF_Exponential(radial_scale=radial_scale)
        elif rbf_type == RBFType.IDENTITY:
            return RBF_Identity()
        else:
            raise ValueError(f"Unknown radial basis function type: {rbf_type}")

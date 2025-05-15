# radial basis functions
# for definitions, see https://rbf.readthedocs.io/en/latest/basis.html#rbf.basis.RBF

import torch
import torch.nn as nn
from typing import Callable, Dict, Union


class RBFType:
    THIN_PLATE_SPLINE = "thin_plate_spline"
    LINEAR_NEGATIVE = "linear_negative"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    INVERSE_QUADRATIC = "inverse_quadratic"
    GAUSSIAN = "gaussian"
    IDENTITY = "identity"
    EXPONENTIAL = "exponential"
    SQUARED_EXPONENTIAL = "squared_exponential"


class RBFBase(nn.Module):
    """
    Base class for radial basis functions (RBFs).
    """

    def __init__(self, rbf_type: str):
        super().__init__()
        self.register_buffer("_rbf_type", torch.tensor([ord(c) for c in rbf_type], dtype=torch.int32))

    @property
    def rbf_type(self) -> str:
        """
        Get the type of the radial basis function.

        return
        ---------------
        str: Type of the radial basis function.
        """
        return "".join([chr(c.item()) for c in self._rbf_type])

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the radial basis function.

        parameters
        ----------------
        r : torch.Tensor
            Input tensor representing radial distances.
        eps : torch.Tensor | None
            The epsilon scale in the original documentation, which is used to scale the radial distance,
            by eps*r or r/eps, depends on the RBF type.

        return
        ---------------
        output: torch.Tensor
            Output of the radial basis function.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class RBF_Identity(RBFBase):
    """
    Identity radial basis function, just return the input.
    """

    def __init__(self):
        super().__init__(RBFType.IDENTITY)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        """Note that the eps is not used in the identity function."""
        return r


class RBF_LinearNegative(RBFBase):
    """
    Linear radial basis function.
    It is also named 1st-order polyharmonic spline, see https://rbf.readthedocs.io/en/latest/basis.html

    Definition:
    output = -er (note that the output is negative)

    Where:
    e=eps_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.LINEAR_NEGATIVE)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        return -r


class RBF_ThinPlateSpline(RBFBase):
    """
    Thin plate spline radial basis function.
    It is also named 2nd-order polyharmonic spline, see https://rbf.readthedocs.io/en/latest/basis.html

    Definition:
    output = (er)^2 * log(er)

    Where:
    e=eps_scale
    r=radial distance

    Note that er will be clamped by eps_guard to avoid log(0).
    """

    def __init__(self, eps_guard: float = 1e-6):
        """
        Initialize the thin plate spline radial basis function.

        parameters
        ----------------
        eps_guard : float
            Small value to avoid log(0).
        """
        super().__init__(RBFType.THIN_PLATE_SPLINE)
        assert eps_guard > 0, "eps_guard must be positive"
        self.register_buffer("_eps_guard", torch.tensor(eps_guard, dtype=torch.float32))

    @property
    def eps_guard(self) -> float:
        """
        Get the epsilon guard value.

        return
        ---------------
        float: Epsilon guard value.
        """
        return self._eps_guard.item()

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        r = torch.clamp(r, min=self.eps_guard)
        return r**2 * torch.log(r)


class RBF_Multiquadric(RBFBase):
    """
    Multiquadric radial basis function.
    It is always negative, range is (-inf, 0).

    Definition:
    output= -sqrt((er)^2 + 1)

    Where:
    e=eps_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.MULTIQUADRIC)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        return -torch.sqrt(r**2 + 1)


class RBF_InverseMultiquadric(RBFBase):
    """
    Inverse multiquadric radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    output=1/sqrt(1+(er)^2)

    Where:
    e=eps_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.INVERSE_MULTIQUADRIC)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        return 1 / torch.sqrt(r**2 + 1)


class RBF_InverseQuadratic(RBFBase):
    """Inverse quadratic radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    output=1/(1+(er)^2)

    Where:
    e=eps_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.INVERSE_QUADRATIC)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        return 1 / (r**2 + 1)


class RBF_Gaussian(RBFBase):
    """Gaussian radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    output=exp(-(er)^2)

    Where:
    e=eps_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.GAUSSIAN)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r * eps
        return torch.exp(-(r**2))


class RBF_Exponential(RBFBase):
    """Exponential radial basis function.
    It is always positive, range is (0, 1).

    Definition:
    output=exp(-r/e)

    Where:
    e=radial_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.EXPONENTIAL)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r / eps
        return torch.exp(-r)


class RBF_SquaredExponential(RBFBase):
    """Squared exponential radial basis function.

    Definition:
    output=exp(-(er)^2)

    Where:
    e=radial_scale
    r=radial distance
    """

    def __init__(self):
        super().__init__(RBFType.SQUARED_EXPONENTIAL)

    def forward(self, r: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        if eps is not None:
            r = r / (2**0.5 * eps)
        return torch.exp(-(r**2))


class RBFFactory:
    @classmethod
    def create_rbf(cls, rbf_type: str) -> RBFBase:
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
            return RBF_ThinPlateSpline()
        elif rbf_type == RBFType.MULTIQUADRIC:
            return RBF_Multiquadric()
        elif rbf_type == RBFType.INVERSE_MULTIQUADRIC:
            return RBF_InverseMultiquadric()
        elif rbf_type == RBFType.INVERSE_QUADRATIC:
            return RBF_InverseQuadratic()
        elif rbf_type == RBFType.GAUSSIAN:
            return RBF_Gaussian()
        elif rbf_type == RBFType.EXPONENTIAL:
            return RBF_Exponential()
        elif rbf_type == RBFType.SQUARED_EXPONENTIAL:
            return RBF_SquaredExponential()
        elif rbf_type == RBFType.LINEAR_NEGATIVE:
            return RBF_LinearNegative()
        elif rbf_type == RBFType.IDENTITY:
            return RBF_Identity()
        else:
            raise ValueError(f"Unknown radial basis function type: {rbf_type}")

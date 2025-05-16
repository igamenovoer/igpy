# TODO: test this
import torch
import torch.nn as nn
import numpy as np
import igpy.util_torch.radial_functions as rbfuncs
from igpy.util_torch.radial_functions import RBFType
import igpy.util_torch.geometry as igt_geom


class ScalarFieldByRBF(nn.Module):
    """A scalar field defined by a sum of Radial Basis Functions (RBFs) and an optional ambient field.
    This module implements a scalar field model using a combination of radial basis functions (RBFs)
    and an optional ambient field. The scalar field is defined as a sum of RBF components, each
    centered at a different location in space, plus an optional ambient field that provides a
    global scalar value. The RBFs and ambient field can be transformed by extrinsic affine
    transformations (rotation, scale, and translation) to allow for flexible shaping of the scalar field.
    The class supports different types of RBFs, specified by the `rbf_type` parameter, and an
    optional ambient field, specified by the `ambient_type` parameter. The extrinsic transformations
    can be either full affine transformations or similarity transformations (rotation, uniform scale, and translation).
    Attributes:
        EpsScaleInitialRange (Tuple[float, float]): Default range for initializing the epsilon scale parameter.
    Args:
        point_dim (int): Dimensionality of the input points (e.g., 3 for 3D points).
        n_rbf_components (int): Number of RBF components in the scalar field.
        n_ambient_components (int): Number of ambient components in the scalar field.
        rbf_type (str): Type of RBF to use (e.g., 'gaussian', 'thin_plate_spline').
        ambient_type (str): Type of ambient field to use (e.g., 'identity', 'constant').
        enforce_similarity_transformation (bool): Whether to enforce a similarity transformation
            (rotation, uniform scale, and translation) for the extrinsic transformations. If False,
            a full affine transformation is used.
    """

    # default settings

    # eps scale default range
    EpsScaleDefaultRange = (1e-3, 1e3)

    # add this in initialization to prevent overflow
    EpsScaleSmallDelta = 1e-3

    def __init__(
        self,
        point_dim: int,
        n_rbf_components: int,
        n_ambient_components: int,
        rbf_type: str = RBFType.EXPONENTIAL,
        ambient_type: str = RBFType.IDENTITY,
        scalar_expected_range: tuple | None = None,
        coordinate_expected_range: np.ndarray | None = None,
        enforce_similarity_transformation: bool = False,
    ):
        """
        Initialize the ScalarFieldByRBF class.

        parameters
        -----------------
        point_dim : int
            Dimensionality of the input points (e.g., 3 for 3D points).
        n_rbf_components : int
            Number of RBF components in the scalar field.
        n_ambient_components : int
            Number of ambient components in the scalar field.
        rbf_type : str
            Type of RBF to use, see RBFType for available types.
        ambient_type : str
            Type of ambient field to use, see RBFType for available types.
        scalar_expected_range : tuple | None
            Expected range of the scalar field values, which will affect the initialization of the RBF and ambient components.
            If not given, use the default range, see self.EpsScaleDefaultRange.
        coordinate_expected_range : np.ndarray | None
            NxD array or 1xD array with auto broadcasting, expected range of the coordinates of the points.
            This is used to initialize the RBF and ambient components.
            coordinate_expected_range[k]=(min, max) for the k-th dimension.
            If not given, the range is assumed to be (-1, 1) for each dimension.
        enforce_similarity_transformation : bool
            Whether to enforce a similarity transformation (rotation, uniform scale, and translation)
            for the extrinsic transformations. If False, a full affine transformation is used.
        """
        super().__init__()

        self.register_buffer("_n_rbf_components", torch.tensor(n_rbf_components))
        self.register_buffer("_n_ambient_components", torch.tensor(n_ambient_components))
        self.register_buffer("_point_dim", torch.tensor(point_dim))

        # ascii encoding of the rbf type
        rbf_type_encoded = torch.tensor([ord(c) for c in rbf_type], dtype=torch.int32)
        self.register_buffer("_rbf_type", rbf_type_encoded)

        # ascii encoding of the ambient type
        ambient_type_encoded = torch.tensor([ord(c) for c in ambient_type], dtype=torch.int32)
        self.register_buffer("_ambient_type", ambient_type_encoded)

        # the expected range of the scalar field values
        # eps_logspace_range: np.ndarray = np.log10(np.array(self.EpsScaleDefaultRange))
        if scalar_expected_range is None:
            scalar_expected_range = self.EpsScaleDefaultRange
        self.register_buffer("_scalar_expected_range", torch.tensor(scalar_expected_range, dtype=torch.float32))

        # if the expected range is given, use it to set the eps scale range
        abs_min = np.min(np.abs(scalar_expected_range))
        abs_max = np.max(np.abs(scalar_expected_range))
        if scalar_expected_range[0] * scalar_expected_range[1] < 0:
            abs_min = 0
        eps_logspace_range = np.log10(np.array((abs_min, abs_max)) + self.EpsScaleSmallDelta)

        if coordinate_expected_range is None:
            coordinate_expected_range = np.array([[-1, 1]] * point_dim)
        coordinate_expected_range = np.atleast_1d(coordinate_expected_range)

        if coordinate_expected_range.ndim == 1:
            coordinate_expected_range = np.array([coordinate_expected_range] * point_dim)

        assert coordinate_expected_range.shape[0] == point_dim, "coordinate_expected_range should be NxD"
        self.register_buffer("_coordinate_expected_range", torch.tensor(coordinate_expected_range, dtype=torch.float32))

        # create parameters
        self.rbf_extrinsic_affine: nn.Parameter = None
        self.rbf_extrinsic_translation: nn.Parameter = None
        self.rbf_post_scale: nn.Parameter = None

        # when used, should be raised to power of 2
        # this make sure the eps scale is positive
        self.rbf_eps_scale_sqrt: nn.Parameter = None

        # for similarity transformation
        self.rbf_extrinsic_quaternions: nn.Parameter = None
        self.rbf_extrinsic_scales: nn.Parameter = None

        if n_rbf_components > 0:
            init_translation = np.random.randn(n_rbf_components, point_dim)
            for i in range(point_dim):
                init_translation[:, i] = np.random.uniform(
                    coordinate_expected_range[i, 0], coordinate_expected_range[i, 1], n_rbf_components
                )

            # negate of the translation because this is extrinsic
            self.rbf_extrinsic_translation = nn.Parameter(torch.tensor(-init_translation, dtype=torch.float32))

            if enforce_similarity_transformation:
                # only applicable in 3d
                assert point_dim == 3, "Similarity transformation is only supported in 3D"

                # when we use simliarity transformation, the affine matrix is defined by rotation and scale
                self.rbf_extrinsic_quaternions = nn.Parameter(torch.randn(n_rbf_components, 4))
                self.rbf_extrinsic_scales = nn.Parameter(torch.randn(n_rbf_components, 3))
            else:
                # if we don't want to enforce similarity transformation, we can use any matrix
                # so we just use a random matrix
                self.rbf_extrinsic_affine = nn.Parameter(torch.randn(n_rbf_components, point_dim, point_dim))

            # scale for the rbf, used to scale the output of the rbf
            self.rbf_post_scale = nn.Parameter(torch.randn(n_rbf_components))

            # eps scale for the rbf, should be > 0
            self.rbf_eps_scale_sqrt = nn.Parameter(
                torch.sqrt(torch.exp(torch.tensor(np.random.uniform(*eps_logspace_range, n_rbf_components)))).float()
            )

        # for ambient field
        self.ambient_extrinsic_affine: nn.Parameter = None
        self.ambient_extrinsic_translation: nn.Parameter = None
        self.ambient_post_scale: nn.Parameter = None
        self.ambient_eps_scale_sqrt: nn.Parameter = None  # when used, should be raised to power of 2
        self.ambient_extrinsic_quaternions: nn.Parameter = None
        self.ambient_extrinsic_scales: nn.Parameter = None

        if n_ambient_components > 0:
            init_translation = np.random.randn(n_ambient_components, point_dim)
            for i in range(point_dim):
                init_translation[:, i] = np.random.uniform(
                    coordinate_expected_range[i, 0], coordinate_expected_range[i, 1], n_ambient_components
                )
            self.ambient_extrinsic_translation = nn.Parameter(torch.tensor(-init_translation, dtype=torch.float32))
            if enforce_similarity_transformation:
                # only applicable in 3d
                assert point_dim == 3, "Similarity transformation is only supported in 3D"

                # when we use simliarity transformation, the affine matrix is defined by rotation and scale
                self.ambient_extrinsic_quaternions = nn.Parameter(torch.randn(n_ambient_components, 4))
                self.ambient_extrinsic_scales = nn.Parameter(torch.randn(n_ambient_components, 3))
            else:
                # if we don't want to enforce similarity transformation, we can use any matrix
                # so we just use a random matrix
                self.ambient_extrinsic_affine = nn.Parameter(torch.randn(n_ambient_components, point_dim, point_dim))

            # scale for the ambient, used to scale the output of the ambient
            self.ambient_post_scale = nn.Parameter(torch.randn(n_ambient_components))

            # eps scale for the ambient, should be > 0
            # this is used to scale the output of the ambient
            self.ambient_eps_scale_sqrt = nn.Parameter(
                torch.sqrt(
                    torch.exp(torch.tensor(np.random.uniform(*eps_logspace_range, n_ambient_components)))
                ).float()
            )

        # done with initialization

    @property
    def n_rbf_components(self) -> int:
        """
        Get the number of RBF components.

        returns
        -----------------
        n_rbf_components : int
            The number of RBF components
        """
        return self._n_rbf_components.item()

    @property
    def n_ambient_components(self) -> int:
        """
        Get the number of ambient components.

        returns
        -----------------
        n_ambient_components : int
            The number of ambient components
        """
        return self._n_ambient_components.item()

    @property
    def point_dim(self) -> int:
        """
        Get the dimensionality of the input points.

        returns
        -----------------
        point_dim : int
            The dimensionality of the input points
        """
        return self._point_dim.item()

    @property
    def rbf_type(self) -> str:
        """
        Get the type of the radial basis function.

        returns
        -----------------
        rbf_type : str
            The type of the radial basis function
        """
        # this is in the registered buffer, which is a string encoded as ascii with ord
        # so we need to decode it
        rbf_type_decoded = "".join([chr(c.item()) for c in self._rbf_type])
        return rbf_type_decoded

    @property
    def ambient_type(self) -> str:
        """
        Get the type of the ambient function.

        returns
        -----------------
        ambient_type : str
            The type of the ambient function
        """
        # this is in the registered buffer, which is a string encoded as ascii with ord
        # so we need to decode it
        ambient_type_decoded = "".join([chr(c.item()) for c in self._ambient_type])
        return ambient_type_decoded

    def _get_rbf_extrinsic_affine(self) -> torch.Tensor:
        """
        Get the rbf extrinsic affine matrix.

        returns
        -----------------
        rbf_extrinsic_affine : torch.Tensor, shape (N, D, D)
            The rbf extrinsic affine matrix
        """
        if self.rbf_extrinsic_affine is not None:
            return self.rbf_extrinsic_affine
        else:
            # construct it from quaternion and scales
            assert hasattr(self, "rbf_extrinsic_quaternions"), "rbf_extrinsic_quaternions is not set"
            assert hasattr(self, "rbf_extrinsic_scales"), "rbf_extrinsic_scales is not set"
            rbf_extrinsic_affine = igt_geom.create_similarity_transform_4x4(
                quaternions=self.rbf_extrinsic_quaternions, scales=self.rbf_extrinsic_scales
            )[:, :3, :3]
            return rbf_extrinsic_affine

    def _get_ambient_extrinsic_affine(self) -> torch.Tensor:
        """
        Get the ambient extrinsic affine matrix.

        returns
        -----------------
        ambient_extrinsic_affine : torch.Tensor, shape (N, D, D)
            The ambient extrinsic affine matrix
        """
        if self.ambient_extrinsic_affine is not None:
            return self.ambient_extrinsic_affine
        else:
            # construct it from quaternion and scales
            assert hasattr(self, "ambient_extrinsic_quaternions"), "ambient_extrinsic_quaternions is not set"
            assert hasattr(self, "ambient_extrinsic_scales"), "ambient_extrinsic_scales is not set"
            ambient_extrinsic_affine = igt_geom.create_similarity_transform_4x4(
                quaternions=self.ambient_extrinsic_quaternions, scales=self.ambient_extrinsic_scales
            )[:, :3, :3]
            return ambient_extrinsic_affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """evaluate the scalar field at the points x. For each point, it is first transformed by the rbf_extrinsic_affine and rbf_extrinsic_translation,
        and gets the rbf value. Then, the ambient field is evaluated similarly, and the two are summed up.

        For mutiple rbf components, the rbf values are averaged over the components.

        parameters
        -----------------
        x : torch.Tensor, shape (N, D)
            The points to evaluate the scalar field at, for N points in D-dimensional space

        returns
        -----------------
        field_values : torch.Tensor, shape (N,)
            The scalar field values at the points
        """
        output: torch.Tensor = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if self.n_rbf_components > 0:
            sdf_by_rbf = self.evaluate_rbf(x)  # (N, M)
            # sum over the rbf components
            output += torch.mean(sdf_by_rbf, dim=-1)  # (N, M) -> (N,)

        if self.n_ambient_components > 0:
            sdf_by_ambient = self.evaluate_ambient(x)
            # sum over the ambient components
            output += torch.mean(sdf_by_ambient, dim=-1)

        return output

    def get_rbf_transformed_points(self, x: torch.Tensor) -> torch.Tensor:
        """transform the points x by the rbf_extrinsic_affine and rbf_extrinsic_translation

        parameters
        -----------------
        x : torch.Tensor, shape (N, D)
            The points to transform, for N points in D-dimensional space

        returns
        -----------------
        transformed_points : torch.Tensor, shape (N, M, D)
            The transformed points, for each of N points and each of M rbf components
        """
        assert x.shape[-1] == self.point_dim, f"point_dim is {self.point_dim}, but x.shape[-1] is {x.shape[-1]}"
        # transform
        affine_transform = self._get_rbf_extrinsic_affine()
        transformed_points = igt_geom.transform_points_linear_all_to_all(
            x, affine_transform, self.rbf_extrinsic_translation
        )
        # transformed_points = x @ affine_transform + self.rbf_extrinsic_translation
        # shape is (N, M, D)
        return transformed_points

    def get_ambient_transformed_points(self, x: torch.Tensor) -> torch.Tensor:
        """transform the points x by the ambient_extrinsic_affine and ambient_extrinsic_translation

        parameters
        -----------------
        x : torch.Tensor, shape (N, D)
            The points to transform, for N points in D-dimensional space

        returns
        -----------------
        transformed_points : torch.Tensor, shape (N, A, D)
            The transformed points, for each of N points and each of A ambient components
        """
        assert x.shape[-1] == self.point_dim, f"point_dim is {self.point_dim}, but x.shape[-1] is {x.shape[-1]}"
        # transform
        affine_transform = self._get_ambient_extrinsic_affine()
        transformed_points = igt_geom.transform_points_linear_all_to_all(
            x, affine_transform, self.ambient_extrinsic_translation
        )
        # transformed_points = x @ affine_transform + self.ambient_extrinsic_translation
        # shape is (N, A, D)
        return transformed_points

    def evaluate_rbf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the rbf at the points x.

        Each point is transformed by the rbf_extrinsic_affine and rbf_extrinsic_translation,
        and then the rbf is evaluated. Given (N,D) points, the result is (N,M) values,
        where M is the number of rbf components.

        Let P=x[i], one of the input point, and let E=extrinsic_affine, d=extrinsic_translation.
        Then, the transformed point is given by:
        P_new = E@P + d
        The rbf value is then given by:
        rbf_value = rbf(||P_new||)

        parameters
        -----------------
        x : torch.Tensor, shape (N, D)
            The points to evaluate the rbf at, for N points in D-dimensional space

        returns
        -----------------
        rbf_values : torch.Tensor, shape (N,M)
            The rbf values at the points, for each of N points and each of M rbf components
        """
        assert x.shape[-1] == self.point_dim, f"point_dim is {self.point_dim}, but x.shape[-1] is {x.shape[-1]}"

        # transform all the points, x_transformed = x@E + d, shape is (N,M,D)
        affine_transform = self._get_rbf_extrinsic_affine()
        x_transformed = igt_geom.transform_points_linear_all_to_all(x, affine_transform, self.rbf_extrinsic_translation)

        # compute the squared distance, result is (N,M)
        # radial_value = torch.norm(x_transformed, dim=-1, p=2)
        radial_value = torch.linalg.norm(x_transformed, dim=-1, ord=2)

        # evaluate the rbf function
        func = rbfuncs.RBFFactory.create_rbf(self.rbf_type)
        rbf_values = func(radial_value, eps=self.rbf_eps_scale_sqrt.view(1, -1) ** 2)  # (N,M)

        # scale the rbf values
        if self.rbf_post_scale is not None:
            rbf_values = rbf_values * self.rbf_post_scale.view(1, -1)
        # (N,M) * (M,) -> (N,M)

        return rbf_values

    def evaluate_ambient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the ambient field at the points x.

        Each point is transformed by the ambient_extrinsic_affine and ambient_extrinsic_translation,
        and then the ambient field is evaluated. Given (N,D) points, the result is (N,A) values,
        where A is the number of ambient components.

        Let P=x[i], one of the input point, and let E=ambient_extrinsic_affine, d=ambient_extrinsic_translation.
        Then, the transformed point is given by:
        P_new = E@P + d
        And the rbf value is given by:
        rbf_value = rbf(||P_new||)

        parameters
        -----------------
        x : torch.Tensor, shape (N, D)
            The points to evaluate the ambient field at, for N points in D-dimensional space

        returns
        -----------------
        ambient_values : torch.Tensor, shape (N, A)
            The ambient field values at the points, for each of N points and each of A ambient components
        """

        assert x.shape[-1] == self.point_dim, f"point_dim is {self.point_dim}, but x.shape[-1] is {x.shape[-1]}"

        # transform all the points, x_transformed = x@E + d, shape is (N,D)
        affine_transform = self._get_ambient_extrinsic_affine()
        x_transformed = igt_geom.transform_points_linear_all_to_all(
            x, affine_transform, self.ambient_extrinsic_translation
        )

        # compute the radial distance
        radial_value = torch.norm(x_transformed, dim=-1, p=2)

        # evaluate the ambient function
        func = rbfuncs.RBFFactory.create_rbf(self.ambient_type)
        ambient_values = func(radial_value, eps=self.ambient_eps_scale_sqrt.view(1, -1) ** 2)
        return ambient_values

# TODO: test this
import torch
import torch.nn as nn
import igpy.util_torch.radial_functions as rbfuncs
from igpy.util_torch.radial_functions import RBFType
import igpy.util_torch.geometry as igt_geom


class ScalarFieldByRBF(nn.Module):
    def __init__(
        self,
        point_dim: int,
        n_rbf_components: int,
        n_ambient_components: int,
        rbf_type: str = RBFType.GAUSSIAN,
        ambient_type: str = RBFType.IDENTITY,
        enforce_similarity_transformation: bool = False,
    ):
        super().__init__()

        self.register_buffer("n_rbf_components", torch.tensor(n_rbf_components))
        self.register_buffer("n_ambient_components", torch.tensor(n_ambient_components))
        self.register_buffer("point_dim", torch.tensor(point_dim))

        # ascii encoding of the rbf type
        rbf_type_encoded = torch.tensor([ord(c) for c in rbf_type], dtype=torch.int32)
        self.register_buffer("rbf_type_encoded", rbf_type_encoded)

        # ascii encoding of the ambient type
        ambient_type_encoded = torch.tensor([ord(c) for c in ambient_type], dtype=torch.int32)
        self.register_buffer("ambient_type_encoded", ambient_type_encoded)

        # create parameters
        if n_rbf_components > 0:
            self.rbf_extrinsic_translation = nn.Parameter(torch.randn(n_rbf_components, point_dim))
            if enforce_similarity_transformation:
                # only applicable in 3d
                assert point_dim == 3, "Similarity transformation is only supported in 3D"

                # when we use simliarity transformation, the affine matrix is defined by rotation and scale
                self.rbf_extrinsic_quaternions = nn.Parameter(torch.randn(n_rbf_components, 4))
                self.rbf_extrinsic_scales = nn.Parameter(torch.randn(n_rbf_components, 3))
                self.rbf_extrinsic_affine = None
            else:
                # if we don't want to enforce similarity transformation, we can use any matrix
                # so we just use a random matrix
                self.rbf_extrinsic_affine = nn.Parameter(torch.randn(n_rbf_components, point_dim, point_dim))

            # scale for the rbf, used to scale the output of the rbf
            self.rbf_post_scale = nn.Parameter(torch.randn(n_rbf_components))

            # scale for the radial distance, used to scale the input of the rbf
            self.rbf_radial_scale = nn.Parameter(torch.randn(n_rbf_components))
        else:
            self.rbf_extrinsic_affine: nn.Parameter = None
            self.rbf_extrinsic_translation: nn.Parameter = None
            self.rbf_post_scale: nn.Parameter = None

        if n_ambient_components > 0:
            self.ambient_extrinsic_translation = nn.Parameter(torch.randn(n_ambient_components, point_dim))
            if enforce_similarity_transformation:
                # only applicable in 3d
                assert point_dim == 3, "Similarity transformation is only supported in 3D"

                # when we use simliarity transformation, the affine matrix is defined by rotation and scale
                self.ambient_extrinsic_quaternions = nn.Parameter(torch.randn(n_ambient_components, 4))
                self.ambient_extrinsic_scales = nn.Parameter(torch.randn(n_ambient_components, 3))
                self.ambient_extrinsic_affine = None
            else:
                # if we don't want to enforce similarity transformation, we can use any matrix
                # so we just use a random matrix
                self.ambient_extrinsic_affine = nn.Parameter(torch.randn(n_ambient_components, point_dim, point_dim))

            # scale for the ambient, used to scale the output of the ambient
            self.ambient_post_scale = nn.Parameter(torch.randn(n_ambient_components))
        else:
            self.ambient_extrinsic_affine: nn.Parameter = None
            self.ambient_extrinsic_translation: nn.Parameter = None
            self.ambient_post_scale: nn.Parameter = None

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
        rbf_type_encoded = self.rbf_type_encoded
        rbf_type_decoded = "".join([chr(c.item()) for c in rbf_type_encoded])
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
        ambient_type_encoded = self.ambient_type_encoded
        ambient_type_decoded = "".join([chr(c.item()) for c in ambient_type_encoded])
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

    def set_extrinsics_to_identity(self):
        """
        Set the extrinsic affine matrices to identity matrices, and the translations to zero.
        """
        # Set affine matrices to identity
        if self.rbf_extrinsic_affine is not None:
            batch_size_rbf = self.rbf_extrinsic_affine.shape[0]
            dim = self.rbf_extrinsic_affine.shape[1]

            # Create identity matrices for each batch element
            identity_rbf = (
                torch.eye(dim, device=self.rbf_extrinsic_affine.device).unsqueeze(0).expand(batch_size_rbf, -1, -1)
            )

            # Set affine matrices to identity
            self.rbf_extrinsic_affine.data.copy_(identity_rbf)

            # Set translations to zero
            self.rbf_extrinsic_translation.data.zero_()

            # Set scale to 1
            self.rbf_post_scale.data.fill_(1.0)
        else:
            # If rbf_extrinsic_affine is None, set the quaternion and scale to identity
            batch_size_rbf = self.rbf_extrinsic_quaternions.shape[0]
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.rbf_extrinsic_quaternions.device).expand(
                batch_size_rbf, -1
            )
            self.rbf_extrinsic_quaternions.data.copy_(identity_quat)
            self.rbf_extrinsic_scales.data.fill_(1.0)
            self.rbf_extrinsic_translation.data.zero_()
            self.rbf_post_scale.data.fill_(1.0)

        if self.ambient_extrinsic_affine is not None:
            batch_size_ambient = self.ambient_extrinsic_affine.shape[0]
            dim = self.ambient_extrinsic_affine.shape[1]
            identity_ambient = (
                torch.eye(dim, device=self.ambient_extrinsic_affine.device)
                .unsqueeze(0)
                .expand(batch_size_ambient, -1, -1)
            )
            self.ambient_extrinsic_affine.data.copy_(identity_ambient)
            self.ambient_extrinsic_translation.data.zero_()
            self.ambient_post_scale.data.fill_(1.0)
        else:
            # If ambient_extrinsic_affine is None, set the quaternion and scale to identity
            batch_size_ambient = self.ambient_extrinsic_quaternions.shape[0]
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.ambient_extrinsic_quaternions.device).expand(
                batch_size_ambient, -1
            )
            self.ambient_extrinsic_quaternions.data.copy_(identity_quat)
            self.ambient_extrinsic_scales.data.fill_(1.0)
            self.ambient_extrinsic_translation.data.zero_()
            self.ambient_post_scale.data.fill_(1.0)

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

        # scale the radial distance, use inverted scale
        if self.rbf_radial_scale is not None:
            radial_value = radial_value / self.rbf_radial_scale.view(1, -1)

        # evaluate the rbf function
        func = rbfuncs.RBFFactory.create_rbf(self.rbf_type)
        rbf_values = func(radial_value)  # (N,M)

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
        ambient_values = func(radial_value)
        return ambient_values

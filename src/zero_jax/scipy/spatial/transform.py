"""SciPy spatial transform submodule."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.ops.registry import get_op

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


class Rotation:
    """Rotation in 3 dimensions.

    This class provides a frontend interface for 3D rotations, matching `jax.scipy.spatial.transform.Rotation`.
    """

    def __init__(self, quat: ArrayLikeSpec, normalize: bool = True) -> None:
        """Initialize the rotation with quaternions.

        Args:
            quat: The quaternions representing the rotation.
            normalize: Whether to normalize the quaternions.
        """
        self._quat = quat
        # Normalization omitted in shell init unless strictly necessary
        self._normalize = normalize

    @classmethod
    def from_quat(cls, quat: ArrayLikeSpec) -> Rotation:
        """Initialize from quaternions.

        Args:
            quat: The quaternions to use.

        Returns:
            A new Rotation object.
        """
        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix: ArrayLikeSpec) -> Rotation:
        """Initialize from rotation matrix.

        Args:
            matrix: The rotation matrix.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationFromMatrix")
        return cls(_wrap(op(_to_tensor(matrix))))

    @classmethod
    def from_rotvec(cls, rotvec: ArrayLikeSpec) -> Rotation:
        """Initialize from rotation vector.

        Args:
            rotvec: The rotation vector.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationFromRotvec")
        return cls(_wrap(op(_to_tensor(rotvec))))

    @classmethod
    def from_mrp(cls, mrp: ArrayLikeSpec) -> Rotation:
        """Initialize from Modified Rodrigues Parameters (MRPs).

        Args:
            mrp: The MRPs.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationFromMrp")
        return cls(_wrap(op(_to_tensor(mrp))))

    @classmethod
    def from_euler(cls, seq: str, angles: ArrayLikeSpec, degrees: bool = False) -> Rotation:
        """Initialize from Euler angles.

        Args:
            seq: Sequence of rotation axes.
            angles: Euler angles.
            degrees: Whether angles are in degrees.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationFromEuler")
        return cls(_wrap(op(seq, _to_tensor(angles), degrees)))

    @classmethod
    def identity(cls, num: int | None = None) -> Rotation:
        """Initialize with identity rotation.

        Args:
            num: Number of identity rotations to generate.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationIdentity")
        return cls(_wrap(op(num)))

    @classmethod
    def random(cls, num: int | None = None, random_state: ArrayLikeSpec = None) -> Rotation:
        """Generate uniformly distributed rotations.

        Args:
            num: Number of random rotations to generate.
            random_state: PRNG state for randomness.

        Returns:
            A new Rotation object.
        """
        op = get_op("ScipySpatialRotationRandom")
        return cls(_wrap(op(num, random_state)))

    @classmethod
    def align_vectors(cls, a: ArrayLikeSpec, b: ArrayLikeSpec, weights: ArrayLikeSpec = None, return_sensitivity: bool = False) -> ArrayLikeSpec:
        """Estimate optimal rotation to align two sets of vectors.

        Args:
            a: First set of vectors.
            b: Second set of vectors.
            weights: Weights for each vector pair.
            return_sensitivity: Whether to return sensitivity matrix.

        Returns:
            Optimal rotation and optionally sensitivity.
        """
        op = get_op("ScipySpatialRotationAlignVectors")
        return _wrap(
            op(
                _to_tensor(a),
                _to_tensor(b),
                _to_tensor(weights) if weights is not None else None,
                return_sensitivity,
            )
        )

    @property
    def single(self) -> bool:
        """Check whether the rotation contains a single representation.

        Returns:
            True if single, False otherwise.
        """
        try:
            return self._quat.ndim == 1
        except AttributeError:
            return True

    def as_quat(self) -> ArrayLikeSpec:
        """Represent as quaternions.

        Returns:
            The quaternions.
        """
        return self._quat

    def as_matrix(self) -> ArrayLikeSpec:
        """Represent as rotation matrix.

        Returns:
            The rotation matrix.
        """
        op = get_op("ScipySpatialRotationAsMatrix")
        return _wrap(op(_to_tensor(self._quat)))

    def as_rotvec(self) -> ArrayLikeSpec:
        """Represent as rotation vector.

        Returns:
            The rotation vector.
        """
        op = get_op("ScipySpatialRotationAsRotvec")
        return _wrap(op(_to_tensor(self._quat)))

    def as_mrp(self) -> ArrayLikeSpec:
        """Represent as Modified Rodrigues Parameters (MRPs).

        Returns:
            The MRPs.
        """
        op = get_op("ScipySpatialRotationAsMrp")
        return _wrap(op(_to_tensor(self._quat)))

    def as_euler(self, seq: str, degrees: bool = False) -> ArrayLikeSpec:
        """Represent as Euler angles.

        Args:
            seq: Sequence of rotation axes.
            degrees: Whether to return angles in degrees.

        Returns:
            The Euler angles.
        """
        op = get_op("ScipySpatialRotationAsEuler")
        return _wrap(op(_to_tensor(self._quat), seq, degrees))

    def apply(self, vectors: ArrayLikeSpec, inverse: bool = False) -> ArrayLikeSpec:
        """Apply the rotation to a set of vectors.

        Args:
            vectors: The vectors to rotate.
            inverse: Whether to apply the inverse rotation.

        Returns:
            The rotated vectors.
        """
        op = get_op("ScipySpatialRotationApply")
        return _wrap(op(_to_tensor(self._quat), _to_tensor(vectors), inverse))

    def inv(self) -> Rotation:
        """Invert the rotation.

        Returns:
            The inverse rotation.
        """
        op = get_op("ScipySpatialRotationInv")
        return Rotation(_wrap(op(_to_tensor(self._quat))))

    def magnitude(self) -> ArrayLikeSpec:
        """Compute the rotation magnitude values.

        Returns:
            The rotation magnitude values.
        """
        op = get_op("ScipySpatialRotationMagnitude")
        return _wrap(op(_to_tensor(self._quat)))

    def mean(self, weights: ArrayLikeSpec = None) -> Rotation:
        """Get the mean of the rotations.

        Args:
            weights: Weights for each rotation.

        Returns:
            The mean rotation.
        """
        op = get_op("ScipySpatialRotationMean")
        return Rotation(
            _wrap(
                op(
                    _to_tensor(self._quat),
                    _to_tensor(weights) if weights is not None else None,
                )
            )
        )

    def __mul__(self, other: Rotation) -> Rotation:
        """Compose this rotation with another.

        Args:
            other: The other rotation.

        Returns:
            The composed rotation.
        """
        op = get_op("ScipySpatialRotationMul")
        return Rotation(_wrap(op(_to_tensor(self._quat), _to_tensor(other._quat))))

    def __len__(self) -> int:
        """Get the number of rotations.

        Returns:
            The number of rotations.
        """
        try:
            return len(self._quat)
        except TypeError:
            return 1

    @classmethod
    def concatenate(cls, rotations: Sequence[Rotation]) -> Rotation:
        """Concatenate a sequence of rotations.

        Args:
            rotations: The sequence of rotations.

        Returns:
            The concatenated rotations.
        """
        op = get_op("ScipySpatialRotationConcatenate")
        quats = [_to_tensor(r._quat) for r in rotations]
        return cls(_wrap(op(*quats)))


class Slerp:
    """Spherical Linear Interpolation of Rotations.

    This class provides a frontend interface for interpolating rotations.
    """

    def __init__(self, times: ArrayLikeSpec, rotations: Rotation) -> None:
        """Initialize the interpolator.

        Args:
            times: The times corresponding to each rotation.
            rotations: The rotations to interpolate between.
        """
        self.times = times
        self.rotations = rotations

    def __call__(self, times: ArrayLikeSpec) -> Rotation:
        """Interpolate the rotations at the given times.

        Args:
            times: The times to interpolate at.

        Returns:
            The interpolated rotations.
        """
        op = get_op("ScipySpatialSlerpCall")
        return Rotation(
            _wrap(
                op(
                    _to_tensor(self.times),
                    _to_tensor(self.rotations._quat),
                    _to_tensor(times),
                )
            )
        )


__all__ = ["Rotation", "Slerp"]

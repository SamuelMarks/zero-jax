"""Tests for scipy.spatial.transform module."""

from unittest import mock

import zero_jax.numpy as jnp
import zero_jax.scipy.spatial.transform as mod


def test_module_exists() -> None:
    """Test module imports correctly."""
    assert mod is not None


@mock.patch("zero_jax.scipy.spatial.transform.get_op")
def test_rotation_methods(mock_get_op: mock.MagicMock) -> None:
    """Test all Rotation methods."""
    mock_op_instance = mock.MagicMock()
    mock_get_op.return_value = mock_op_instance
    mock_op_instance.return_value = jnp.array([0.0, 0.0, 0.0, 1.0])

    quat = jnp.array([0.0, 0.0, 0.0, 1.0])

    # Class methods
    rot1 = mod.Rotation.from_quat(quat)
    assert jnp.allclose(rot1.as_quat(), quat)

    rot = mod.Rotation.from_matrix(jnp.eye(3))
    mock_get_op.assert_called_with("ScipySpatialRotationFromMatrix")
    assert mock_op_instance.call_count == 1

    rot = mod.Rotation.from_rotvec(jnp.array([0.1, 0.2, 0.3]))
    mock_get_op.assert_called_with("ScipySpatialRotationFromRotvec")

    rot = mod.Rotation.from_mrp(jnp.array([0.1, 0.2, 0.3]))
    mock_get_op.assert_called_with("ScipySpatialRotationFromMrp")

    rot = mod.Rotation.from_euler("xyz", jnp.array([0.1, 0.2, 0.3]), degrees=True)
    mock_get_op.assert_called_with("ScipySpatialRotationFromEuler")

    rot = mod.Rotation.identity(num=2)
    mock_get_op.assert_called_with("ScipySpatialRotationIdentity")

    rot = mod.Rotation.random(num=3, random_state=42)
    mock_get_op.assert_called_with("ScipySpatialRotationRandom")

    res = mod.Rotation.align_vectors(
        jnp.array([[1.0, 0.0, 0.0]]),
        jnp.array([[0.0, 1.0, 0.0]]),
        weights=jnp.array([1.0]),
        return_sensitivity=True,
    )
    mock_get_op.assert_called_with("ScipySpatialRotationAlignVectors")

    # Instance methods
    rot = mod.Rotation.from_quat(quat)

    _ = rot.single
    # property check

    rot.as_matrix()
    mock_get_op.assert_called_with("ScipySpatialRotationAsMatrix")

    rot.as_rotvec()
    mock_get_op.assert_called_with("ScipySpatialRotationAsRotvec")

    rot.as_mrp()
    mock_get_op.assert_called_with("ScipySpatialRotationAsMrp")

    rot.as_euler("xyz", degrees=False)
    mock_get_op.assert_called_with("ScipySpatialRotationAsEuler")

    rot.apply(jnp.array([1.0, 0.0, 0.0]), inverse=True)
    mock_get_op.assert_called_with("ScipySpatialRotationApply")

    rot.inv()
    mock_get_op.assert_called_with("ScipySpatialRotationInv")

    rot.magnitude()
    mock_get_op.assert_called_with("ScipySpatialRotationMagnitude")

    rot.mean(weights=jnp.array([1.0]))
    mock_get_op.assert_called_with("ScipySpatialRotationMean")

    rot * rot
    mock_get_op.assert_called_with("ScipySpatialRotationMul")

    len(rot)

    mod.Rotation.concatenate([rot, rot])
    mock_get_op.assert_called_with("ScipySpatialRotationConcatenate")


@mock.patch("zero_jax.scipy.spatial.transform.get_op")
def test_slerp(mock_get_op: mock.MagicMock) -> None:
    """Test Slerp class."""
    mock_op_instance = mock.MagicMock()
    mock_get_op.return_value = mock_op_instance
    mock_op_instance.return_value = jnp.array([0.0, 0.0, 0.0, 1.0])

    quat = jnp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    times = jnp.array([0.0, 1.0])
    rot = mod.Rotation.from_quat(quat)

    slerp = mod.Slerp(times, rot)
    assert slerp is not None

    res = slerp(jnp.array([0.5]))
    mock_get_op.assert_called_with("ScipySpatialSlerpCall")
    assert isinstance(res, mod.Rotation)


def test_rotation_len_single_fallback() -> None:
    """Test Rotation single/len fallbacks for scalars."""

    # Test property fallback
    class DummyQuat:
        pass

    rot = mod.Rotation.from_quat(DummyQuat())
    assert rot.single is True
    assert len(rot) == 1


def test_align_vectors_none_weights() -> None:
    """Test align_vectors with None weights."""
    with mock.patch("zero_jax.scipy.spatial.transform.get_op") as mock_get_op:
        mock_op_instance = mock.MagicMock()
        mock_get_op.return_value = mock_op_instance

        mod.Rotation.align_vectors(
            jnp.array([[1.0, 0.0, 0.0]]), jnp.array([[0.0, 1.0, 0.0]]), weights=None
        )
        assert mock_op_instance.call_args[0][2] is None


def test_mean_none_weights() -> None:
    """Test mean with None weights."""
    with mock.patch("zero_jax.scipy.spatial.transform.get_op") as mock_get_op:
        mock_op_instance = mock.MagicMock()
        mock_get_op.return_value = mock_op_instance
        mock_op_instance.return_value = jnp.array([0.0, 0.0, 0.0, 1.0])

        rot = mod.Rotation.from_quat(jnp.array([0.0, 0.0, 0.0, 1.0]))
        rot.mean(weights=None)
        assert mock_op_instance.call_args[0][1] is None

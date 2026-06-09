import zero_jax


def test_devices():
    devices = zero_jax.devices()
    assert len(devices) == 1
    assert devices[0].platform == "cpu"

    local_devs = zero_jax.local_devices()
    assert len(local_devs) == 1
    assert local_devs[0].platform == "cpu"

    dev = zero_jax.Device("gpu")
    assert dev.platform == "gpu"

    assert zero_jax.device_get(5) == 5

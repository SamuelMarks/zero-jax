"""Tests for zero_optax.schedules."""

from zero_optax import schedules


def test_constant_schedule():
    """Test constant schedule."""
    sched = schedules.constant_schedule(5.0)
    assert sched(0) == 5.0
    assert sched(10) == 5.0


def test_cosine_schedules():
    """Test cosine schedules."""
    sched1 = schedules.cosine_decay_schedule(1.0, 10)
    assert sched1(0) == 1.0
    assert sched1(10) == 0.0

    sched2 = schedules.cosine_onecycle_schedule(10, 1.0)
    assert sched2(0) < 1.0
    assert sched2(10) < 1.0
    assert sched2(3) == 1.0  # peak


def test_exponential_decay():
    """Test exponential decay."""
    sched1 = schedules.exponential_decay(1.0, 10, 0.5)
    assert sched1(0) == 1.0
    assert sched1(10) == 0.5

    sched2 = schedules.exponential_decay(1.0, 10, 0.5, staircase=True, end_value=0.1)
    assert sched2(5) == 1.0
    assert sched2(15) == 0.5


def test_inject_hyperparams():
    """Test inject hyperparams."""

    def dummy_factory():
        return "dummy"

    f1 = schedules.inject_hyperparams(dummy_factory)
    assert f1() == "dummy"

    f2 = schedules.inject_stateful_hyperparams(dummy_factory)
    assert f2() == "dummy"


def test_join_schedules():
    """Test join schedules."""
    scheds = [
        schedules.constant_schedule(1.0),
        schedules.constant_schedule(2.0),
        schedules.constant_schedule(3.0),
    ]
    boundaries = [10, 20]

    joined = schedules.join_schedules(scheds, boundaries)
    assert joined(5) == 1.0
    assert joined(15) == 2.0
    assert joined(25) == 3.0


def test_linear_schedules():
    """Test linear schedules."""
    sched1 = schedules.linear_schedule(1.0, 0.0, 10)
    assert sched1(0) == 1.0
    assert sched1(5) == 0.5
    assert sched1(10) == 0.0

    sched2 = schedules.linear_onecycle_schedule(10, 1.0)
    assert sched2(0) < 1.0
    assert sched2(3) == 1.0


def test_piecewise_schedules():
    """Test piecewise schedules."""
    bounds = {10: 0.5, 20: 0.1}
    sched1 = schedules.piecewise_constant_schedule(1.0, bounds)
    assert sched1(0) == 1.0
    assert sched1(15) == 0.5
    assert sched1(25) == 0.1

    sched2 = schedules.piecewise_constant_schedule(1.0)
    assert sched2(10) == 1.0

    sched3 = schedules.piecewise_interpolate_schedule("linear", 1.0, bounds)
    assert sched3(10) == 1.0

    sched4 = schedules.piecewise_interpolate_schedule("linear", 1.0)
    assert sched4(10) == 1.0


def test_polynomial_schedule():
    """Test polynomial schedule."""
    sched = schedules.polynomial_schedule(1.0, 0.0, 2.0, 10)
    assert sched(0) == 1.0
    assert sched(10) == 0.0


def test_sgdr_schedule():
    """Test SGDR schedule."""
    sched = schedules.sgdr_schedule()
    assert sched(0) == 0.1

    sched2 = schedules.sgdr_schedule([{"init_value": 0.1, "decay_steps": 10}])
    assert sched2(0) == 0.1


def test_warmup_schedules():
    """Test warmup schedules."""
    sched1 = schedules.warmup_constant_schedule(0.0, 1.0, 10)
    assert sched1(0) == 0.0
    assert sched1(10) == 1.0
    assert sched1(20) == 1.0

    sched2 = schedules.warmup_cosine_decay_schedule(0.0, 1.0, 10, 10)
    assert sched2(0) == 0.0
    assert sched2(10) == 1.0
    assert sched2(20) == 0.0

    sched3 = schedules.warmup_exponential_decay_schedule(0.0, 1.0, 10, 10, 0.5)
    assert sched3(0) == 0.0
    assert sched3(10) == 1.0
    assert sched3(20) == 0.5

    sched4 = schedules.warmup_exponential_decay_schedule(
        0.0, 1.0, 10, 10, 0.5, staircase=True, end_value=0.1
    )
    assert sched4(0) == 0.0
    assert sched4(15) == 1.0
    assert sched4(20) == 0.5


def test_schedule_edge_cases():
    """Test edge cases."""
    # 3rd phase of linear_onecycle_schedule
    sched1 = schedules.linear_onecycle_schedule(10, 1.0, pct_start=0.3, pct_final=0.8)
    assert sched1(9) < 1.0

    # p < 0 in warmup_exponential_decay_schedule
    sched2 = schedules.warmup_exponential_decay_schedule(
        0.0, 1.0, 2, 5, 0.5, transition_begin=5
    )
    assert sched2(4) == 1.0


def test_sgdr_and_interpolate_schedules():
    """Test sgdr and piecewise interpolate."""
    # SGDR
    sched = schedules.sgdr_schedule(
        [
            {"init_value": 0.1, "decay_steps": 10},
            {"init_value": 0.05, "decay_steps": 20},
        ]
    )
    assert sched(0) == 0.1
    assert sched(15) < 0.05
    assert sched(50) < 0.1

    # Piecewise interpolate
    bounds = {10: 0.5, 20: 0.1}
    sched2 = schedules.piecewise_interpolate_schedule("linear", 1.0, bounds)
    assert sched2(5) == 1.0  # before first bound
    assert sched2(15) == 0.3  # exact halfway between 0.5 and 0.1
    assert sched2(25) == 0.1  # after last bound

    import pytest

    with pytest.raises(ValueError):
        schedules.piecewise_interpolate_schedule("cubic", 1.0, bounds)(15)


def test_inject_hyperparams_exec():
    """Test hyperparams exec."""

    def factory(x):
        return x * 2

    f1 = schedules.inject_hyperparams(factory)
    assert f1(5) == 10

    f2 = schedules.inject_stateful_hyperparams(factory)
    assert f2(6) == 12

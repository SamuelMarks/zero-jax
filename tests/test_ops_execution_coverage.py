import os
import sys

import pytest

import zero_jax._compiler_proxy_ops as ops


def test_all_ops_executed(request):
    # This test will run last or we can use a fixture that collects them.
    # Actually, we can just write a script that runs pytest and collects them.
    pass

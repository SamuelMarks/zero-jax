"""Tests for zero_grain (pygrain) parity."""

import numpy as np
from zero_grain import MapDataset, IterDataset


def test_pygrain_mapdataset():
    data = [1, 2, 3]
    ds = MapDataset(data)
    assert len(ds) == 3
    assert ds[0] == 1
    assert ds[2] == 3


def test_pygrain_iterdataset():
    data = [1, 2, 3]
    ds = IterDataset(data)
    res = list(ds)
    assert res == [1, 2, 3]


def test_pygrain_filteroperation():
    data = [1, 2, 3, 4]
    ds = IterDataset(data).filter(lambda x: x % 2 == 0)
    res = list(ds)
    assert res == [2, 4]


def test_pygrain_batchoperation():
    data = [np.array(1), np.array(2), np.array(3), np.array(4), np.array(5)]
    ds = IterDataset(data).batch(2, drop_remainder=False)
    res = list(ds)
    assert len(res) == 3
    assert res[0].tolist() == [1, 2]
    assert res[1].tolist() == [3, 4]
    assert res[2].tolist() == [5]

    ds2 = IterDataset(data).batch(2, drop_remainder=True)
    res2 = list(ds2)
    assert len(res2) == 2


def test_pygrain_batchoperation_non_numpy():
    data = ["a", "b", "c"]
    ds = IterDataset(data).batch(2, drop_remainder=False)
    res = list(ds)
    assert res[0].tolist() == ["a", "b"]
    assert res[1].tolist() == ["c"]


def test_pygrain_map():
    data = [1, 2, 3]
    ds = IterDataset(data).map(lambda x: x * 2)
    res = list(ds)
    assert res == [2, 4, 6]


def test_pygrain_mapdataset_to_iter():
    data = [1, 2, 3]
    ds = MapDataset(data).to_iter_dataset()
    res = list(ds)
    assert res == [1, 2, 3]


def test_pygrain_batchoperation_ragged():
    data = [[1, 2], [3], [4, 5], 6]
    ds = IterDataset(data).batch(2, drop_remainder=False)
    res = list(ds)
    assert res[0] == [[1, 2], [3]]
    assert res[1] == [[4, 5], 6]

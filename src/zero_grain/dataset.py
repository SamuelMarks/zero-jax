"""Pure Python fallback for pygrain datasets."""

from typing import Any, Callable, Iterator, Iterable, List


class FilterOperation:
    def __init__(self, condition_fn: Callable):
        self.condition_fn = condition_fn

    def __call__(self, iterator: Iterator) -> Iterator:
        for item in iterator:
            if self.condition_fn(item):
                yield item


class BatchOperation:
    def __init__(self, batch_size: int, drop_remainder: bool = False):
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    def __call__(self, iterator: Iterator) -> Iterator:
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == self.batch_size:
                # Mock numpy stacking
                import numpy as np

                try:
                    yield np.stack(batch)
                except Exception:  # pragma: no cover
                    yield batch
                batch = []
        if batch and not self.drop_remainder:
            import numpy as np

            try:
                yield np.stack(batch)
            except Exception:  # pragma: no cover
                yield batch


class IterDataset:
    def __init__(self, source: Iterable):
        self.source = source
        self.operations = []

    def map(self, map_fn: Callable) -> "IterDataset":
        class MapOp:
            def __call__(self, iterator):
                for item in iterator:
                    yield map_fn(item)

        self.operations.append(MapOp())
        return self

    def filter(self, condition_fn: Callable) -> "IterDataset":
        self.operations.append(FilterOperation(condition_fn))
        return self

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "IterDataset":
        self.operations.append(BatchOperation(batch_size, drop_remainder))
        return self

    def __iter__(self) -> Iterator:
        iterator = iter(self.source)
        for op in self.operations:
            iterator = op(iterator)
        return iterator


class MapDataset:
    def __init__(self, elements: List[Any]):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        return self.elements[idx]

    def to_iter_dataset(self) -> IterDataset:
        return IterDataset(self.elements)

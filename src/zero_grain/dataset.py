"""Pure Python fallback for pygrain datasets."""

from typing import Any, Callable, Iterator, Iterable, List


class FilterOperation:
    """Docstring."""

    def __init__(self, condition_fn: Callable):
        """Docstring."""
        self.condition_fn = condition_fn

    def __call__(self, iterator: Iterator) -> Iterator:
        """Docstring."""
        for item in iterator:
            if self.condition_fn(item):
                yield item


class BatchOperation:
    """Docstring."""

    def __init__(self, batch_size: int, drop_remainder: bool = False):
        """Docstring."""
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    def __call__(self, iterator: Iterator) -> Iterator:
        """Docstring."""
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
    """Docstring."""

    def __init__(self, source: Iterable):
        """Docstring."""
        self.source = source
        self.operations = []

    def map(self, map_fn: Callable) -> "IterDataset":
        """Docstring."""

        class MapOp:
            """Docstring."""

            def __call__(self, iterator):
                """Docstring."""
                for item in iterator:
                    yield map_fn(item)

        self.operations.append(MapOp())
        return self

    def filter(self, condition_fn: Callable) -> "IterDataset":
        """Docstring."""
        self.operations.append(FilterOperation(condition_fn))
        return self

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "IterDataset":
        """Docstring."""
        self.operations.append(BatchOperation(batch_size, drop_remainder))
        return self

    def __iter__(self) -> Iterator:
        """Docstring."""
        iterator = iter(self.source)
        for op in self.operations:
            iterator = op(iterator)
        return iterator


class MapDataset:
    """Docstring."""

    def __init__(self, elements: List[Any]):
        """Docstring."""
        self.elements = elements

    def __len__(self):
        """Docstring."""
        return len(self.elements)

    def __getitem__(self, idx):
        """Docstring."""
        return self.elements[idx]

    def to_iter_dataset(self) -> IterDataset:
        """Docstring."""
        return IterDataset(self.elements)

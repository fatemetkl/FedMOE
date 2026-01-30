from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self, transform: Callable | None, target_transform: Callable | None) -> None:
        """
        Abstract base class for datasets used in this library. This class inherits from the torch Dataset base class.

        Args:
            transform (Callable | None, optional): Optional transformation to be applied to the input data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
            target_transform (Callable | None, optional): Optional transformation to be applied to the target data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
        """
        self.transform = transform
        self.target_transform = target_transform

    def update_transform(self, f: Callable) -> None:
        if self.transform:
            original_transform = self.transform
            self.transform = lambda *x: f(original_transform(*x))
        else:
            self.transform = f

    def update_target_transform(self, g: Callable) -> None:
        if self.target_transform:
            original_target_transform = self.target_transform
            self.target_transform = lambda *x: g(original_target_transform(*x))
        else:
            self.target_transform = g

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to be implemented by any inheriting dataset to produce a data value at provided index from
        the underlying data.

        Args:
            index (int): Index at which to extract the data from the dataset.

        Raises:
            NotImplementedError: Throws if one attempts to use this function.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Input and target tensors extracted at the provided index.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to be implemented by any inheriting dataset to produce a length value for the underlying data.

        Raises:
            NotImplementedError: Throws if one attempts to use this function.

        Returns:
            (int): Length of the underlying data.
        """
        raise NotImplementedError

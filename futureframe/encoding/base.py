import os
from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd


class BaseFeatureEncoder(ABC):
    def __init__(
        self, root: str, download: bool = True, overwrite: bool = False, name: str | None = None, *args, **kwargs
    ) -> None:
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.root = root

        root = os.path.join(root, name)
        if download:
            os.makedirs(root, exist_ok=True)
            self._download(root, overwrite=overwrite)

    def prepare(self, data: pd.DataFrame) -> "Self":
        """
        Prepate the encoder to the data with the training set.

        Args:
            data (pd.DataFrame): The data to prepare the encoder on.

        Returns:
            self: The prepared encoder object.
        """
        return self

    @abstractmethod
    def encode(self, data: pd.DataFrame) -> Any:
        """
        Encode the data into model inputs.

        This method should be implemented by subclasses to perform the actual encoding.

        Args:
            data (pd.DataFrame): The data to encode.

        Returns:
            dict[str, Tensor]: A dictionary of encoded model inputs.
        """
        pass

    def __call__(self, data) -> Any:
        return self.encode(data)

    def _download(self, root: str, overwrite: bool = False):
        """
        Download the encoder to the specified path.

        Args:
            root (str): The path to download the encoder to.
        """
        pass

    def load(self, path: str):
        """
        Load the encoder from the specified path.

        Args:
            path (str): The path to load the encoder from.
        """
        pass

    def save(self, path: str):
        """
        Save the encoder to the specified path.

        Args:
            path (str): The path to save the encoder to.
        """
        pass

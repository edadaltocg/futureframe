from typing_extensions import Self

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler

from futureframe.encoding.base import BaseFeatureEncoder
from futureframe.logger import log


def get_transformer(preprocess_transform: str):
    if preprocess_transform == "power":
        pt = PowerTransformer(standardize=True)
    elif preprocess_transform == "quantile":
        pt = QuantileTransformer(output_distribution="normal")
    elif preprocess_transform == "robust":
        pt = RobustScaler(unit_variance=True)
    elif preprocess_transform == "minmax":
        pt = MinMaxScaler()
    else:
        log.error(f"Preprocess transform {preprocess_transform} not found.")
        raise ValueError(f"Preprocess transform {preprocess_transform} not found.")

    return pt


class NumericFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        root: str,
        transformation,
        download: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, download, overwrite, name, *args, **kwargs)
        self.transformation = transformation

    def prepare(self, data: pd.DataFrame) -> "Self":
        data = data.select_dtypes(exclude=["object"])
        self.transformation.fit(data)
        return self

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.select_dtypes(exclude=["object"])
        data_ = self.transformation.transform(data)
        res = pd.DataFrame(data_, columns=data.columns, index=data.index)
        return res

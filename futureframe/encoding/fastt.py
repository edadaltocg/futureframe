import shutil
import os

from encoding.base import BaseFeatureEncoder
from futureframe import config
import numpy as np
import pandas as pd
import fasttext
import fasttext.util


class FastTextFeatureEncoder(BaseFeatureEncoder):
    model_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
    model_name = "cc.en.300.bin"
    dim = 300

    def __init__(self, root: str, download: bool = True, overwrite: bool = False, name: str | None = None) -> None:
        super().__init__(root, download, overwrite, name)

        self.lm_model = fasttext.load_model(os.path.join(root, self.model_name))
        self.cache = {}

    def _preprocess(self, data: pd.DataFrame):
        obj_df = data.select_dtypes(include=["object"])
        obj_cols = obj_df.columns
        obj_df = pd.melt(obj_df)["value"]
        obj_vals = obj_df.dropna().astype(str).unique()
        all_objs = np.concatenate([obj_cols, obj_vals])
        return all_objs

    def _str2vec(self, s: str):
        s_ = s.lower().replace("\n", " ")
        return self.lm_model.get_sentence_vector(s_)

    def prepare(self, data: pd.DataFrame):
        all_objs = self._preprocess(data)

        for s in all_objs:
            self.cache[s] = self._str2vec(s)

        return self

    def encode(self, data: pd.DataFrame):
        all_objs = self._preprocess(data)

        encoded = np.empty((all_objs.shape[0], self.dim), np.float32)
        for i, s in enumerate(all_objs):
            if s in self.cache:
                encoded[i] = self.cache[s]
            else:
                encoded[i] = self._str2vec(s)

        # Replace encoded in df
        return dict(objs=all_objs, encoded=encoded, data=data)

    def _download(self, root: str, overwrite: bool):
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(os.path.join(root, self.name, self.model_name)):
            fasttext.util.download_model("en")
            shutil.move(self.model_name, root)
            os.remove(self.model_name + ".gz")

    def save(self, path: str):
        dest = os.path.join(path, self.name, self.model_name, "cache.npz")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        np.savez(dest, **self.cache)

    def load(self, path: str):
        src = os.path.join(path, self.name, self.model_name, "cache.npz")
        self.cache = np.load(src)


def test_fasttext_feature_encoder(root=config.CACHE_ROOT, download=False):
    import time

    encoder = FastTextFeatureEncoder(root=root, download=download)
    data = pd.DataFrame({"a": ["hello", "world", "hello"], "b": ["world", "hello", "world"]})
    t0 = time.perf_counter()
    encoder.prepare(data)
    t1 = time.perf_counter()
    encoded = encoder.encode(data)
    str_encoded = encoded["encoded"]
    t2 = time.perf_counter()
    print(f"Preparation time: {t1 - t0:.2f}s")
    print(f"Encoding time: {t2 - t1:.2f}s")
    print(encoded)
    assert str_encoded.shape == (4, 300)
    encoder.save(root)
    cache = encoder.cache
    encoder.cache = {}
    encoder.load(root)
    assert all([np.allclose(cache[k], encoder.cache[k]) for k in cache])


if __name__ == "__main__":
    from fire import Fire

    Fire(test_fasttext_feature_encoder)
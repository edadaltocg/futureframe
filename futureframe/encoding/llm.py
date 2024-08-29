import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from encoding.base import BaseFeatureEncoder
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing_extensions import Self

from futureframe import config

log = logging.getLogger(__name__)


hf_model_id2dim = {
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "Alibaba-NLP/gte-base-en-v1.5": 768,
    "Alibaba-NLP/gte-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large-instruct": 1024,
}


class LLMFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        root: str,
        hf_model_id: str = "BAAI/bge-base-en-v1.5",
        device="cpu",
        batch_size=32,
        download: bool = True,
        overwrite: bool = False,
        dim: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(root, download, overwrite, name)
        self.hf_model_id = hf_model_id

        local_files_only = not download
        self.llm = SentenceTransformer(hf_model_id, cache_folder=root, device=device, local_files_only=local_files_only)
        self.llm.eval()

        self.device = device
        self.batch_size = batch_size

        if dim is None:
            dim = hf_model_id2dim[hf_model_id]
        self.dim = dim

        self.cache = {}
        # shelve.open(os.path.join(root, hf_model_id, "cache_dict"), writeback=True)
        # CacheDict(memory_limit=1, root=os.path.join(self.root, self.hf_model_id))

    def _preprocess(self, data: pd.DataFrame) -> list[str]:
        obj_cols = data.columns
        obj_df = pd.melt(data.select_dtypes(include=["object"]))["value"]
        obj_vals = obj_df.dropna().astype(str).unique()
        all_objs = np.concatenate([obj_cols, obj_vals]).tolist()
        return all_objs

    def _forward(self, batch, show_progress_bar=True):
        return self.llm.encode(
            batch,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

    def prepare(self, data: pd.DataFrame) -> "Self":
        all_objs = self._preprocess(data)
        try:
            self.load()
        except FileNotFoundError:
            pass
        new_objs = list(set(all_objs) - set(self.cache.keys()))

        encoded = self._forward(new_objs)
        # Setup cache
        for i, s in enumerate(tqdm(new_objs, desc="Caching embeddings", total=len(encoded))):
            self.cache[s] = encoded[i]
        self.save()

        return self

    def encode(self, data: pd.DataFrame) -> dict[str, Any]:
        all_objs = self._preprocess(data)

        not_on_cache = []
        encoded = torch.empty((len(all_objs), self.dim), dtype=torch.float32)
        for i, s in enumerate(all_objs):
            if s in self.cache:
                encoded[i] = self.cache[s]
            else:
                not_on_cache.append((i, s))

        if len(not_on_cache) > 0:
            batch = [s for _, s in not_on_cache]
            idxs = [idx for idx, _ in not_on_cache]
            encoded_batch = self._forward(batch, False)
            for j in range(len(idxs)):
                encoded[idxs[j]] = encoded_batch[j]
                # fill cache
                self.cache[batch[j]] = encoded_batch[j]

        return dict(objs=all_objs, embeddings=encoded)

    @property
    def dest(self):
        return os.path.join(self.root, self.hf_model_id)

    def save(self, path: str | None = None):
        if path is None:
            path = self.dest
            os.makedirs(path, exist_ok=True)
        torch.save(self.cache, os.path.join(path, "cache.pt"))

    def load(self, path: str | None = None):
        if path is None:
            path = self.dest
        self.cache = torch.load(os.path.join(path, "cache.pt"))


def test_llm_feature_encoder(root=config.CACHE_ROOT, download=False):
    import time

    encoder = LLMFeatureEncoder(root=root, download=download)
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
    assert str_encoded.shape == (4, encoder.dim)
    encoder.save(root)
    cache = encoder.cache
    encoder.cache = {}
    encoder.load(root)
    assert all([np.allclose(cache[k], encoder.cache[k]) for k in cache])


if __name__ == "__main__":
    from fire import Fire

    Fire(test_llm_feature_encoder)

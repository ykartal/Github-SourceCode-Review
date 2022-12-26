from __future__ import annotations
from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from .base import BaseVectorizer
from tqdm import tqdm
from pathlib import Path


class WordDocVectorizer(BaseVectorizer):
    model = None
    take_tokenized_data = True
    trainable = True

    def save(self, path):
        pth = Path(path, f"{self.name}.model")
        pth.parent.mkdir(parents=True, exist_ok=True)
        return self.model.save(pth)

    @classmethod
    def save_vectors(cls, x, path):
        """Save the vectors to a file with .npy format"""
        return np.save(path, x)

    @classmethod
    def load_vectors(cls, path) -> np.ndarray:
        """Load the vectors from a file with .npy format"""
        return np.load(path)


class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        print(f'Epoch {self.epoch} completed.')
        self.epoch += 1


class Word2VecMeanVectorizer(WordDocVectorizer):
    name = "word2vec-mean"

    def __init__(self, path: str = None):
        self.model = Word2Vec.load(path) if path is not None else None

    def fit(self, x, **kwargs):
        if self.model is None:
            self.model = Word2Vec(x, callbacks=[callback()], **kwargs)
            self.dim = len(self.model[next(iter(self.model))])
        return self

    def transform(self, x):
        return np.array([
            np.mean([self.model.wv[w] for w in words if w in self.model.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in tqdm(x)
        ])


class Doc2VecVectorizer(WordDocVectorizer):
    name = "doc2vec"

    def __init__(self, path: str = None):
        self.model = Doc2Vec.load(path) if path is not None else None

    def fit(self, x, **kwargs):
        if self.model is None:
            tagged_data = [TaggedDocument(
                words=tokens, tags=[str(i)]) for i, tokens in enumerate(x)]
            self.model = Doc2Vec(tagged_data,  callbacks=[
                                 callback()], **kwargs)
        return self

    def transform(self, x):
        return np.array([self.model.infer_vector(words) for words in tqdm(x)])


class FastTextMeanVectorizer(WordDocVectorizer):
    name = "fasttext-mean"

    def __init__(self, path: str = None):
        self.model = FastText.load(path) if path is not None else None

    def fit(self, x, **kwargs):
        if self.model is None:
            self.model = FastText(x, callbacks=[callback()], **kwargs)
        return self

    def transform(self, x):
        return np.array([
            np.mean([self.model.wv[w] for w in words if w in self.model.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in tqdm(x)])

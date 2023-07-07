from __future__ import annotations
from .base import BaseVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as sk_TfidfVectorizer, CountVectorizer as sk_CountVectorizer
import pickle
from scipy import sparse
from pathlib import Path

class TraditionalVectorizer(BaseVectorizer):
    vectorizer = None
    take_tokenized_data = False
    trainable = True

    def fit(self, x):
        self.vectorizer.fit(x)

    def transform(self, x):
        return self.vectorizer.transform(x)

    def save(self, path):
        pth = Path(path, f"{self.name}.pkl")
        pth.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.vectorizer, open(pth, "wb"))

    @classmethod
    def save_vectors(cls, x, path):
        """Save the vectors to a file with .npz format"""
        return sparse.save_npz(path, x)

    @classmethod
    def load_vectors(cls, path) -> sparse.csr_matrix:
        """Load the vectors from a file with .npz format"""
        return sparse.load_npz(path)

class BowVectorizer(TraditionalVectorizer):
    name = "bow"

    def __init__(self, path = None, **train_args):
        self.vectorizer = pickle.load(open(path, 'rb')) if path else sk_CountVectorizer(**train_args)


class TfIdfVectorizer(TraditionalVectorizer):
    name = "tfidf"

    def __init__(self, path = None, **train_args):
        self.vectorizer = pickle.load(open(path, 'rb')) if path else sk_TfidfVectorizer(**train_args)

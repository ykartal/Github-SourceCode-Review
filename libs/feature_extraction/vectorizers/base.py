from abc import ABC, abstractmethod

class BaseVectorizer(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def take_tokenized_data(self) -> bool:
        pass

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @classmethod
    @abstractmethod
    def load_vectors(cls, path):
        pass
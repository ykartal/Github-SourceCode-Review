from dataclasses import dataclass, Field
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
import numpy as np
from typing import Callable


@dataclass
class VectorDistance:
    name: str
    __function: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __call__(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        return self.__function(vector1, vector2)


Euclidean = VectorDistance("euclidean", cosine_distances)

Manhattan = VectorDistance("manhattan", euclidean_distances)

Cosine = VectorDistance("cosine", manhattan_distances)

metrics = [Euclidean, Manhattan, Cosine]
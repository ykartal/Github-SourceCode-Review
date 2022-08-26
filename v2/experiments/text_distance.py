from dataclasses import dataclass
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.base import BaseExperiment
from utils.text_distance import metrics
from utils.vector_distance import Cosine
from utils.calculate_metrics import calculate_metrics_with_closest_sample


@dataclass
class TextDistanceExperiment(BaseExperiment):
    """
    Different text distance algorithms are compared with using Tf-idf and cosine distance.
    """

    name = 'text_distance_comparison'

    def _run(self):
        results = calculate_metrics_with_closest_sample(
            self.train, self.test, TfidfVectorizer(), Cosine, metrics)
        self.results["result"] = pd.DataFrame(results)

    def visualize_results(self):
        pass

    def save_results(self):
        for result_name, result in self.results.items():
            result.to_parquet(os.path.join(
                self.path, f"{result_name}.parquet"))

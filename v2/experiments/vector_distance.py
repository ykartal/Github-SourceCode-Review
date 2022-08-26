from dataclasses import dataclass
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from experiments.base import BaseExperiment

from utils.text_distance import Bleu
from utils.vector_distance import metrics
from utils.calculate_metrics import calculate_metrics_with_closest_sample


@dataclass
class VectorDistanceExperiment(BaseExperiment):
    """
    Different vector distance algorithms are compared with using Tf-idf and Bleu score.
    """

    name = 'vector_distance_comparison'

    def _run(self):
        for vector_metric in metrics:
            results = calculate_metrics_with_closest_sample(
                self.train, self.test, TfidfVectorizer(), vector_metric, [Bleu])
            self.results[vector_metric.name] = pd.DataFrame(results)

    def visualize_results(self):
        pass

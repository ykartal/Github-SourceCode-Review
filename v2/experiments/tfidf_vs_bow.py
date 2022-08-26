import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from experiments.base import BaseExperiment
from utils.text_distance import Bleu
from utils.vector_distance import Cosine
from utils.calculate_metrics import calculate_metrics_with_closest_sample


class TfidfvsBowExperiment(BaseExperiment):
    """
    With using Cosine distance and Bleu score, compare the tfidf-vectorized and bag-of-words vectorized data with closest sample.
    """
    name = 'tfidf_vs_bow'

    def _run(self):
        selected_metrics = [Bleu]
        tf_idf = calculate_metrics_with_closest_sample(self.train, self.test, TfidfVectorizer(), Cosine,
                                                       selected_metrics)
        bow = calculate_metrics_with_closest_sample(
            self.train, self.test, CountVectorizer(), Cosine, selected_metrics)
        self.results["tf_idf"] = pd.DataFrame(tf_idf)
        self.results["bow"] = pd.DataFrame(bow)

    def visualize_results(self):
        pass

    def save_results(self):
        for result_name, result in self.results.items():
            result.to_parquet(os.path.join(
                self.path, f"{result_name}.parquet"))

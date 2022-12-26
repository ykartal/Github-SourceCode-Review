from dataclasses import dataclass
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from experiments.base import BaseExperiment
from helpers.text_similarity import Bleu
from helpers.vector_distance import Cosine
from helpers.calculate_metrics import calculate_metrics_with_closest_sample


@dataclass
class TfidfvsBowExperiment(BaseExperiment):
    """
    With using Cosine distance and Bleu score, compare the tfidf-vectorized and bag-of-words vectorized data with closest sample.
    """
    name = 'tfidf_vs_bow'

    def get_exact_matches(self):
        exact_matches = []
        for name, df in self.results.items():
            exact_match_count = df[df["test_comment"]
                                   == df["recommended_comment"]].shape[0]
            ratio = (exact_match_count / df.shape[0]) * 100
            exact_matches.append(
                {"name": name, "exact_match_count": exact_match_count, "ratio": round(ratio, 3)})
        return exact_matches

    def _run(self):
        selected_metrics = [Bleu]
        vectorizers = [CountVectorizer(), TfidfVectorizer()]
        for vectorizer in vectorizers:
            vectorizer_name = vectorizer.__class__.__name__.lower().replace("vectorizer", "")
            print(f"Running {vectorizer_name} vectorizer...")
            self.results[vectorizer_name] = pd.DataFrame(calculate_metrics_with_closest_sample(self.train, self.test,
                vectorizer, Cosine, selected_metrics))

    def calculate_range_means(self):
        norm_results = self._normalize_results_column("distance")
        means = []
        calc_col = "distance"
        metric_col = "metric:bleu"
        for i in range(10):
            metric_info = {}
            metric_info[calc_col] = f"{i/10} - {(i+1)/10}"
            for name, result in norm_results.items():
                mean = result[(result[calc_col] >= i/10) & (
                    result[calc_col] <= (i+1)/10)][metric_col].mean()
                metric_info[name] = round(mean, 3)
            means.append(metric_info)
        return means

    def visualize_results(self, title: str = "Cosine Distance/Bleu Score with Different Vectorization Methods"):
        bleu_means = self.calculate_range_means()
        plt.style.use("seaborn")
        ax = pd.DataFrame(bleu_means).plot(
            x="distance", kind="line", figsize=(8, 5), fontsize=12, linewidth=3)
        plt.tight_layout()
        ax.legend([name.upper() for name, _ in self.results.items()],
                  fontsize=12, frameon=True)
        ax.set_title(title, fontsize=15)
        ax.invert_xaxis()
        plt.ylabel("Blue Score Mean", fontsize=13)
        plt.xlabel("Normalized Distance Range", fontsize=13)
        plt.savefig(os.path.join(self.path, "bow_vs_tfidf.png"),
                    facecolor="w", edgecolor="w", bbox_inches="tight")

    def generate_report_stuff(self):
        range_means = self.calculate_range_means()
        pd.DataFrame(range_means).to_csv(
            os.path.join(self.path, "bleu_means.csv"), index=False)
        pd.DataFrame(self.get_exact_matches()).to_csv(
            os.path.join(self.path, "exact_matches.csv"), index=False)

    def get_manuel_labeled_data(self):
        results = self._normalize_results_column("distance")
        df = results["tf_idf"]
        distance_df = df[(df["distance"] < 0.4) & (df["distance"] > 0.3)].sort_values(by=["distance"], ascending=True)
        filtered_df = distance_df[distance_df["metric:bleu"] != 0].sort_values(by=["metric:bleu"], ascending=True)
        filtered_df[:300].to_csv("tfidf_vs_bow_filtered.csv")


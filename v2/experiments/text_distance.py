from dataclasses import dataclass
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.base import BaseExperiment
from helpers.text_similarity import metrics
from helpers.vector_distance import Cosine
from helpers.calculate_metrics import calculate_metrics_with_closest_sample


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

    def calculate_range_means(self):
        results = self._normalize_results_column("distance")["result"]
        calc_col = "distance"
        means = []
        for i in range(10):
            metrics = results.filter(like="metric:", axis=1).columns.to_list()
            metric_info = {}
            metric_info["distance"] = f"{i/10} - {(i+1)/10}"
            for metric in metrics:
                mean = results[(results[calc_col] >= i/10) & (results[calc_col] <= (i+1)/10)][metric].mean()
                metric_info[metric.replace("metric:", "")] = round(mean, 3)
            means.append(metric_info)
        return means

    def filter_mean_metrics(self, topK: int):
        means = self.calculate_range_means()
        results = self.results["result"]
        selected_cols = results.corr()["distance"].sort_values(ascending=True)[:topK].keys()
        selected_cols = [col.replace("metric:", "") for col in selected_cols]
        filtered_means = []
        for mean in means:
            mean_info = { "distance": mean["distance"] }
            x = mean_info | {key: mean[key] for key in selected_cols}
            filtered_means.append(x)
        return filtered_means

    def visualize_results(self, title:str = "Distance/Similarity Score with Different Text Similarity Algorithms"):
        means = self.filter_mean_metrics(8)
        keys = list(means[0].keys())[1:]
        plt.style.use("seaborn")
        ax = pd.DataFrame(means).plot(
            x="distance", kind="line", figsize=(8, 5), fontsize=12, linewidth=3,marker="o")
        plt.tight_layout()
        ax.legend([name.upper() for name in keys], fontsize=12, frameon=True, bbox_to_anchor=(1, 1))
        ax.set_title(title, fontsize=15)
        plt.ylabel("Similarity Score Mean", fontsize=13)
        plt.xlabel("Normalized Distance Range", fontsize=13)
        plt.savefig(os.path.join(self.path, "text_distances.png"), facecolor="w", edgecolor="w", bbox_inches="tight")

    def generate_report_stuff(self):
        range_means = self.calculate_range_means()
        filtered_means = self.filter_mean_metrics(8)
        corr = self._normalize_results_column("distance")["result"].corr()
        pd.DataFrame(range_means).to_csv(os.path.join(self.path, "range_means.csv"), index=False)
        pd.DataFrame(filtered_means).to_csv(os.path.join(self.path, "filtered_means.csv"), index=False)
        corr.to_csv(os.path.join(self.path, "correlation.csv"), index=False)
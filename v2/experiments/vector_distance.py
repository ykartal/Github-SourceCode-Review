from dataclasses import dataclass
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from experiments.base import BaseExperiment

from helpers.text_similarity import Bleu
from helpers.vector_distance import metrics
from helpers.calculate_metrics import calculate_metrics_with_closest_sample


@dataclass
class VectorDistanceExperiment(BaseExperiment):
    """
    Different vector distance algorithms are compared with using Tf-idf and Bleu score.
    """

    name = 'vector_distance_comparison'



    def _run(self):
        for vector_metric in metrics:
            print(f"Running for {vector_metric.name}...")
            results = calculate_metrics_with_closest_sample(
                self.train, self.test, TfidfVectorizer(), vector_metric, [Bleu])
            self.results[vector_metric.name] = pd.DataFrame(results)



    def visualize_results(self, title: str = "Distance/Bleu Score with Different Vector Distance Algorithms"):
        means = self.calculate_range_means()
        plt.style.use("seaborn")
        ax = pd.DataFrame(means).plot(
            x="distance", kind="line", figsize=(8, 5), fontsize=12, linewidth=3)
        plt.tight_layout()
        ax.legend([name.upper() for name, _ in self.results.items()],
                  fontsize=12, frameon=True)
        ax.set_title(title, fontsize=15)
        ax.invert_xaxis()
        plt.ylabel("Blue Score Mean", fontsize=13)
        plt.xlabel("Normalized Distance Range", fontsize=13)
        plt.savefig(os.path.join(self.path, "vector_distances.png"),
                    facecolor="w", edgecolor="w", bbox_inches="tight")

    def generate_report_stuff(self):
        range_means = self.calculate_range_means()
        pd.DataFrame(range_means).to_csv(os.path.join(
            self.path, "range_means.csv"), index=False)
        pd.DataFrame(self.get_exact_matches()).to_csv(
            os.path.join(self.path, "exact_matches.csv"), index=False)

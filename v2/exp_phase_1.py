import uuid
from experiments.tfidf_vs_bow import TfidfvsBowExperiment
from experiments.vector_distance import VectorDistanceExperiment
from experiments.text_distance import TextDistanceExperiment
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from helpers.cross_validation import CrossValidator
from helpers.plotter import visualize_results

K_FOLD = 5
RANDOM_STATE = 1


def tf_idf_vs_bow_with_cv(datapath: str):
    """
    Tf-idf and Bag of Words are compared with using Cosine distance and Bleu score.
    """

    data = pd.read_csv(datapath)
    id = uuid.uuid4().__str__()
    validator = CrossValidator(id, TfidfvsBowExperiment, data)
    results = validator.validate(K_FOLD, True, RANDOM_STATE)
    validator.save_results(results)
    plot_means = pd.concat([pd.DataFrame(fold.calculate_range_means(
    )) for fold in results]).groupby(['distance']).sum().div(K_FOLD).reset_index()
    visualize_results(plot_means, "distance", "metric:bleu", "line", "Distance/Bleu Score with Different Vectorization Algorithms",
                      "Normalized Distance Range", "Bleu Score", [name.upper() for name, _ in results[0].results.items()], True, f"results/{TfidfvsBowExperiment.name}/{id}/plot.png")
    #pd.concat([pd.DataFrame(fold.get_exact_matches()) for fold in results]).groupby(['name']).sum().div(5).reset_index() Exact Matches

def vector_distances(datapath: str):
    """
    Different vector distance algorithms are compared with using Tf-idf and Bleu score.
    """

    data = pd.read_csv(datapath)
    id = uuid.uuid4().__str__()
    validator = CrossValidator(id, VectorDistanceExperiment, data)
    results = validator.validate(K_FOLD, True, RANDOM_STATE)
    validator.save_results(results)
    means = pd.concat([pd.DataFrame(fold.calculate_range_means(
    )) for fold in results]).groupby(['distance']).sum().div(K_FOLD).reset_index()
    visualize_results(means, "distance", None, "line", "Distance/Bleu Score with Different Vector Distance Algorithms",
                      "Normalized Distance Range", "Bleu Score Mean", [name.upper() for name, _ in results[0].results.items()], True, f"results/{VectorDistanceExperiment.name}/{id}/plot.png")


if __name__ == "__main__":
    tf_idf_vs_bow_with_cv("data/comment_finder_data.csv")

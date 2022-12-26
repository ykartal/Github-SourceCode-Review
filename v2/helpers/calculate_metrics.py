import math
import pandas as pd
import numpy as np
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer

from .text_similarity import TextSimilarity
from .vector_distance import VectorDistance

def calculate_metrics_with_closest_sample(train: pd.DataFrame,
                                          test: pd.DataFrame,
                                          vectorizer: CountVectorizer,
                                          distance_method: VectorDistance,
                                          selected_metrics: list[TextSimilarity],
                                          chunk_size: int = 1000) -> list[dict]:

    train_codes = train["code"].to_numpy()
    test_codes = test["code"].to_numpy()
    train_vectors = vectorizer.fit_transform(train_codes)
    results = []

    for idx in range(math.ceil(len(test_codes) / chunk_size)):
        if (idx + 1) * chunk_size > len(test_codes):
            end = len(test_codes)
        else:
            end = (idx + 1) * chunk_size

        idx_offset = idx * chunk_size
        test_chunk = test_codes[idx_offset : end]
        test_vectors = vectorizer.transform(test_chunk)
        calculations = distance_method(test_vectors, train_vectors)

        for i, calculation in enumerate(calculations):

            test_idx = idx_offset + i
            closest_sample_idx = np.argmin(calculation)
            test_code = test.iloc[test_idx]["code"]
            test_comment = test.iloc[test_idx]["comment"]
            train_code = train.iloc[closest_sample_idx]["code"]
            train_comment = train.iloc[closest_sample_idx]["comment"]

            results.append({
                "test_code": test_code,
                "test_comment": test_comment,
                "recommended_code": train_code,
                "recommended_comment": train_comment,
                "distance": calculation[closest_sample_idx]
            } | {f"metric:{metric.name}": metric(train_comment,  test_comment) for metric in selected_metrics})

            print(f"Finished: {test_idx+1}/{len(test_codes)}", end="\r")
    print("")
    return results


def calculate_metric_means_with_top_k_closest_samples(k: int,
                                                      train: pd.DataFrame,
                                                      test: pd.DataFrame,
                                                      vectorizer: type,
                                                      distance_method: VectorDistance,
                                                      text_similarity_method: TextSimilarity,
                                                      chunk_size: int = 1000) -> Union[list[dict], int]:
    vectorizer = vectorizer()
    train_codes = train["code"].to_numpy()
    test_codes = test["code"].to_numpy()
    train_vectors = vectorizer.fit_transform(train_codes)
    recommendation_indexes = []
    distances = []
    metric_scores = []

    for idx in range(math.ceil(len(test_codes) / chunk_size)):
        if (idx + 1) * chunk_size > len(test_codes):
            end = len(test_codes)
        else:
            end = (idx + 1) * chunk_size
        idx_offset = idx * chunk_size
        test_chunk = test_codes[idx_offset: end]
        test_vectors = vectorizer.transform(test_chunk)
        calculations = distance_method(test_vectors, train_vectors)

        for i, distance in enumerate(calculations):
            test_idx = idx_offset + i
            closest_sample_indexes = np.argsort(distance)[:k]
            test_comment = test.iloc[test_idx]["comment"]
            train_comments = train.iloc[closest_sample_indexes]["comment"]

            metric_results = {}
            for metric in selected_metrics:
                metric_values = []
                for train_comment in train_comments:
                    metric_values.append(metric(train_comment, test_comment))
                idx_max = np.argmax(metric_values)
                metric_results[f"metric:{metric.name}"] = metric_values[idx_max]
                metric_results[f"(distance,{metric.name})"] = (round(distance[closest_sample_indexes[idx_max]], 2), round(metric_values[idx_max], 2))


            results.append({
                "test_idx": test_idx,
                "recommendation_idx": closest_sample_indexes[idx_max]
            }
                | {name: mean for name, mean in metric_results.items()})

            print(f"Finished: {test_idx+1}/{len(test_codes)}", end="\r")
    print("")
    return results

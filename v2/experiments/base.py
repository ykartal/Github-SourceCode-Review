import os
import time
import traceback
import pandas as pd
import numpy as np
from dateutil import parser
from abc import ABC, abstractmethod
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import TypedDict, Union
import uuid
from helpers.calculate_metrics import calculate_metric_means_with_top_k_closest_samples
from helpers.text_similarity import TextDistance

from helpers.vector_distance import VectorDistance
from config import RESULT_PATH, RANDOM_STATE, CHUNK_SIZE
from sklearn.feature_extraction.text import CountVectorizer

class Result(TypedDict):
    """
    Result of an experiment.
    """
    name: str
    result: pd.DataFrame


@dataclass
class PhaseOneExperiment():
    name: str
    vectorizers: list[CountVectorizer]
    distance_methods: list[VectorDistance]
    text_similarity_methods: list[TextDistance]
    candidate_k: int
    info: list = field(init=False, default_factory=list)
    path: str = field(init=False, default=None)
    results: Result = field(init=False, default_factory=Result, repr=False)

    def _run(self):
        for vectorizer in self.vectorizers:
            vectorizer_name = vectorizer.__name__.lower().replace("vectorizer", "")
            for distance_method in self.distance_methods:
                results = calculate_metric_means_with_top_k_closest_samples(self.candidate_k,
                    self.train, self.test, vectorizer, distance_method, self.text_similarity_methods)
                self.results[f"{vectorizer_name}_{distance_method.name}"] = pd.DataFrame(results)


    def get_exact_matches(self):
        exact_matches = []
        for name, df in self.results.items():
            exact_match_count = df[df["test_comment"]
                                   == df["recommended_comment"]].shape[0]
            ratio = (exact_match_count / df.shape[0]) * 100
            exact_matches.append(
                {"name": name, "exact_match_count": exact_match_count, "ratio":  round(ratio, 3)})

    def calculate_range_means(self):
        calc_col = "distance"
        results = self._normalize_results_column(calc_col)["result"]
        means = []
        for i in range(10):
            metrics = results.filter(like="metric:", axis=1).columns.to_list()
            metric_info = {}
            metric_info["distance"] = f"{i/10} - {(i+1)/10}"
            for metric in metrics:
                mean = results[(results[calc_col] >= i/10) &
                               (results[calc_col] <= (i+1)/10)][metric].mean()
                metric_info[metric.replace("metric:", "")] = round(mean, 3)
            means.append(metric_info)
        return means

    def save_results(self, id=None, name=None):
        id = uuid.uuid4().__str__() if id is None else id
        self.path = os.path.join(RESULT_PATH, self.name, id, name)

        os.makedirs(self.path, exist_ok=True)
        for result_name, result in self.results.items():
            result.to_parquet(os.path.join(
                self.path, f"{result_name}.parquet"))
        pd.DataFrame(self.info).to_csv(
            os.path.join(self.path, "info.csv"), index=False)

        self.generate_report()

    def _normalize_results_column(self, column: str, inplace=False):
        new_results = self.results.copy() if not inplace else self.results
        for _, result in new_results.items():
            result[column] = self.normalize_data(result[column])
        return new_results

    def normalize_data(self, data):
        '''
        Normalize data to range [0, 1] with MaxMin
        '''
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def load_results(self, experiment_time: str):
        self.path = os.path.join(RESULT_PATH, self.name, experiment_time)
        for result_name in os.listdir(self.path):
            if ".parquet" in result_name:
                self.results[result_name.removesuffix(".parquet")] = pd.read_parquet(os.path.join(
                    self.path, result_name))

    def write_info(self, key, value):
        self.info.append({"key": key, "value": value})

    def run(self, train, test):
        self.train = train
        self.test = test
        try:
            self.write_info("Experiment", self.name.replace('_', ' ').title())
            self.write_info("Train size", len(self.train))
            self.write_info("Test size", len(self.test))
            experiment_start = time.strftime("%X")
            self.write_info("Experiment start", experiment_start)
            self._run()
            experiment_end = time.strftime("%X")
            self.write_info("Experiment end", experiment_end)
            execution_time = parser.parse(
                experiment_end) - parser.parse(experiment_start)
            self.write_info("Execution time", execution_time)
        except Exception as e:
            self.write_info("Error", traceback.format_exc())
            raise e

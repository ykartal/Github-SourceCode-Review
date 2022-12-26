from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import  Union
import pandas as pd
import time
import numpy as np
import math
from libs.metrics.vector_distance import VectorDistance
from scipy import sparse
from sklearn.model_selection import train_test_split
from libs.feature_extraction.vectorizers.base import BaseVectorizer
from libs.metrics.text_similarity import TextSimilarity
from libs.preprocessing.scaling import normalize_columns
from libs.metrics import text_similarities, vector_distances

from sklearn.model_selection import KFold
from tqdm import tqdm


@dataclass
class ExperimentParams:
    vectorizer: BaseVectorizer
    vector_distance: str
    text_similarity: str
    k_fold: int = 0
    train_test_split: float = 0.2
    candidate_count: int = 1
    random_state: int = 42
    chunk_size: int = 1000
    round_degree: int = 4
    load_embeddings: bool = False
    embeddings_path: str = ""
    vector_distance_method: VectorDistance = field(init=False)
    text_similarity_method: TextSimilarity = field(init=False)

    def __post_init__(self):
        self.vector_distance_method = vector_distances[self.vector_distance]
        self.text_similarity_method = text_similarities[self.text_similarity]


@dataclass
class ExperimentData:
    df: pd.DataFrame
    train_df: pd.DataFrame = field(init=False)
    test_df: pd.DataFrame = field(init=False)
    vectors: Union[sparse.csr_matrix, np.ndarray] = field(init=False)
    train_vectors: Union[sparse.csr_matrix, np.ndarray] = field(init=False)
    test_vectors: Union[sparse.csr_matrix, np.ndarray] = field(init=False)
    df_folds: dict = field(default_factory=dict, init=False)
    vector_folds: dict = field(default_factory=dict, init=False)

    def prepare(self, params: ExperimentParams):
        if params.load_embeddings:
            if params.embeddings_path == "":
                raise Exception("Embeddings path is not provided")
            self.vectors = params.vectorizer.load_vectors(
                params.embeddings_path)
        else:
            self.vectors = params.vectorizer.transform(
                self.df["code"])

        if params.k_fold != 0:
            kfold = KFold(n_splits=params.k_fold, shuffle=True,
                          random_state=params.random_state)
            for i, (train_index, test_index) in enumerate(kfold.split(self.df)):
                train_df, test_df = self.df.iloc[train_index], self.df.iloc[test_index]
                train_vectors, test_vectors = self.vectors[
                    train_index], self.vectors[test_index]
                self.df_folds[i] = (train_df, test_df)
                self.vector_folds[i] = (train_vectors, test_vectors)
        else:
            self.train_df, self.test_df = train_test_split(
                self.df, test_size=params.train_test_split, random_state=params.random_state)
            self.train_vectors, self.test_vectors = self.vectors[
                self.train_df.index], self.vectors[self.test_df.index]

        if self.vectors.shape[0] != self.df.shape[0]:
            raise Exception("Number of vectors is not equal to number of rows in dataframe")

@dataclass
class Experiment:
    """Only runs with trained vectorization models or already generated embeddings."""
    data: ExperimentData
    params: ExperimentParams
    results: Union[pd.DataFrame, list[pd.DataFrame]] = field(init=False)
    execution_time: float = field(init=False)

    @property
    def name(self):
        fold = f"_F{self.params.k_fold}" if self.params.k_fold != 0 else ""
        return f"{self.params.vectorizer.name}_{self.params.vector_distance}_{self.params.text_similarity}_C{self.params.candidate_count}{fold}"

    def __post_init__(self):
        self.data.prepare(self.params)

    def run(self):
        self.results, self.execution_time = self.__run() if self.params.k_fold == 0 else self.__run_k_fold()
        return self.results, self.execution_time

    def save_results(self, path: str):
        pth = Path(path, self.name)
        pth.mkdir(parents=True, exist_ok=True)

        if self.params.k_fold == 0:
            self.results.to_csv(Path(pth, "results.csv"), index=False)
        else:
            for i, result in enumerate(self.results):
                result.to_csv(Path(pth, f"fold_{i+1}.csv"), index=False)

    def __run(self):
        print("Running experiment: ", self.name)
        start = time.time()
        results = self.__run_experiment(
            self.data.train_df, self.data.test_df, self.data.train_vectors, self.data.test_vectors)
        return results, time.time() - start

    def __run_k_fold(self):
        results = []
        exec_times = []
        print("Running experiment: ", self.name)
        for i, (train_df, test_df) in self.data.df_folds.items():
            print("Running fold: ", i)
            start = time.time()
            train_vectors, test_vectors = self.data.vector_folds[i]
            result = self.__run_experiment(
                train_df, test_df, train_vectors, test_vectors)
            results.append(result)
            exec_times.append(time.time() - start)
        return results, np.mean(exec_times)

    def __run_experiment(self, train_df: pd.DataFrame, test_df: pd.DataFrame, train_vectors, test_vectors):
        test_vectors_length = test_vectors.shape[0]
        chunk_size = self.params.chunk_size
        candidate_count = self.params.candidate_count

        results = []
        for idx in tqdm(range(math.ceil(test_vectors_length / chunk_size))):
            idx_offset = idx * chunk_size
            calculations = self.params.vector_distance_method(test_vectors[idx_offset: min((idx + 1) * chunk_size, test_vectors_length)], train_vectors)

            for i, distance in enumerate(calculations):
                test_idx = idx_offset + i
                closest_sample_indexes = self.get_closest_sample_indexes(distance, candidate_count)
                test_comment = test_df.iloc[test_idx]["comment"]
                train_comments = train_df.iloc[closest_sample_indexes]["comment"]
                text_similarities, idx_max = self.get_similarities_and_argmax(self.params.text_similarity_method, test_comment, train_comments)

                results.append({
                    "test_idx": test_df.iloc[test_idx].name,
                    "recommended_idx": train_df.iloc[closest_sample_indexes[idx_max]].name,
                    "distance": round(distance[closest_sample_indexes[idx_max]], self.params.round_degree),
                    f"{self.params.text_similarity}": round(text_similarities[idx_max], self.params.round_degree),
                })

        return pd.DataFrame(results)

    @classmethod
    def get_closest_sample_indexes(cls, distances, candidate_count):
        return [np.argmin(distances)] if candidate_count == 1 else np.argpartition(distances, candidate_count)[:candidate_count]

    @classmethod
    def get_similarities_and_argmax(cls, method, test_comment, train_comments):
        similarities = [method(train_comment, test_comment) for train_comment in train_comments]
        return similarities, np.argmax(similarities)


@dataclass
class ExperimentEvaluation:
    experiment: Experiment
    normalize_results: bool = True
    results: dict = field(init=False)
    __distance_column: str = field(default="distance", init=False)

    @property
    def is_cv(self):
        return self.experiment.params.k_fold > 0

    @property
    def metric(self):
        return self.experiment.params.text_similarity

    def __post_init__(self):
        if self.normalize_results:
            if self.is_cv:
                self.result = [normalize_columns(
                    result, [self.__distance_column]) for result in self.experiment.results]
                return
            self.result = normalize_columns(
                self.experiment.results, [self.__distance_column])

    @classmethod
    def idx_to_text(cls, data_df, idx_df):
        text_df = pd.DataFrame()
        text_df[["test_code", "test_comment"]
                ] = data_df.loc[idx_df["test_idx"]].values
        text_df[["recommended_code", "recommended_comment"]
                ] = data_df.loc[idx_df["recommended_idx"]].values
        return text_df

    def evaluate(self):
        self.results = {
            "execution_time": self.experiment.execution_time,
            "mean": self.get_metric_mean(),
            "exact_matches": self.get_exact_matches(),
            "ranged_means" : self.get_ranged_means(),
        }
        return self.results

    def save_results(self, path: str):
        pth = Path(path, self.experiment.name)
        pth.mkdir(parents=True, exist_ok=True)
        results = {
            "execution_time": self.results["execution_time"],
            f"{self.metric}_mean": self.results["mean"],
            "exact_matches": self.results["exact_matches"],
            f"{self.metric}_ranged_means": self.results["ranged_means"].to_dict(),
        }
        with open(Path(pth, "evaluation.json"), "w") as f:
            json.dump(results, f, indent=4)

    @classmethod
    def load_results(cls, path: str):
        pth = Path(path)
        experiment_name = pth.name
        params = experiment_name.split("_")
        with open(Path(pth, "evaluation.json"), "r") as f:
            results = json.load(f)
            results[f"{params[2]}_ranged_means"] = pd.DataFrame.from_dict(results[f"{params[2]}_ranged_means"])
        return experiment_name, results

    def get_metric_mean(self):
        return self.__get_cv_means(self.result) if self.is_cv else self.__get_means(self.result)

    def get_ranged_means(self):
        return self.__get_cv_ranged_means(self.result) if self.is_cv else self.__get_ranged_means(self.result)

    def get_exact_matches(self):
        return self.__get_cv_exact_matches(self.result) if self.is_cv else self.__get_exact_matches(self.result)

    def __get_means(self, result):
        return result[self.metric].mean()

    def __get_cv_means(self, results):
        return np.mean([self.__get_means(result) for result in results])

    def __get_exact_matches(self, result):
        text_df = self.idx_to_text(self.experiment.data.df, result)
        count = text_df[text_df["test_comment"] ==
                        text_df["recommended_comment"]].shape[0]
        percentage = (count / text_df.shape[0]) * 100
        return count, percentage

    def __get_cv_exact_matches(self, results):
        exact_matches = [self.__get_exact_matches(
            result) for result in results]
        means = np.mean(exact_matches, axis=0)
        return means[0], means[1]

    def __get_ranged_means(self, result):
        df = result.copy()
        df["range"] = pd.cut(df[self.__distance_column], np.arange(0, 1.1, 0.1), labels=[
            f"{i / 10} - {(i + 1) / 10}" for i in range(10)], include_lowest=True)
        return df.groupby("range").mean().reset_index().loc[:, ["range", self.metric]]

    def __get_cv_ranged_means(self, results):
        ranged_means = [self.__get_ranged_means(result) for result in results]
        return pd.concat(ranged_means).groupby("range").mean().reset_index()

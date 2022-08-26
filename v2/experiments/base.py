import os
import time
import traceback
import pandas as pd, numpy as np
from dateutil import parser
from abc import ABC, abstractmethod
from configparser import ConfigParser
from dataclasses import dataclass, field

config = ConfigParser()
config.read(config.read('config.ini'))


@dataclass
class BaseExperiment(ABC):
    result_path = config["PATH"]["result_path"]
    results: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.path = os.path.join(
            self.result_path, self.name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.path, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def visualize_results(self):
        pass

    def save_results(self):
        for result_name, result in self.results.items():
            result.to_parquet(os.path.join(
                self.path, f"{result_name}.parquet"))

    def normalize_results_column(self, column: str):
        for result_name, result in self.results.items():
            result[column] = self.normalize_data(
                result[column])
        result[column] = self.normalize_data(result[column])

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __prepare_data(self, data_path: str, train_test_split: float, random_state: int):
        self.data = pd.read_csv(data_path)
        self.train = self.data.sample(
            frac=train_test_split, random_state=random_state)
        self.test = self.data.drop(self.train.index)

    def load_results(self, experiment_time: str):
        self.path = os.path.join(self.result_path, self.name, experiment_time)
        for result_name in os.listdir(self.path):
            if ".parquet" in result_name:
                self.results[result_name.removesuffix(".parquet")] = pd.read_parquet(os.path.join(
                    self.path, result_name))

    def write_info(self, msg):
        with open(os.path.join(self.path, "info.txt"), "a") as f:
            f.write(msg)

    def run(self, data_path: str, train_test_split: float = .95, random_state: int = 1):
        try:
            self.write_info(
                f"Experiment: {self.name.replace('_', ' ').title()}\n")
            self.__prepare_data(data_path, train_test_split, random_state)
            self.write_info(f"Train size: {len(self.train)}\n")
            self.write_info(f"Test size: {len(self.test)}\n")
            experiment_start = time.strftime("%X")
            self.write_info(f"Start: {experiment_start}\n")
            self._run()
            experiment_end = time.strftime("%X")
            self.write_info(f"End: {experiment_end}\n")
            execution_time = parser.parse(
                experiment_end) - parser.parse(experiment_start)
            self.write_info(f"Execution time: {execution_time}\n")
        except Exception:
            self.write_info(f"{traceback.format_exc()}")

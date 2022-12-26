from dataclasses import dataclass, field
from typing import Optional
import uuid
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from experiments.base import BaseExperiment


@dataclass
class CrossValidator:
    id: str
    experiment: BaseExperiment
    data: Optional[pd.DataFrame] = field(default=None)

    def get_folds(self, k, shuffle, random_state):
        data = self.data
        kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
            yield fold + 1, (data.iloc[train_idx], data.iloc[test_idx])

    def validate(self, k, shuffle, random_state):
        results = []
        for fold, (train, test) in self.get_folds(k, shuffle, random_state):
            print(f"Running fold {fold}...")
            exp = self.experiment()
            exp.run(train, test)
            results.append(exp)
        return results

    def save_results(self, results: list[BaseExperiment]):
        for i, result in enumerate(results):
            result.save_results(self.id, f"fold_{i+1}")

    def load_results(self):
        id = self.id
        results = []
        files = os.listdir(f"results/{self.experiment.name}/{id}")
        length = len([file for file in files if file.startswith("fold")])
        for i in range(length):
            exp = self.experiment()
            exp.load_results(f"{id}/fold_{i+1}")
            results.append(exp)
        return results
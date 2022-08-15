from __future__ import unicode_literals
import pandas as pd
import numpy

data = pd.read_csv('dataset/codeReviewDataset.csv', keep_default_na=False, na_values=[""], skip_blank_lines=True)
data = data.dropna()
data = data.drop_duplicates(subset=['diff_hunk','repo'])
data = data.reset_index(drop=True)

indices = pd.Series(data.index, index=data['reviewer_id'])

train_data = pd.DataFrame([], columns=data.columns)
test_data = pd.DataFrame([], columns=data.columns)

for index, row in data.iterrows():
    print(index)
    reviewer_indices = indices[row['reviewer_id']]
    if type(reviewer_indices) == numpy.int64 or len(reviewer_indices)<7:
        continue;
    first_index = reviewer_indices.iloc[0]
    if index <= first_index:
        size = reviewer_indices.size
        train_size = int(int(size) * 8 / 10)
        train_data = train_data.append(data.iloc[reviewer_indices.iloc[0:train_size]],ignore_index=True)
        test_data = test_data.append(data.iloc[reviewer_indices.iloc[train_size:]],ignore_index=True)


data = pd.read_csv('dataset/codeReviewDataset2-1.csv', keep_default_na=False, na_values=[""], skip_blank_lines=True)
data = data.dropna()
data = data.drop_duplicates(subset=['diff_hunk','repo'])
data = data.reset_index(drop=True)

indices = pd.Series(data.index, index=data['reviewer_id'])

for index, row in data.iterrows():
    print("2-", index)
    reviewer_indices = indices[row['reviewer_id']]
    if type(reviewer_indices) == numpy.int64 or len(reviewer_indices)<5:
        continue;
    first_index = reviewer_indices.iloc[0]
    if index <= first_index:
        size = reviewer_indices.size
        train_size = int(int(size) * 7 / 10)
        test_size = size - train_size
        train_data = train_data.append(data.iloc[reviewer_indices.iloc[0:train_size]],ignore_index=True)
        test_data = test_data.append(data.iloc[reviewer_indices.iloc[train_size:]],ignore_index=True)


train_data.to_csv('dataset/train_dataset.csv', encoding='utf-8', index=False)
test_data.to_csv('dataset/test_dataset.csv', encoding='utf-8', index=False)

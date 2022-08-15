# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:37:33 2021

@author: ykartal
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
import csv

start = time.time()

#train_data operations

train_data = pd.read_csv('dataset/train_dataset_lex.csv', keep_default_na=False, na_values=[""], skip_blank_lines=True, encoding='utf-8')
train_data = train_data.dropna()

print("Head: ", train_data.head())
print("Shape: ", train_data.shape)
print("Info: ", train_data.info)


#test_data operations
test_data = pd.read_csv('dataset/test_dataset_lex.csv', keep_default_na=False, na_values=[""], skip_blank_lines=True, encoding='utf-8')
test_data = test_data.dropna()

print("Head: ", test_data.head())
print("Shape: ", test_data.shape)
print("Info: ", test_data.info)

tfv_train = TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{3,}',
            ngram_range=(1, 3),
            stop_words = 'english')

train_data = train_data.append(test_data, ignore_index=True)
    
tfv_train_matrix = tfv_train.fit_transform(train_data['diff_hunk'].apply(lambda x: np.str_(x)))
#tfv_temp_matrix, tfv_test_matrix = train_test_split(tfv_train_matrix,test_size=0.2,shuffle=False)

#This function helps to find the most similar papers to specified paper.
def calc_similarity(method_name, content_matrix):
    if method_name == 'sigmoid_kernel':
        matrix = sigmoid_kernel(content_matrix, tfv_train_matrix, gamma = 0.8, coef0=0.5)
    elif method_name == 'linear_kernel':
        matrix = linear_kernel(content_matrix, tfv_train_matrix)
    elif method_name == 'euclidean_distances':
        matrix = euclidean_distances(content_matrix, tfv_train_matrix)
    elif method_name == 'cosine_similarity':
        matrix = cosine_similarity(content_matrix, tfv_train_matrix)
    elif method_name == 'pearsons_correlation':
        matrix = []
        # for i in range(tfv_train_matrix.size-1):
        #     tes = tfv_matrix.data[0][idx]
        #     matrix.append(pearsonr((tfv_matrix.data)[0][idx], tfv_matrix.data[0][i])[0])
    return matrix
result_stat_df = pd.DataFrame([], columns=['id', 'found', 'score'])
total_stat = 0
def give_rec(matrix, content, index):
    global result_stat_df
    global total_stat
    # Get the pairwise similarity scores 
    sig_scores = list(enumerate(matrix[0]))

    # Sort the paper 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar
    sig_scores = sig_scores[0:10]
    
    records = []
    count = 0
    source_code_id = train_data.loc[[index]].index[0]
    expected_reviewer_username = train_data['reviewer_username'].iloc[index]
    result_stat_df = result_stat_df.append({'id':source_code_id, 'found':0, 'score': 0},  ignore_index=True)
    while count<10 and count<len(sig_scores):
        if sig_scores[count][0]==index:
            count=count+1
            continue;
        tes = (train_data.iloc[[sig_scores[count][0]]])
        tes = tes['reviewer_username']
        # if sig_scores[count][1] > 0.8:
        reviewer_username = (train_data['reviewer_username'].iloc[sig_scores[count][0]])
        records.append([count,train_data.loc[[sig_scores[count][0]]].index[0],reviewer_username, sig_scores[count][1], expected_reviewer_username, train_data.loc[[index]].index[0]])
        # else:
            # break;
        count=count+1
        if reviewer_username == expected_reviewer_username:
            result_stat_df.loc[result_stat_df.shape[0]-1,['found']] = 1
            result_stat_df.loc[result_stat_df.shape[0]-1,['score']] = count
            total_stat+=1
            break;
        
            
    
    # cleaned_dict = dict()
    # for obj in records:
    #     if obj[2] not in cleaned_dict:
    #         cleaned_dict[obj[2]] = obj
    # records = list(cleaned_dict.values())
    
    df = pd.DataFrame(records, columns = ["order", "id", "reviewer_username", "score", "expected_reviewer", "expected_id"])
    return df


result_df = pd.DataFrame(columns = ["order", "id", "reviewer_username", "score", "expected_reviewer", "expected_id"])

def calcualte_sim(method, content, index):
    global result_df
    matrix = calc_similarity(method, content)
    # print(matrix[0].shape)
    
    #print("--------------------------------------------------------")
    # print(method, " Results:")
    calc_df = give_rec(matrix, content, index)
    result_df = result_df.append(calc_df)
    # print(calc_df)
    
sample_count=64553
for index in range(tfv_train_matrix.shape[0]-sample_count, tfv_train_matrix.shape[0]):
    calcualte_sim('linear_kernel', tfv_train_matrix[index], index);
    print("Test Index", index)

result_df.to_csv('dataset/results-'+str(sample_count)+'.csv', encoding='utf-8', index=True)
# calcualte_sim('linear_kernel');
# calcualte_sim('euclidean_distances');
# calcualte_sim('cosine_similarity');
# calcualte_sim('pearsons_correlation');


result_stat_df = result_stat_df.drop_duplicates(subset=['id'])
result_stat_df = result_stat_df.reset_index(drop=True)
result_stat_df.to_csv('result_stat-'+str(sample_count)+'.csv', encoding='utf-8', index=False)
print("Statistics; ")
print("Found Percentage: ", (total_stat*100/sample_count))
# print("Average Score-100: ", (total_stat*100/tfv_train_matrix.shape[0]))
# print("Average Score-10: ", (total_stat*100/tfv_train_matrix.shape[0]))


end = time.time()

print('Time: ', end - start)






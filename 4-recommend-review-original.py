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
from scipy.stats import pearsonr
import time


selected_code = '(public, Id) (class, Cl) (GlobalMetricNames, Id) ({, Symb) '

data = pd.read_csv('yedek/codeReviewDataset_lex_cleaned.csv', keep_default_na=False, na_values=[""], skip_blank_lines=True, encoding='utf-8')
data = data.dropna()
row = {'code': selected_code, 'review': 'NEW'}
data = data.append(row, ignore_index = True)

print("Head: ", data.head())
print("Shape: ", data.shape)
print("Info: ", data.info)

indices = pd.Series(data.index, index=data['code'])
idx = indices.size - 1
start = time.time()

ind = indices[selected_code]
print(ind)

tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{3,}',
            ngram_range=(1, 3),
            stop_words = 'english')

tfv_matrix = tfv.fit_transform(data['code'].apply(lambda x: np.str_(x)))

print(tfv_matrix.shape)



#This function helps to find the most similar papers to specified paper.
def calc_similarity(method_name):
    
    if method_name == 'sigmoid_kernel':
        content = tfv_matrix[idx]
        matrix = sigmoid_kernel(tfv_matrix[idx], tfv_matrix,gamma = 0.8, coef0=0.5)
    elif method_name == 'linear_kernel':
        matrix = linear_kernel(tfv_matrix[idx], tfv_matrix)
    elif method_name == 'euclidean_distances':
        matrix = euclidean_distances(tfv_matrix[idx])
    elif method_name == 'cosine_similarity':
        matrix = cosine_similarity(tfv_matrix[idx],tfv_matrix)
    elif method_name == 'pearsons_correlation':
        matrix = []
        for i in range(tfv_matrix.size-1):
            tes = tfv_matrix.data[0][idx]
            matrix.append(pearsonr((tfv_matrix.data)[0][idx], tfv_matrix.data[0][i])[0])
        
    return matrix

def give_rec(content, matrix):
    # Get the index corresponding to content
    #idx = indices[content]

    # Get the pairwise similarity scores 
    sig_scores = list(enumerate(matrix[0]))

    # Sort the paper 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar
    sig_scores = sig_scores[1:11]
    
    records = []
    count = 0
    while count<10 and count<len(sig_scores):
        tes = (data.iloc[[sig_scores[count][0]]])
        tes = tes['review']
        if sig_scores[count][1] > 0.8:
            records.append([(data.iloc[[sig_scores[count][0]]].review),(data['review'].iloc[sig_scores[count][0]]), sig_scores[count][1]])
        count=count+1

    df = pd.DataFrame(records, columns = ["id", "review", "score"])
    return df

def calcualte_sim(method):
    matrix = calc_similarity(method)
    print(matrix[0].shape)
    
    print("--------------------------------------------------------")
    print(method, " Results:")
    print(give_rec(selected_code, matrix).head(10))
    
calcualte_sim('sigmoid_kernel');
calcualte_sim('linear_kernel');
calcualte_sim('euclidean_distances');
calcualte_sim('cosine_similarity');
# calcualte_sim('pearsons_correlation');

end = time.time()

print(end - start)

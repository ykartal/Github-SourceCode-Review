'''
Creates a TF-IDF matrix and computes the cosine similarity between reviewed codes
and recommends review. Calculates blue score, meteor score and rouge-1 precision score.
Writes the results to a csv file.
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate import bleu_score, meteor_score
import math
from rouge import Rouge

CHUNK_SIZE = 1000

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_codes = train["code"]
test_codes = test["code"]


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_codes)
chencherry = bleu_score.SmoothingFunction()
rouge = Rouge()
results = []

for idx in range(math.ceil(len(test_codes) / CHUNK_SIZE)):
    if (idx + 1) * CHUNK_SIZE > len(test_codes):
        end = len(test_codes)
    else:
        end = (idx + 1) * CHUNK_SIZE
    test_chunk = test_codes[idx * CHUNK_SIZE : end]
    test_vectors = vectorizer.transform(test_chunk)
    similarities = cosine_similarity(test_vectors, train_vectors)
    for i, similarity in enumerate(similarities):
        max_sim = np.argmax(similarity)
        test_code = test.iloc[idx + i]["code"]
        test_comment = test.iloc[idx + i]["comment"]
        train_code = train.iloc[max_sim]["code"]
        train_comment = train.iloc[max_sim]["comment"]
        bleu_s = bleu_score.sentence_bleu([test_comment], train_comment, smoothing_function=chencherry.method1)
        meteor_s = meteor_score.single_meteor_score(test_comment.split(" "), train_comment.split(" "))
        rouge_s = rouge.get_scores(train_comment, test_comment)[0]['rouge-1']['p']
        results.append({
            "test_code": test_code,
            "test_comment": test_comment,
            "recommended_code": train_code,
            "recommended_comment": train_comment,
            "similarity": similarity[max_sim],
            "bleu": bleu_s,
            "meteor": meteor_s,
            "rouge": rouge_s
        })

pd.DataFrame(results).to_csv("data/results.csv",index=False)





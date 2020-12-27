# -*- coding: utf-8 -*-
import vectorizer
import lexical_analysis.lexical as lexical
import pandas as pd

data = pd.read_csv('codeReviewDataset.csv', keep_default_na=False, na_values=[""])

x_all=data.iloc[:,0]
for x in range(len(x_all)):
    try:
        x_all[x] = lexical.to_lex(x_all[x])
    except:
        print("Exception on lexical analysis")

data.iloc[:,0]=x_all
train = vectorizer.vectorize(data)    
print(train)

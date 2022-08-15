# -*- coding: utf-8 -*-
import lexical
import pandas as pd
result = ""
def toLex(dataset_name):
    global result
    data = pd.read_csv('dataset/'+dataset_name+'.csv', keep_default_na=False, na_values=[""], encoding='utf-8')
    for index, row in data.iterrows():
        lex_str = lexical.to_lex(row['diff_hunk'])
        if len(lex_str)<=5 or lex_str.startswith('(') == False:
            data.loc[data.iloc[index].name, 'diff_hunk'] = ""
        else:
            data.loc[data.iloc[index].name, 'diff_hunk'] = [lex_str]
        print(index, flush=True)
    
    data = data[data['diff_hunk']!=""]
    data = data.reset_index(drop=True)
    result = result + dataset_name + " Shape: "+ str(data.shape)+"\n"
    print(dataset_name, " Shape: ")
    print(data.shape)
    data.to_csv('dataset/'+dataset_name+'_lex.csv', encoding='utf-8', index=False)
                
toLex('train_dataset')
toLex('test_dataset')
print(result)
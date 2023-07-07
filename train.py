from libs.feature_extraction.vectorizers import vectorizers
import pandas as pd

print("Loading data...")
df = pd.read_pickle('./data/codesc_tokens.pkl')

for vectorizer in vectorizers.values():
    if vectorizer.name == 'fasttext-mean':
        vectorizer = vectorizer()
        print(f"{vectorizer.name} training..")
        x = df['tokens'] if vectorizer.take_tokenized_data else df['code']
        if vectorizer.trainable:
            vectorizer.fit(x)
            vectorizer.save('./model')
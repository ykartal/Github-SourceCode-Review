from dataclasses import dataclass
import pandas as pd
from libs.feature_extraction.vectorizers import BertVectorizer
from libs.metrics.vector_distance import Cosine
from experiment import Experiment

@dataclass
class ReviewBot:

    def review(self, code, candidate, history, history_vecs):
        vectorizer = BertVectorizer("microsoft/unixcoder-base")
        df = pd.read_csv(history)
        vectors = vectorizer.load_vectors(history_vecs)
        code_vec = vectorizer.transform([code])
        calculations = Cosine(code_vec, vectors)
        closest_samples_indexes = Experiment.get_closest_sample_indexes(calculations[0], candidate)
        recommendations = df.iloc[closest_samples_indexes]["comment"].tolist()
        print("\n".join(recommendations))

if __name__ == '__main__':
    review_bot = ReviewBot()
    review_bot.review("code", 5, "data/comment_finder/all.csv", "data/comment_finder/vectors/unix_base.npy")
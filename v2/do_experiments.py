from experiments.tfidf_vs_bow import TfidfvsBowExperiment
from experiments.vector_distance import VectorDistanceExperiment
from experiments.text_distance import TextDistanceExperiment

datapath = "data/comment_finder_data.csv"

tfidf_vs_bow_experiment = TfidfvsBowExperiment()
tfidf_vs_bow_experiment.run(datapath)
tfidf_vs_bow_experiment.save_results()
vector_distance_experiment = VectorDistanceExperiment()
vector_distance_experiment.run(datapath)
vector_distance_experiment.save_results()
text_distance_experiment = TextDistanceExperiment()
text_distance_experiment.run(datapath)
text_distance_experiment.save_results()

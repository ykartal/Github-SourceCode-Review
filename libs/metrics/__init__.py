from .text_similarity import Bleu, Meteor, Rouge
from .vector_distance import Cosine, Euclidean, Manhattan


text_similarities =  {
    Bleu.name: Bleu,
    Meteor.name: Meteor,
    Rouge.name: Rouge
}

vector_distances = {
    Cosine.name: Cosine,
    Euclidean.name: Euclidean,
    Manhattan.name: Manhattan
}

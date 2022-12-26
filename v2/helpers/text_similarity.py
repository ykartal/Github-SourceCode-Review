from dataclasses import dataclass
import textdistance
from nltk.translate import bleu_score, meteor_score
from rouge import Rouge as RougeObj
from typing import Callable


@dataclass
class TextSimilarity:
    name: str
    __function: Callable[[str, str], float]

    def __call__(self, hypothesis: str, reference: str) -> float:
        return self.__function(reference, hypothesis)


def __get_bleu_score(hypothesis: str, reference: str) -> float:
    chencherry = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method1)


def __get_meteor_score(hypothesis: str, reference: str) -> float:
    return meteor_score.single_meteor_score(reference.split(" "), hypothesis.split(" "))


def __get_rouge_score(hypothesis: str, reference: str) -> float:
    rouge = RougeObj()
    return rouge.get_scores(hypothesis, reference)[0]['rouge-1']['p']


Bleu = TextSimilarity("bleu", __get_bleu_score)

Meteor = TextSimilarity("meteor", __get_meteor_score)

Rouge = TextSimilarity("rouge", __get_rouge_score)

Jaccard = TextSimilarity("jaccard", textdistance.jaccard.normalized_similarity)

Jaro = TextSimilarity("jaro", textdistance.jaro.normalized_similarity)

JaroWinkler = TextSimilarity("jaro_winkler", textdistance.jaro_winkler.normalized_similarity)

Levensthein = TextSimilarity("levensthein", textdistance.levenshtein.normalized_similarity)

MongeElkan = TextSimilarity("monge_elkan", textdistance.monge_elkan.normalized_similarity)

Overlap = TextSimilarity("overlap", textdistance.overlap.normalized_similarity)

Dice = TextSimilarity("dice", textdistance.sorensen_dice.normalized_similarity)

Tversky = TextSimilarity("tversky", textdistance.tversky.normalized_similarity)

Hamming = TextSimilarity("hamming", textdistance.hamming.normalized_similarity)

Strcmp95 = TextSimilarity("strcmp95", textdistance.strcmp95.normalized_similarity)

Bag = TextSimilarity("bag", textdistance.bag.normalized_similarity)

Damerau = TextSimilarity("damerau", textdistance.damerau_levenshtein.normalized_similarity)

Ratcliff = TextSimilarity("ratcliff", textdistance.ratcliff_obershelp.normalized_similarity)

Mlipns = TextSimilarity("mlipns", textdistance.mlipns.normalized_similarity)

Cosine = TextSimilarity("cosine", textdistance.cosine.normalized_similarity)

Needleman = TextSimilarity("needleman", textdistance.needleman_wunsch.normalized_similarity)

Waterman = TextSimilarity("waterman", textdistance.smith_waterman.normalized_similarity)

Gotoh = TextSimilarity("gotoh", textdistance.gotoh.normalized_similarity)

metrics = [Bleu, Meteor, Rouge, Jaccard, Jaro, JaroWinkler, Levensthein, MongeElkan, Overlap, Dice, Tversky,
           Hamming, Strcmp95, Bag, Damerau, Ratcliff, Mlipns, Cosine, Needleman, Waterman, Gotoh]

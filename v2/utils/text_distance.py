from dataclasses import dataclass
import textdistance
from nltk.translate import bleu_score, meteor_score
from rouge import Rouge as RougeObj
from typing import Callable


@dataclass
class TextDistance:
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


Bleu = TextDistance("bleu", __get_bleu_score)

Meteor = TextDistance("meteor", __get_meteor_score)

Rouge = TextDistance("rouge", __get_rouge_score)

Jaccard = TextDistance("jaccard", textdistance.jaccard)

Jaro = TextDistance("jaro", textdistance.jaro)

JaroWinkler = TextDistance("jaro_winkler", textdistance.jaro_winkler)

Levensthein = TextDistance("Levensthein", textdistance.levenshtein)

MongeElkan = TextDistance("monge_elkan", textdistance.monge_elkan)

Overlap = TextDistance("overlap", textdistance.overlap)

Dice = TextDistance("dice", textdistance.sorensen_dice)

Tversky = TextDistance("tversky", textdistance.tversky)

Hamming = TextDistance("hamming", textdistance.hamming)

Strcmp95 = TextDistance("strcmp95", textdistance.strcmp95)

Bag = TextDistance("bag", textdistance.bag)

Damerau = TextDistance("damerau", textdistance.damerau_levenshtein)

LCSSeq = TextDistance("lcs_seq", textdistance.lcsseq)

LCSStr = TextDistance("lcs_str", textdistance.lcsstr)

Ratcliff = TextDistance("ratcliff", textdistance.ratcliff_obershelp)

Tanimoto = TextDistance("tanimoto", textdistance.tanimoto)

Mlipns = TextDistance("mlipns", textdistance.mlipns)

Cosine = TextDistance("cosine", textdistance.cosine)

Needleman = TextDistance("needleman", textdistance.needleman_wunsch)

Waterman = TextDistance("waterman", textdistance.smith_waterman)

Gotoh = TextDistance("gotoh", textdistance.gotoh)

metrics = [Bleu, Meteor, Rouge, Jaccard, Jaro, JaroWinkler, Levensthein, MongeElkan, Overlap, Dice, Tversky,
           Hamming, Strcmp95, Bag, Damerau, LCSSeq, LCSStr, Ratcliff, Tanimoto, Mlipns, Cosine, Needleman, Waterman, Gotoh]

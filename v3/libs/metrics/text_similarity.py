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
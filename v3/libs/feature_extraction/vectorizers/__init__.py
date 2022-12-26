from libs.feature_extraction.vectorizers.base import BaseVectorizer
from .traditional import TfIdfVectorizer, BowVectorizer
from .word_doc_based import Word2VecMeanVectorizer, Doc2VecVectorizer
from .transformer_based import BertVectorizer

vectorizers: dict[str, type[BaseVectorizer]] = {
    TfIdfVectorizer.name: TfIdfVectorizer,
    BowVectorizer.name: BowVectorizer,
    Word2VecMeanVectorizer.name: Word2VecMeanVectorizer,
    Doc2VecVectorizer.name: Doc2VecVectorizer,
    BertVectorizer.name: BertVectorizer
}
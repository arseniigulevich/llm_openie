from gensim.matutils import kullback_leibler
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from source.converter import combine_extraction_texts, triples_to_extraction_texts


def _ensure_nltk_resource(resource: str, path_hint: str) -> None:
    try:
        nltk.data.find(path_hint)
    except LookupError:
        nltk.download(resource, quiet=True)


_ensure_nltk_resource('stopwords', 'corpora/stopwords')
_ensure_nltk_resource('punkt', 'tokenizers/punkt')

# Eagerly load stopwords to avoid LazyCorpusLoader races under threaded execution.
STOP_WORDS = frozenset(stopwords.words('english'))


def preprocess(text):
    words = word_tokenize(text)
    return [word for word in words if word not in STOP_WORDS and word.isalnum()]


def calculate_ts_score(data, dictionary, lda, output_all_scores=False):
    """LDA-based topical similarity between source text and extracted triples.

    Score per sentence = exp(-KL(LDA(source) || LDA(triples))).
    Returns mean over all sentences, and optionally per-sentence scores.
    """
    all_ts_scores = {}
    for source_text in data.keys():
        triples = data[source_text]
        triples_str = combine_extraction_texts(triples_to_extraction_texts(triples))
        processed_source = preprocess(source_text)
        processed_triples = preprocess(triples_str)
        source_corpus = dictionary.doc2bow(processed_source)
        triples_corpus = dictionary.doc2bow(processed_triples)

        source_dist = lda.get_document_topics(source_corpus, minimum_probability=0)
        triples_dist = lda.get_document_topics(triples_corpus, minimum_probability=0)

        kl_div = max(0.0, kullback_leibler(source_dist, triples_dist))
        all_ts_scores[source_text] = math.exp(-kl_div)

    average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)

    if output_all_scores:
        return average_ts_score, all_ts_scores
    return average_ts_score

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from source.embeddings import load_element_embeddings, get_triple_embedding


def _ensure_embeddings_loaded():
    return load_element_embeddings()


def _get_triple_embedding(triple):
    return get_triple_embedding(triple, _ensure_embeddings_loaded())


def calculate_uniqueness(vectors, phi=0.95):
    """Uniqueness score: fraction of off-diagonal pairs with cosine similarity < phi."""
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 1)
    count_smaller_than_phi = np.sum(similarity_matrix < phi)
    total_pairs = len(vectors) * (len(vectors) - 1)
    return count_smaller_than_phi / total_pairs if total_pairs > 0 else 1


def calculate_uniqueness_for_text(triples):
    vectors = []
    for triple in triples:
        try:
            vectors.append(_get_triple_embedding(triple))
        except Exception:
            continue
    try:
        return calculate_uniqueness(np.array(vectors))
    except Exception:
        return 0


def calculate_uniqueness_score(data_to_evaluate, output_all_scores=False):
    """Embedding-based uniqueness score averaged over all sentences."""
    scores = {}
    for text, triples in tqdm(data_to_evaluate.items()):
        scores[text] = calculate_uniqueness_for_text(triples) if triples else 0

    avg_score = np.mean(list(scores.values()))
    if output_all_scores:
        return avg_score, scores
    return avg_score

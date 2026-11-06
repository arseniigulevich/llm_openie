from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from gensim.models import CoherenceModel

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]

#for each dataset estimate the best k
def estimate_k(processed_texts, dictionary, corpus, k_min=10, k_max=150, step=5,
               rel_impr_thresh=0.005, patience=2, random_state=0):
    scores, best_k, best_score, flat_steps = [], None, -1, 0
    for k in range(k_min, k_max + 1, step):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                       random_state=random_state, chunksize=2000,
                       passes=5, iterations=50, alpha='asymmetric', eval_every=None)
        cm = CoherenceModel(model=lda, texts=processed_texts, dictionary=dictionary,
                            coherence='c_v')
        score = cm.get_coherence()
        scores.append((k, score))
        if score > best_score * (1 + rel_impr_thresh):
            best_k, best_score, flat_steps = k, score, 0
        else:
            flat_steps += 1
            if flat_steps >= patience:
                break
    return best_k, scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPICAL_DIR = Path(__file__).resolve().parent

DATASET_TEXT_PATHS = {
    "RE-OIE-2016": PROJECT_ROOT / "data" / "RE-OIE-2016" / "Re-OIE2016_all_text.json",
    "ROBUST": PROJECT_ROOT / "data" / "ROBUST" / "ROBUST_all_text.json",
    # MINE: trained on the article SENTENCES (see data/MINE/MINE_all_text.json),
    # because the 101 whole articles are too few for a stable topic model. Note
    # that TS scores MINE at the ARTICLE level (the source unit is a full essay),
    # so the metric is coarser and near-saturated there -- see meta/codebase_summary.md.
    "mine": PROJECT_ROOT / "data" / "MINE" / "MINE_all_text.json",
}

datasets = ["mine"]

for dataset in tqdm(datasets):
    print(f"Training LDA model for {dataset}...")

    data_path = DATASET_TEXT_PATHS.get(dataset)
    if data_path is None or not data_path.exists():
        print(f"No data file found for {dataset}: {data_path}, skipping...")
        continue

    with data_path.open('r', encoding='utf-8') as f:
        all_source_texts = json.load(f)
    print(f"Loaded {len(all_source_texts)} sentences from {data_path}")

    # Preprocess all texts
    processed_texts = [preprocess(text) for text in all_source_texts]
    processed_texts = [tokens for tokens in processed_texts if tokens]
    if not processed_texts:
        print(f"No non-empty tokenized texts for {dataset}, skipping...")
        continue

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    with (TOPICAL_DIR / f'{dataset}_dictionary.pkl').open('wb') as f:
        pickle.dump(dictionary, f)
    
    best_k, scores = estimate_k(processed_texts, dictionary, corpus)
    print("Best K (coarse):", best_k)
    print("Scores:", scores)

    # Train LDA model
    lda = LdaModel(corpus, num_topics=best_k, id2word=dictionary)

    with (TOPICAL_DIR / f'{dataset}_lda.pkl').open('wb') as f:
        pickle.dump(lda, f)

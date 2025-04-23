# src/feature_extraction.py
"""
Feature extraction module for authorship attribution pipeline.
Transforms preprocessed records into numeric feature vectors for modeling.

Provides:
- Stylometric features: sentence length, TTR, hapax legomena, function-word frequencies
- Character n-gram features
- POS tag distribution features
- Embedding features via sentence-transformers

Functions to import and use in a notebook:
    from feature_extraction import (
        extract_stylometric_features,
        char_ngram_features,
        pos_tag_features,
        embedding_features
    )

Usage example:
    processed_df = pd.read_pickle('data/processed.pkl')
    stylom_df = extract_stylometric_features(processed_df)
    ngram_df = char_ngram_features(processed_df)
    pos_df = pos_tag_features(processed_df)
    emb_df = embedding_features(processed_df)
    features = stylom_df.join([ngram_df, pos_df, emb_df])
"""
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import pos_tag

# Ensure NLTK tagger data
nltk.download('averaged_perceptron_tagger', quiet=True)

from sentence_transformers import SentenceTransformer

# List of common function words
FUNCTION_WORDS = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for',
    'you', 'was', 'with', 'on', 'as', 'i', 'but', 'be', 'at', 'by'
]

# POS categories mapping
POS_CATEGORIES = {
    'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj' : ['JJ', 'JJR', 'JJS'],
    'adv' : ['RB', 'RBR', 'RBS'],
    'pron': ['PRP', 'PRP$'],
    'prep': ['IN']
}


def avg_sentence_length(words, sentences):
    if not sentences:
        return 0.0
    return len(words) / len(sentences)


def type_token_ratio(words):
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hapax_legomena_ratio(words):
    if not words:
        return 0.0
    freqs = Counter(words)
    hapaxes = sum(1 for c in freqs.values() if c == 1)
    return hapaxes / len(words)


def function_word_freqs(words):
    freqs = Counter(words)
    total = len(words)
    return {f'w_freq_{fw}': freqs.get(fw, 0) / total for fw in FUNCTION_WORDS}


def extract_stylometric_features(processed_df):
    """Compute basic stylometric features."""
    rows = []
    for _, row in processed_df.iterrows():
        words = row['tokens']
        sentences = row.get('sentences', [])
        feats = {
            'file_path': row['file_path'],
            'author': row['author'],
            'prompt': row['prompt'],
            'split': row['split'],
            'avg_sentence_length': avg_sentence_length(words, sentences),
            'type_token_ratio': type_token_ratio(words),
            'hapax_legomena_ratio': hapax_legomena_ratio(words)
        }
        feats.update(function_word_freqs(words))
        rows.append(feats)
    return pd.DataFrame(rows).set_index('file_path')


def char_ngram_features(processed_df, ngram_range=(3,5), max_features=100):
    """Extract normalized character n-gram frequencies."""
    texts = processed_df['clean_text'].tolist()
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(texts)
    df_ngrams = pd.DataFrame(X.toarray(), columns=[f'ngram_{g}' for g in vectorizer.get_feature_names_out()])
    df_ngrams = df_ngrams.div(df_ngrams.sum(axis=1), axis=0).fillna(0)
    df_ngrams.index = processed_df['file_path']
    return df_ngrams


def pos_tag_features(processed_df):
    """Compute POS tag distribution ratios for each category."""
    rows = []
    for _, row in processed_df.iterrows():
        words = row['tokens']
        tags = [t for _, t in pos_tag(words)]
        total = len(tags)
        counts = Counter(tags)
        feats = {}
        for cat, taglist in POS_CATEGORIES.items():
            count = sum(counts.get(t, 0) for t in taglist)
            feats[f'pos_{cat}_ratio'] = count / total if total else 0.0
        rows.append(feats)
    df_pos = pd.DataFrame(rows)
    df_pos.index = processed_df['file_path']
    return df_pos


def embedding_features(processed_df, model_name='all-MiniLM-L6-v2', batch_size=16):
    """Generate sentence embeddings for each document."""
    model = SentenceTransformer(model_name)
    sentences = processed_df['clean_text'].tolist()
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    df_emb = pd.DataFrame(embeddings, index=processed_df['file_path'])
    df_emb.columns = [f'emb_{i}' for i in range(df_emb.shape[1])]
    return df_emb


# Optional script entrypoint for standalone execution
if __name__ == '__main__':
    import pandas as pd
    processed = pd.read_pickle('data/processed.pkl')
    stylom = extract_stylometric_features(processed)
    ngrams = char_ngram_features(processed)
    posd = pos_tag_features(processed)
    emb = embedding_features(processed)
    features = stylom.join([ngrams, posd, emb])
    features.reset_index().to_csv('data/features.csv', index=False)
    print(f"Saved features matrix with shape {features.shape} to data/features.csv")
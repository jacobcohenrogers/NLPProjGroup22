# src/preprocessing.py
"""
Text preprocessing module for authorship attribution pipeline.
Provides functions to clean and tokenize raw text files without relying on NLTK resources.

Usage example:
    from preprocessing import load_and_preprocess
    result = load_and_preprocess('data/raw_text/training/Rivera/rivera_0.txt')
    print(result['words'])
"""
"""Lowercase, expand contractions, selectively remove punctuation, and normalize whitespace."""
import re
import string
from pathlib import Path

# Contraction mapping (expand common contractions)
CONTRACTION_MAP = {
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
    "that's": "that is",
    "you're": "you are",
    # add more as needed
}

# Compile regex for contractions and punctuation
_contraction_pattern = re.compile(
    r"(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")",
    flags=re.IGNORECASE
)
# Better for poetry- only remove quotes and brackets
_keep_punctuation = ['.', ',', ';', '-', ':', '?', '!']
_punct_pattern = re.compile(f"[{re.escape(''.join(c for c in string.punctuation if c not in _keep_punctuation))}]")


def expand_contractions(text: str) -> str:
    """Replace contractions in text using CONTRACTION_MAP."""
    def replace(match):
        c = match.group(0).lower()
        return CONTRACTION_MAP.get(c, c)
    return _contraction_pattern.sub(replace, text)


def clean_text(text: str) -> str:
    """Lowercase, expand contractions, remove punctuation, and normalize whitespace."""
    text = text.lower()
    text = expand_contractions(text)
    text = _punct_pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_sentence_split(text: str) -> list:
    """Split text into sentences using punctuation (.!?)."""
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def preprocess_text(text: str) -> dict:
    """
    Perform full preprocessing on raw text:
      - Clean text
      - Simple sentence splitting
      - Word tokenization via whitespace
    Returns a dict with keys: clean_text, sentences, words
    """
    clean = clean_text(text)
    sentences = simple_sentence_split(clean)
    words = clean.split()
    return {
        'clean_text': clean,
        'sentences': sentences,
        'words': words
    }


def load_and_preprocess(file_path: str or Path) -> dict:
    """
    Load text from file and apply preprocessing.
    """
    file_path = Path(file_path)
    raw = file_path.read_text(encoding='utf-8')
    return preprocess_text(raw)
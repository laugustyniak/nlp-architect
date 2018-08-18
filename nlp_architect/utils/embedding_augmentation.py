from functools import lru_cache
from typing import List

import spacy


@lru_cache(None)
def load_spacy(model):
    print('Start loading spacy model: ', model)
    return spacy.load(model)


@lru_cache(None)
def most_similar(word: str, threshold: float = 0.8):
    nlp = load_spacy('en_vectors_web_lg')
    word = nlp.vocab[word]
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    return [token for token in queries if word.similarity(token) > threshold and _filter_pos(token)]


@lru_cache(None)
def top_n_most_similar(word, n=10):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    return sorted(queries, key=lambda w: word.similarity(w), reverse=True)[:n]


def filter_words_to_augment(tokens: List[str]):
    nlp = load_spacy('en_core_web_sm')
    tokens = nlp(' '.join(tokens))
    return [
        (token.text, True, token_id)
        if _word_filter(token)
        else (token.text, False, token_id)
        for token_id, token
        in enumerate(tokens)
    ]


@lru_cache(maxsize=100000)
def _word_filter(token):
    # return not token.is_stop and not token.is_digit and not token.is_punct
    return token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']


@lru_cache(maxsize=100000)
def _filter_pos(word: str):
    nlp = load_spacy('en_core_web_sm')
    return _word_filter([w for w in nlp(word.text)][0])

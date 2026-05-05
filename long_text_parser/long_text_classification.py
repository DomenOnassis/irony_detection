from __future__ import annotations

import sys
from pathlib import Path
import re

import joblib
import numpy as np
from ekphrasis.utils.nlp import polarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slovene_pipeline"))

LONG_TEXT_DIRECTORY = Path(__file__).resolve().parent / "output"
MODEL_PATH = Path(__file__).resolve().parents[1] / "slovene_pipeline" / "subtaskA_notebooks" / "finalized_model.sav"
OUTPUT_FILE = Path(__file__).resolve().parent / "classification_results.txt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_COMPARE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

POSITIVE_WORDS = [
    "rad", "obožujem", "lepo", "super", "odličko", "hvala", "rabi", "rabu",
    "great", "love", "perfect", "amazing", "wonderful", "fantastic",
]
NEGATIVE_WORDS = [
    "zaprta", "zaprto", "zaprte", "slabo", "čudno", "groza", "strašno",
    "problematično", "dead", "horrible", "ugly", "terrible", "bad",
]
SLOVENE_NEGATIONS = ["ne", "nema", "nimam", "nimajo", "nič", "nikoli", "nobeden", "brez"]
EMOJI_SENTIMENTS_MAP = {
    "🙄": -1, "😒": -1, "😑": -1, "🤦": -1, "🤦\u200d♂️": -1, "🤦\u200d♀️": -1,
    "😬": -1, "😏": -1, "🙃": -1, "🙂": 1, "😊": 1, "😁": 1,
}
EMOJI_IRONY_SET = {"🙄", "😒", "😑", "🤦", "🤦\u200d♂️", "🤦\u200d♀️", "😬", "😏", "🙃"}
INCONVENIENCE_PATTERNS = [
    r"\b(še\s+dobro\s+da)\b",
    r"\b(super|odlično|fajn|hvala|vrhunsko)\b.*\b(ko|da)\b.*\b(zamujam|dežuje|zaprta|gužva|gneča|pokvar|crkn|čakanja|čakalnici)",
]
NEG_EVENT_HINTS = ["zamujam", "dežuje", "zaprta", "gužva", "gneča", "pokvar", "crkn", "čakanja", "čakalnici"]
POS_IRONIC_OPENERS = ["super", "odlično", "fajn", "hvala", "vrhunsko"]


def load_classifier(model_path: Path):
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        return bundle["classifier"]

    return bundle


def classify_text(text: str, classifier) -> float:
    features_array = np.asarray(extract_enhanced_features(text), dtype=float).reshape(1, -1)
    probabilities = np.asarray(classifier.predict_proba(features_array))

    return float(probabilities[0, 1])


def classify_by_threshold(probability: float, threshold: float) -> str:
    return "Ironic" if probability >= threshold else "Not Ironic"


def iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".txt" and path.name.startswith("sentence_"):
            yield path


def iter_ngram_lines(text_file: Path):
    for line_number, line in enumerate(text_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        ngram = line.strip()
        if ngram:
            yield line_number, ngram


def extract_enhanced_features(text: str):
    words = text.split()
    words_lower = [word.lower() for word in words]

    positive_count = sum(
        1 for word in words_lower if any(word.startswith(pos_word) or pos_word in word for pos_word in POSITIVE_WORDS))
    negative_count = sum(
        1 for word in words_lower if any(word.startswith(neg_word) or neg_word in word for neg_word in NEGATIVE_WORDS))
    sarcasm_contrast = 1 if (positive_count > 0 and negative_count > 0) else 0

    left_half = words[:len(words) // 2]
    right_half = words[len(words) // 2:]
    left_word_lens = [len(word) for word in left_half]
    right_word_lens = [len(word) for word in right_half]

    left_intensity = 1 if (sum(left_word_lens) / max(len(left_half), 1)) < 4 else 0
    right_intensity = 1 if (sum(right_word_lens) / max(len(right_half), 1)) < 4 else 0
    polarity_diff = 1 if len(left_half) > 0 and len(right_half) > 0 else 0

    contrast = 0
    if len(left_half) > 0 and len(right_half) > 0 and polarity is not None:
        try:
            left_text = ' '.join(left_half)
            right_text = ' '.join(right_half)
            left_polarity = polarity(left_text)
            right_polarity = polarity(right_text)
            if left_polarity and right_polarity and len(left_polarity) > 1 and len(right_polarity) > 1:
                contrast = 1 if abs(left_polarity[0] - right_polarity[0]) > 0.5 else 0
        except Exception:
            contrast = 0

    exclamation_count = text.count('!')
    question_count = text.count('?')
    ellipsis_count = text.count('...')
    ellipsis_signal = 1 if ellipsis_count > 0 else 0
    excessive_punct = 1 if (exclamation_count > 2 or question_count > 2) else 0

    elongated_words = len([word for word in words if re.search(r'(.)\1{2,}', word)])
    elongation_score = min(elongated_words / max(len(words), 1), 1.0)

    negation_count = sum(
        1 for word in words_lower if any(word.startswith(neg) or neg in word for neg in SLOVENE_NEGATIONS))
    negation_score = min(negation_count / max(len(words), 1), 1.0)

    emoji_vals = [sentiment_value for emoji, sentiment_value in EMOJI_SENTIMENTS_MAP.items() if emoji in text]

    if emoji_vals:
        neg_ratio = sum(1 for sentiment_value in emoji_vals if sentiment_value < 0) / len(emoji_vals)
        mixed = 1.0 if ({-1, 1}.issubset(set(emoji_vals))) else 0.0
        emoji_sentiment_prior = min(1.0, 0.7 * neg_ratio + 0.3 * mixed)
    else:
        emoji_sentiment_prior = 0.0

    hashtag_tokens = [token[1:] for token in text.lower().split() if token.startswith('#') and len(token) > 1]
    pos_hash = {'super', 'odlicno', 'vrhunsko', 'hvala', 'top'}
    neg_hash = {'katastrofa', 'groza', 'slabo', 'zamuda', 'problem'}
    pos_hits = sum(1 for hashtag in hashtag_tokens if any(pos_pattern in hashtag for pos_pattern in pos_hash))
    neg_hits = sum(1 for hashtag in hashtag_tokens if any(neg_pattern in hashtag for neg_pattern in neg_hash))

    if (pos_hits + neg_hits) == 0:
        hashtag_sentiment_prior = 0.0
    else:
        mixed = 1.0 if (pos_hits > 0 and neg_hits > 0) else 0.0
        spread = abs(pos_hits - neg_hits) / max((pos_hits + neg_hits), 1)
        hashtag_sentiment_prior = min(1.0, 0.6 * mixed + 0.4 * (1.0 - spread))

    emoji_hits = sum(text.count(emoji) for emoji in EMOJI_IRONY_SET)
    emoji_irony_prior = min(0.35, 0.12 * min(emoji_hits, 2) + (0.08 if emoji_hits > 0 else 0.0))

    text_lower = text.lower()
    pattern_hits = sum(1 for pattern in INCONVENIENCE_PATTERNS if re.search(pattern, text_lower))
    neg_event_hits = sum(1 for token in NEG_EVENT_HINTS if token in text_lower)
    opener = 1 if any(token in text_lower for token in POS_IRONIC_OPENERS) else 0
    inconvenience_prior = min(0.35, 0.10 * pattern_hits + 0.05 * min(neg_event_hits, 3) + 0.05 * opener)

    core = [left_intensity, right_intensity, polarity_diff, contrast]

    auxiliary_features = np.zeros(54, dtype=float)
    auxiliary_features[0] = exclamation_count / 5
    auxiliary_features[1] = question_count / 5
    auxiliary_features[2] = ellipsis_count / 3
    auxiliary_features[3] = excessive_punct
    auxiliary_features[4] = elongation_score
    auxiliary_features[5] = negation_score
    auxiliary_features[6] = sarcasm_contrast
    auxiliary_features[7] = ellipsis_signal
    auxiliary_features[8] = positive_count / max(len(words), 1)
    auxiliary_features[9] = negative_count / max(len(words), 1)
    auxiliary_features[52] = emoji_sentiment_prior
    auxiliary_features[53] = hashtag_sentiment_prior

    all_features = core + auxiliary_features.tolist() + [emoji_irony_prior, inconvenience_prior]

    return all_features


def process_text_file(text_file: Path, input_dir: Path, classifier, threshold: float, compare_thresholds, output_file):
    display_path = text_file.relative_to(input_dir)
    output_file.write(f"FILE: {display_path}\n")

    ngram_probabilities = []

    for line_number, ngram in iter_ngram_lines(text_file):
        irony_probability = classify_text(ngram, classifier)
        ngram_probabilities.append(irony_probability)

    sentence_probability = float(np.mean(ngram_probabilities))
    all_thresholds = sorted({threshold, *compare_thresholds})

    output_file.write(f"\t[PROBABILITY]: {sentence_probability:.4f}\n")

    for current_threshold in all_thresholds:
        sentence_result = classify_by_threshold(sentence_probability, current_threshold)
        output_file.write(f"\t[THRESHOLD = {current_threshold}]: {sentence_result}\n")

    output_file.write("-" * 80 + "\n")


if __name__ == "__main__":
    classifier = load_classifier(MODEL_PATH)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for text_file in iter_text_files(LONG_TEXT_DIRECTORY):
            process_text_file(text_file, LONG_TEXT_DIRECTORY, classifier, DEFAULT_THRESHOLD, DEFAULT_COMPARE_THRESHOLDS,
                              output_file)

    print(f"Results saved to:", output_file.name)

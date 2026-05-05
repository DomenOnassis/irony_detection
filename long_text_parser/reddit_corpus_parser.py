import argparse
import os
import re
import pandas as pd


# Sentence splitting
def split_into_sentences(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []

    # Split on '. ' or '.\n' or end-of-string after a dot.
    # We use a regex that keeps the dot as part of the preceding segment.
    raw_parts = re.split(r'(?<=\.)\s+', text.strip())

    sentences = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Must start with an uppercase letter (ignore leading non-alpha chars)
        first_alpha = re.search(r'[A-Za-z]', part)
        if first_alpha and part[first_alpha.start()].isupper():
            sentences.append(part)

    return sentences


# Tokenisation
def tokenize(sentence: str) -> list[str]:
    # Insert spaces around punctuation clusters while keeping emoji-like
    # sequences (e.g., ':)') together.  Strategy:
    #   1. Split on whitespace to get rough tokens.
    #   2. For each rough token, peel off leading/trailing punctuation.
    tokens = []
    for raw in sentence.split():
        # Peel leading punctuation (but not ':' that starts emoticons)
        leading = re.match(r'^([^\w:]+)', raw)
        if leading:
            tokens.append(leading.group(1))
            raw = raw[leading.end():]

        # Peel trailing punctuation
        trailing = re.search(r'([^\w)]+)$', raw)
        if trailing and trailing.start() > 0:
            core = raw[:trailing.start()]
            tail = raw[trailing.start():]
            if core:
                tokens.append(core)
            tokens.append(tail)
        else:
            if raw:
                tokens.append(raw)

    return tokens


# N-gram generation
def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# File writing
def write_ngram_file(path: str, ngrams: list[tuple[str, ...]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for gram in ngrams:
            fh.write(' '.join(gram) + '\n')


# Text extraction — one list of raw text blocks per format
def read_texts_from_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    if 'Post Text' not in df.columns:
        raise ValueError(
            f"CSV does not contain a 'Post Text' column. "
            f"Found columns: {df.columns.tolist()}"
        )
    return [t for t in df['Post Text'] if isinstance(t, str) and t.strip()]


def read_texts_from_txt(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    return [content] if content.strip() else []


# Main corpus processing
def process_corpus(input_path: str, n: int = 2, output_dir: str = 'output') -> None:
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.csv':
        texts = read_texts_from_csv(input_path)
    elif ext == '.txt':
        texts = read_texts_from_txt(input_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supply a .csv or .txt file."
        )

    corpus_name = os.path.splitext(os.path.basename(input_path))[0]
    corpus_dir = os.path.join(output_dir, f"{n}_gram_{corpus_name}")
    os.makedirs(corpus_dir, exist_ok=True)

    sentence_index = 0  # global sentence counter across all text blocks

    for text in texts:
        for sentence in split_into_sentences(text):
            tokens = tokenize(sentence)
            ngrams = generate_ngrams(tokens, n)

            if not ngrams:
                continue  # sentence too short for the chosen n-gram size

            sentence_index += 1
            file_path = os.path.join(corpus_dir, f'sentence_{sentence_index}.txt')
            write_ngram_file(file_path, ngrams)

    print(
        f"Done. Processed {sentence_index} sentence(s) from '{input_path}'.\n"
        f"Output written to: {corpus_dir}/"
    )


# CLI entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse a corpus (.csv or .txt) into word n-gram files for irony classification.'
    )
    parser.add_argument('input_file', help='Path to the input corpus file (.csv or .txt).')
    parser.add_argument(
        '--n', type=int, default=2, metavar='N',
        help='N-gram size (default: 2).'
    )
    parser.add_argument(
        '--output-dir', default='output', metavar='OUTPUT_DIR',
        help='Root output directory (default: ./output).'
    )
    return parser.parse_args()


# Example program argument: <path_to_corpus> --n <n> --output <path_to_output_folder>
# long_slovenian_reddit_corpora/mock_txt_corpus.txt --n 3 --output-dir ./output
if __name__ == '__main__':
    args = parse_args()
    process_corpus(
        input_path=args.input_file,
        n=args.n,
        output_dir=args.output_dir,
    )
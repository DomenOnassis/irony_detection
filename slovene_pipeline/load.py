from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Union


def _read_tsv_rows(dataset: Union[str, Path]) -> List[List[str]]:
    with open(dataset, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return [row for row in reader if row]


def parse_dataset(dataset: Union[str, Path]) -> Union[List[str], Tuple[List[str], List[int]]]:
    """Loads SemEval-style TSV and returns corpus (+ labels for train/gold files)."""
    dataset = Path(dataset)
    dataset_name = dataset.name.lower()
    rows = _read_tsv_rows(dataset)

    if rows and rows[0] and "tweet" in rows[0][0].lower():
        rows = rows[1:]

    y: List[int] = []
    corpus: List[str] = []
    is_labeled = ("train" in dataset_name) or ("gold" in dataset_name)

    for row in rows:
        if not row:
            continue

        if is_labeled:
            if len(row) < 3:
                continue
            try:
                y.append(int(row[1]))
            except ValueError:
                continue
            corpus.append(row[2])
        else:
            corpus.append(row[-1])

    if is_labeled:
        return corpus, y
    return corpus

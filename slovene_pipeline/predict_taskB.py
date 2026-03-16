from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from features import build_tfidf, build_w2v_mean, combine_features
from load import parse_dataset
from word2vec_api import Word2VecAPI, maybe_load_emoji2vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with trained Slovene Task B model.")
    parser.add_argument("--model", default="slovene_pipeline/model_taskB.joblib")
    parser.add_argument("--input", default="datasets/slovene/test_TaskB/SemEval2018-T3_input_test_taskB_emoji.txt")
    parser.add_argument("--w2v-path", default="../w2v-slo/all-token-prelim.ft.sg.bin")
    parser.add_argument("--w2v-binary", action="store_true", default=True)
    parser.add_argument("--emoji2vec-path", default="")
    parser.add_argument("--emoji2vec-binary", action="store_true")
    parser.add_argument("--pred-out", default="slovene_pipeline/predictions_taskB.txt")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    vectorizer = bundle["vectorizer"]
    clf = bundle["classifier"]

    corpus = parse_dataset(args.input)
    w2v = Word2VecAPI.load_word2vec_format(args.w2v_path, binary=args.w2v_binary)
    emoji_model = maybe_load_emoji2vec(args.emoji2vec_path, binary=args.emoji2vec_binary)

    tfidf_x, _ = build_tfidf(corpus, fitted_vectorizer=vectorizer)
    w2v_x = build_w2v_mean(corpus, w2v, emoji_model)
    x = combine_features(tfidf_x, w2v_x)

    preds = clf.predict(x)
    out_path = Path(args.pred_out)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        for p in preds:
            handle.write(f"{int(p)}\n")

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

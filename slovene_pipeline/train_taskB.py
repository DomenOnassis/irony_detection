from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from api_hooks import build_api_features
from features import build_tfidf, build_w2v_mean, combine_features
from load import parse_dataset
from word2vec_api import Word2VecAPI, maybe_load_emoji2vec


def _write_predictions(path: Path, preds: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for p in preds:
            handle.write(f"{int(p)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Slovene Task B model.")
    parser.add_argument("--train", default="datasets/slovene/train/SemEval2018-T3-train-taskB_emoji.txt")
    parser.add_argument("--test", default="datasets/slovene/test_TaskB/SemEval2018-T3_input_test_taskB_emoji.txt")
    parser.add_argument("--gold", default="datasets/slovene/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt")
    parser.add_argument("--w2v-path", default="../w2v-slo/all-token-prelim.ft.sg.bin")
    parser.add_argument("--w2v-binary", action="store_true", default=True)
    parser.add_argument("--emoji2vec-path", default="")
    parser.add_argument("--emoji2vec-binary", action="store_true")
    parser.add_argument("--use-api-features", action="store_true")
    parser.add_argument("--model-out", default="slovene_pipeline/model_taskB.joblib")
    parser.add_argument("--pred-out", default="slovene_pipeline/predictions_taskB.txt")
    args = parser.parse_args()

    train_corpus, y_train = parse_dataset(args.train)
    y_train = np.array(y_train, dtype=np.int32)

    w2v = Word2VecAPI.load_word2vec_format(args.w2v_path, binary=args.w2v_binary)
    emoji_model = maybe_load_emoji2vec(args.emoji2vec_path, binary=args.emoji2vec_binary)

    tfidf_train, vectorizer = build_tfidf(train_corpus)
    w2v_train = build_w2v_mean(train_corpus, w2v, emoji_model)
    x_train = combine_features(tfidf_train, w2v_train)
    if args.use_api_features:
        api_train = build_api_features(train_corpus, task="B")
        if api_train.shape[1] > 0:
            x_train = combine_features(x_train, api_train)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(clf, x_train, y_train, cv=cv)
    cv_f1_per_class = f1_score(y_train, y_cv, average=None)
    cv_f1_macro = f1_score(y_train, y_cv, average="macro")
    print(f"Task B CV F1 per class: {cv_f1_per_class}")
    print(f"Task B CV F1 macro: {cv_f1_macro:.4f}")

    clf.fit(x_train, y_train)

    test_corpus = parse_dataset(args.test)
    tfidf_test, _ = build_tfidf(test_corpus, fitted_vectorizer=vectorizer)
    w2v_test = build_w2v_mean(test_corpus, w2v, emoji_model)
    x_test = combine_features(tfidf_test, w2v_test)
    if args.use_api_features:
        api_test = build_api_features(test_corpus, task="B")
        if api_test.shape[1] > 0:
            x_test = combine_features(x_test, api_test)
    y_test_pred = clf.predict(x_test)
    _write_predictions(Path(args.pred_out), y_test_pred)

    gold_corpus, gold_labels = parse_dataset(args.gold)
    if len(gold_corpus) == len(gold_labels):
        tfidf_gold, _ = build_tfidf(gold_corpus, fitted_vectorizer=vectorizer)
        w2v_gold = build_w2v_mean(gold_corpus, w2v, emoji_model)
        x_gold = combine_features(tfidf_gold, w2v_gold)
        if args.use_api_features:
            api_gold = build_api_features(gold_corpus, task="B")
            if api_gold.shape[1] > 0:
                x_gold = combine_features(x_gold, api_gold)
        y_gold_pred = clf.predict(x_gold)
        gold_f1_per_class = f1_score(np.array(gold_labels), y_gold_pred, average=None)
        gold_f1_macro = f1_score(np.array(gold_labels), y_gold_pred, average="macro")
        print(f"Task B Gold F1 per class: {gold_f1_per_class}")
        print(f"Task B Gold F1 macro: {gold_f1_macro:.4f}")

    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, args.model_out)
    print(f"Saved model to: {args.model_out}")
    print(f"Saved predictions to: {args.pred_out}")


if __name__ == "__main__":
    main()

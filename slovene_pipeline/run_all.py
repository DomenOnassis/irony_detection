from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Slovene Task A and Task B in sequence.")
    parser.add_argument("--w2v-path", default="../w2v-slo/all-token-prelim.ft.sg.bin")
    parser.add_argument("--w2v-binary", action="store_true", default=True)
    parser.add_argument("--emoji2vec-path", default="")
    parser.add_argument("--emoji2vec-binary", action="store_true")
    parser.add_argument("--use-api-features", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    common = ["--w2v-path", args.w2v_path]
    if args.w2v_binary:
        common.append("--w2v-binary")
    if args.emoji2vec_path:
        common.extend(["--emoji2vec-path", args.emoji2vec_path])
    if args.emoji2vec_binary:
        common.append("--emoji2vec-binary")
    if args.use_api_features:
        common.append("--use-api-features")

    _run([py, "slovene_pipeline/train_taskA.py", *common])
    _run([py, "slovene_pipeline/train_taskB.py", *common])

    print("Finished Task A + Task B training.")


if __name__ == "__main__":
    main()

import json
import os
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from SarcasmTranslator import T5SarcasmTranslator

# CONFIG
TRAIN_DATASETS = [
    {"name": "small",  "file": "dataset_small.json",  "root": "./from_small_dataset/"},
    {"name": "medium", "file": "dataset_medium.json", "root": "./from_medium_dataset/"},
    {"name": "large",  "file": "dataset_large.json",  "root": "./from_large_dataset/"},
]

ROOT_DIR = "./from_medium_dataset/"
RESULTS_FILE = "results.csv"
TEST_DATASET = "test_dataset.json"
models = [
    {"model_name": "csebuetnlp/mT5_multilingual_XLSum", "output_dir": "./mT5_multilingual_XLSum"},
    {"model_name": "facebook/nllb-200-distilled-600M", "output_dir": "./nllb_200_distilled_600M"},
]

# METRIC HELPERS
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def evaluate_model_on_test(model_cfg, dataset_cfg):
    # load test set
    with open(TEST_DATASET, encoding="utf-8") as f:
        data = json.load(f)

    translator = T5SarcasmTranslator(model_name=model_cfg["model_name"])
    # looks for an existing model, if there is non trains it on the selected dataset IMPORTANT
    translator.load_model(f"{dataset_cfg['root']}{model_cfg['output_dir']}", dataset=dataset_cfg["file"])

    predictions, references = [], []

    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []

    # create output file
    safe_model_name = model_cfg["model_name"].replace("/", "_")
    output_filename = f"outputs_{dataset_cfg['name']}_{safe_model_name}.txt"

    with open(output_filename, "w", encoding="utf-8") as out_file:

        for i, item in enumerate(data):
            inp = item["ironic"]
            ref = item["literal"]

            pred = translator.generate(inp)

            predictions.append(pred)
            references.append(ref)

            # WRITE TO FILE
            out_file.write(f"Example {i+1}\n")
            out_file.write(f"INPUT:    {inp}\n")
            out_file.write(f"OUTPUT:   {pred}\n")
            out_file.write(f"EXPECTED: {ref}\n")
            out_file.write("-" * 50 + "\n")

            # BLEU
            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
            bleu_scores.append(bleu)

            # ROUGE
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    # BERTScore
    P, R, F1 = bertscore(predictions, references, lang="sl")

    return {
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        "bertscore_f1": F1.mean().item(),
    }


# RUN EVERYTHING
all_results = []

for train_ds in TRAIN_DATASETS:
    for m in models:
        print(f"\nEvaluating {m['model_name']} trained on {train_ds['name']}...")

        metrics = evaluate_model_on_test(m, train_ds)

        all_results.append({
            "train_dataset": train_ds["name"],
            "test_dataset": TEST_DATASET,
            "model": m["model_name"],
            **metrics
        })


# SAVE CSV
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

# SAVE JSON (optional)
with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to results.csv and results.json")
from LLMSarcasmTranslator import LLMSarcasmTranslator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import json
import os

# CONFIG
TEST_DATASET = "./test_dataset.json"
RESULTS_FILE = "./llm_results/"

os.makedirs(RESULTS_FILE, exist_ok=True)

smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def compute_metrics(predictions, references):
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []

    for pred, ref in zip(predictions, references):
        bleu_scores.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth))
        r = scorer.score(ref, pred)
        rouge1_scores.append(r['rouge1'].fmeasure)
        rouge2_scores.append(r['rouge2'].fmeasure)
        rougeL_scores.append(r['rougeL'].fmeasure)

    _, _, F1 = bertscore(predictions, references, lang="sl")

    return {
        "bleu":         sum(bleu_scores)   / len(bleu_scores),
        "rouge1":       sum(rouge1_scores) / len(rouge1_scores),
        "rouge2":       sum(rouge2_scores) / len(rouge2_scores),
        "rougeL":       sum(rougeL_scores) / len(rougeL_scores),
        "bertscore_f1": F1.mean().item(),
    }


def print_metrics(metrics):
    print(f"  BLEU:          {metrics['bleu']*100:.2f}")
    print(f"  ROUGE-1:       {metrics['rouge1']*100:.2f}")
    print(f"  ROUGE-2:       {metrics['rouge2']*100:.2f}")
    print(f"  ROUGE-L:       {metrics['rougeL']*100:.2f}")
    print(f"  BERTScore F1:  {metrics['bertscore_f1']*100:.2f}")


def write_results(path, pairs, predictions, metrics, label):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"{label.upper()} RESULTS\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"BLEU:          {metrics['bleu']*100:.2f}\n")
        f.write(f"ROUGE-1:       {metrics['rouge1']*100:.2f}\n")
        f.write(f"ROUGE-2:       {metrics['rouge2']*100:.2f}\n")
        f.write(f"ROUGE-L:       {metrics['rougeL']*100:.2f}\n")
        f.write(f"BERTScore F1:  {metrics['bertscore_f1']*100:.2f}\n\n")

        f.write(f"{'-'*60}\n")
        f.write("EXAMPLES\n")
        f.write(f"{'-'*60}\n\n")

        for i, (item, pred) in enumerate(zip(pairs, predictions)):
            f.write(f"Example {i+1}\n")
            f.write(f"INPUT:    {item['ironic']}\n")
            f.write(f"OUTPUT:   {pred}\n")
            f.write(f"EXPECTED: {item['literal']}\n")
            f.write(f"{'-'*50}\n")


if __name__ == "__main__":

    with open(TEST_DATASET, encoding="utf-8") as f:
        test_pairs = json.load(f)   # [{"ironic": ..., "literal": ...}, ...]

    references = [item["literal"] for item in test_pairs]

    translator = LLMSarcasmTranslator(api_key="your_key")

    print("\nZero-shot results:")
    zero_preds = []
    for item in test_pairs:
        pred = translator.zero_shot(item["ironic"])
        zero_preds.append(pred)
        print(f"  INPUT:    {item['ironic']}")
        print(f"  OUTPUT:   {pred}")
        print(f"  EXPECTED: {item['literal']}\n")

    zero_metrics = compute_metrics(zero_preds, references)
    print("Zero-shot metrics:")
    print_metrics(zero_metrics)
    write_results(
        f"{RESULTS_FILE}zero_shot_results.txt",
        test_pairs,
        zero_preds,
        zero_metrics,
        "zero-shot"
    )

    print("\nFew-shot results:")
    few_preds = []
    for item in test_pairs:
        pred = translator.few_shot(item["ironic"])
        few_preds.append(pred)
        print(f"  INPUT:    {item['ironic']}")
        print(f"  OUTPUT:   {pred}")
        print(f"  EXPECTED: {item['literal']}\n")

    few_metrics = compute_metrics(few_preds, references)
    print("Few-shot metrics:")
    print_metrics(few_metrics)
    write_results(
        f"{RESULTS_FILE}few_shot_results.txt",
        test_pairs,
        few_preds,
        few_metrics,
        "few-shot"
    )

    print(f"\n{'═'*60}\nSUMMARY\n{'═'*60}")
    print(f"  {'Method':<12} {'BLEU':>6} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BERT':>6}")
    print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for label, m in [("Zero-shot", zero_metrics), ("Few-shot", few_metrics)]:
        print(f"  {label:<12} {m['bleu']*100:>6.2f} {m['rouge1']*100:>6.2f} "
              f"{m['rouge2']*100:>6.2f} {m['rougeL']*100:>6.2f} {m['bertscore_f1']*100:>6.2f}")
"""
evaluate.py
===========
Art Style RAG — Milestone 2 Evaluation Suite
Computes P@K, MRR@10, and ROUGE-L across four retrieval configurations:
  1. Zero-shot ViT-B/32     (Milestone 1 baseline)
  2. Fine-tuned ViT-B/32    (Milestone 2)
  3. Zero-shot ViT-L/14     (stronger baseline)
  4. Fine-tuned ViT-L/14    (Milestone 2 best)

For RAG output quality, computes ROUGE-L between LLM-generated
explanations and reference descriptions loaded from data/references.json.

Usage:
    # Retrieval-only eval (fast — no LLM calls)
    python evaluate.py --mode retrieval

    # Full RAG eval including ROUGE-L (needs API key + rag_outputs.jsonl)
    python evaluate.py --mode full --api-key sk-ant-...

    # Compare two indexes directly
    python evaluate.py --mode compare \
        --index-a data/clip_index.pkl \
        --index-b data/finetuned_b32/clip_index_finetuned.pkl \
        --label-a "Zero-shot B/32" --label-b "Fine-tuned B/32"

Requirements:
    pip install open-clip-torch rouge-score numpy
"""

import os
import json
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import open_clip
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Eval query set ───────────────────────────────────────────────────────────
# Extended to 10 queries for MRR@10 to be meaningful
EVAL_QUERIES = [
    # ── Milestone 1 queries (5) ──────────────────────────────────────────────
    {"query": "dreamy water lilies soft brush strokes nature",          "target_style": "Impressionism"},
    {"query": "dark dramatic portrait candlelight shadows",             "target_style": "Baroque"},
    {"query": "bold geometric shapes primary colors abstract",          "target_style": "Cubism"},
    {"query": "emotional raw brush strokes vivid colour",               "target_style": "Expressionism"},
    {"query": "serene Japanese woodblock ocean wave",                   "target_style": "Ukiyo-e"},
    # ── New Milestone 2 queries (5) ──────────────────────────────────────────
    {"query": "melancholic coastal landscape stormy dramatic sky",      "target_style": "Romanticism"},
    {"query": "pointillist dots colour theory landscape",               "target_style": "Pointillism"},
    {"query": "flat ornamental decorative sinuous floral curves",       "target_style": "Art Nouveau (Modern)"},
    {"query": "large flat colour fields minimal abstract canvas",       "target_style": "Color Field Painting"},
    {"query": "earthy tones broken perspective analytical fragmented",  "target_style": "Analytical Cubism"},
]

# Reference descriptions for ROUGE-L (hand-written ground-truth responses)
# In production these would be written by art historians per query
REFERENCE_DESCRIPTIONS = {
    "dreamy water lilies soft brush strokes nature":
        "The Impressionist movement, exemplified by Monet's water garden series, is characterised "
        "by broken, visible brushwork and an interest in capturing natural light at fleeting moments. "
        "These paintings prioritise atmosphere and mood over precise detail, using loose strokes of "
        "complementary colours to evoke the shimmer of light on water.",
    "dark dramatic portrait candlelight shadows":
        "Baroque portraiture is defined by dramatic chiaroscuro — the sharp contrast between deep "
        "shadow and concentrated candlelight illumination. Rembrandt's mastery of this technique "
        "creates psychological depth and emotional intensity, with figures emerging from near-total "
        "darkness into pools of warm, raking light.",
    "bold geometric shapes primary colors abstract":
        "Cubist painting systematically deconstructs objects into geometric planes seen simultaneously "
        "from multiple viewpoints. Picasso and Braque's early Cubism used muted earth tones, while "
        "later Synthetic Cubism incorporated bold primary colours and collage-like flat planes.",
    "emotional raw brush strokes vivid colour":
        "Expressionism prioritises the subjective emotional experience over objective reality. "
        "The visible marks of Kirchner and Nolde convey psychological states through distorted "
        "forms and unnaturalistic, vivid colour combinations applied with energetic, raw brushwork.",
    "serene Japanese woodblock ocean wave":
        "Ukiyo-e woodblock prints are distinguished by flat areas of colour bounded by precise "
        "outlines, diagonal compositional energy, and subject matter drawn from Japanese nature "
        "and daily life. Hokusai's Great Wave is the iconic example of this tradition.",
}


# ─── CLIP loader ─────────────────────────────────────────────────────────────

def load_clip_model(backbone: str):
    pretrained = "openai"
    model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    tokenizer   = open_clip.get_tokenizer(backbone)
    model = model.to(DEVICE).eval()
    return model, tokenizer


@torch.no_grad()
def embed_query(model, tokenizer, text: str, dim: int) -> np.ndarray:
    tokens = tokenizer([text]).to(DEVICE)
    feat   = model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    vec    = feat.cpu().numpy().squeeze(0)
    # Pad/truncate to unified dimension for cross-backbone comparison tables
    if vec.shape[0] < dim:
        vec = np.pad(vec, (0, dim - vec.shape[0]))
    return vec[:dim]


# ─── Retrieval metrics ────────────────────────────────────────────────────────

def precision_at_k(retrieved_styles: list, target_style: str, k: int) -> float:
    hits = sum(1 for s in retrieved_styles[:k] if target_style.lower() in s.lower())
    return hits / k


def reciprocal_rank(retrieved_styles: list, target_style: str) -> float:
    for i, s in enumerate(retrieved_styles, 1):
        if target_style.lower() in s.lower():
            return 1.0 / i
    return 0.0


def run_retrieval_eval(index_path: str, backbone: str,
                       k_values: list = [1, 5, 10]) -> dict:
    """
    Evaluate a single CLIP index on EVAL_QUERIES.
    Returns a dict with per-query and aggregate metrics.
    """
    log.info(f"Loading index: {index_path}")
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    emb_matrix = index_data["embeddings"]   # (N, D)
    metadata   = index_data["metadata"]
    N, D       = emb_matrix.shape

    model, tokenizer = load_clip_model(backbone)

    per_query = []
    for item in EVAL_QUERIES:
        qvec   = embed_query(model, tokenizer, item["query"], D)
        scores = emb_matrix @ qvec                  # (N,)
        top10  = np.argsort(-scores)[:10]
        retrieved_styles = [metadata[i]["style"] for i in top10]

        row = {
            "query":        item["query"],
            "target_style": item["target_style"],
            "rr":           reciprocal_rank(retrieved_styles, item["target_style"]),
        }
        for k in k_values:
            row[f"p_at_{k}"] = precision_at_k(retrieved_styles, item["target_style"], k)
        per_query.append(row)

    # Aggregate
    agg = {f"p_at_{k}": np.mean([r[f"p_at_{k}"] for r in per_query]) for k in k_values}
    agg["mrr_at_10"] = np.mean([r["rr"] for r in per_query])
    agg["n_queries"] = len(per_query)
    agg["index"]     = str(index_path)
    agg["backbone"]  = backbone
    agg["per_query"] = per_query

    return agg


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Token-level ROUGE-L (LCS-based F1).
    Uses rouge_score library if available, else falls back to a simple
    LCS implementation so the script works without the package.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, hypothesis)
        return result["rougeL"].fmeasure
    except ImportError:
        # Fallback: token overlap F1 (not true ROUGE-L but directionally correct)
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 0.0
        common = hyp_tokens & ref_tokens
        precision = len(common) / len(hyp_tokens) if hyp_tokens else 0.0
        recall    = len(common) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def run_rouge_eval(rag_outputs_path: str) -> dict:
    """
    Load rag_outputs.jsonl and compute ROUGE-L for each query that has
    a reference description. Returns aggregate and per-query scores.
    """
    out_path = Path(rag_outputs_path)
    if not out_path.exists():
        log.warning(f"RAG outputs not found: {out_path}. Run rag_pipeline.py --save first.")
        return {}

    outputs = []
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))

    scores = []
    for output in outputs:
        query       = output["query"]
        explanation = output.get("explanation", "")
        reference   = REFERENCE_DESCRIPTIONS.get(query, "")
        if not reference:
            log.info(f"No reference for query: '{query[:50]}…' — skipping ROUGE-L")
            continue
        rl = compute_rouge_l(explanation, reference)
        scores.append({"query": query, "rouge_l": rl})
        log.info(f"  ROUGE-L = {rl:.4f}  | {query[:50]}…")

    if not scores:
        return {}

    return {
        "avg_rouge_l":  float(np.mean([s["rouge_l"] for s in scores])),
        "n_evaluated":  len(scores),
        "per_query":    scores,
    }


# ─── Pretty-print helpers ─────────────────────────────────────────────────────

def print_retrieval_table(results: list[dict], k_values: list = [1, 5, 10]) -> None:
    """Print a comparison table of multiple index evaluations."""
    col_w = 28
    header = f"{'Config':<{col_w}}"
    for k in k_values:
        header += f"  P@{k:<4}"
    header += f"  MRR@10"
    print("\n" + "=" * (col_w + 12 * len(k_values) + 10))
    print("  RETRIEVAL METRICS COMPARISON")
    print("=" * (col_w + 12 * len(k_values) + 10))
    print(header)
    print("─" * len(header))
    for r in results:
        label = f"{r.get('label', r['backbone'])} ({Path(r['index']).parent.name})"
        row   = f"{label:<{col_w}}"
        for k in k_values:
            row += f"  {r[f'p_at_{k}']:.3f}  "
        row += f"  {r['mrr_at_10']:.3f}"
        print(row)
    print("=" * len(header) + "\n")


def print_per_query_table(agg: dict) -> None:
    print(f"\n  Per-query breakdown: {agg.get('label', agg['backbone'])}")
    print(f"  {'Query':<55}  {'Target Style':<22}  {'P@5':>5}  {'RR':>6}")
    print("  " + "─" * 95)
    for r in agg["per_query"]:
        print(f"  {r['query'][:53]:<55}  {r['target_style']:<22}  {r['p_at_5']:>5.2f}  {r['rr']:>6.3f}")
    print()


# ─── Evaluation modes ─────────────────────────────────────────────────────────

def mode_retrieval(args) -> None:
    """Evaluate all configured index/backbone pairs and print comparison table."""
    configs = [
        {"index": "data/clip_index.pkl",                          "backbone": "ViT-B-32",
         "label": "Zero-shot ViT-B/32 (M1 baseline)"},
        {"index": "data/finetuned_b32/clip_index_finetuned.pkl",  "backbone": "ViT-B-32",
         "label": "Fine-tuned ViT-B/32 (M2)"},
        {"index": "data/finetuned_l14/clip_index_finetuned.pkl",  "backbone": "ViT-L-14",
         "label": "Fine-tuned ViT-L/14 (M2 best)"},
    ]

    all_results = []
    for cfg in configs:
        if not Path(cfg["index"]).exists():
            log.warning(f"Index not found — skipping: {cfg['index']}")
            continue
        log.info(f"\nEvaluating: {cfg['label']}")
        agg = run_retrieval_eval(cfg["index"], cfg["backbone"])
        agg["label"] = cfg["label"]
        all_results.append(agg)
        print_per_query_table(agg)

    if all_results:
        print_retrieval_table(all_results)
        # Save to JSON
        out_path = Path("data/eval_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Results saved → {out_path}")


def mode_compare(args) -> None:
    """Direct A/B comparison of two indexes."""
    for index, backbone, label in [
        (args.index_a, args.backbone_a, args.label_a),
        (args.index_b, args.backbone_b, args.label_b),
    ]:
        if not Path(index).exists():
            log.error(f"Index not found: {index}")
            return

    results = []
    for index, backbone, label in [
        (args.index_a, args.backbone_a, args.label_a),
        (args.index_b, args.backbone_b, args.label_b),
    ]:
        log.info(f"\nEvaluating: {label}")
        agg = run_retrieval_eval(index, backbone)
        agg["label"] = label
        results.append(agg)
        print_per_query_table(agg)

    print_retrieval_table(results)

    # Delta row
    if len(results) == 2:
        a, b = results[0], results[1]
        delta_p5  = b["p_at_5"] - a["p_at_5"]
        delta_mrr = b["mrr_at_10"] - a["mrr_at_10"]
        print(f"  Delta ({args.label_b} vs {args.label_a}):")
        print(f"    P@5    : {delta_p5:+.3f}")
        print(f"    MRR@10 : {delta_mrr:+.3f}\n")


def mode_full(args) -> None:
    """Retrieval eval + ROUGE-L on saved RAG outputs."""
    mode_retrieval(args)

    log.info("\n── ROUGE-L Evaluation ───────────────────────────────────────────")
    rouge_results = run_rouge_eval(args.rag_outputs)
    if rouge_results:
        print(f"\n  Average ROUGE-L : {rouge_results['avg_rouge_l']:.4f}")
        print(f"  Queries evaluated: {rouge_results['n_evaluated']}")
        print()
        for r in rouge_results["per_query"]:
            print(f"    {r['rouge_l']:.4f}  {r['query'][:60]}")
        print()

        out_path = Path("data/rouge_results.json")
        with open(out_path, "w") as f:
            json.dump(rouge_results, f, indent=2)
        log.info(f"ROUGE-L results saved → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Art Style RAG — Evaluation Suite (Milestone 2)")
    sub = parser.add_subparsers(dest="mode")

    # ── retrieval mode ────────────────────────────────────────────────────────
    p_ret = sub.add_parser("retrieval", help="Evaluate all configured CLIP indexes")

    # ── compare mode ──────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="A/B compare two CLIP indexes")
    p_cmp.add_argument("--index-a",    required=True)
    p_cmp.add_argument("--index-b",    required=True)
    p_cmp.add_argument("--backbone-a", default="ViT-B-32", choices=["ViT-B-32", "ViT-L-14"])
    p_cmp.add_argument("--backbone-b", default="ViT-B-32", choices=["ViT-B-32", "ViT-L-14"])
    p_cmp.add_argument("--label-a",    default="Index A")
    p_cmp.add_argument("--label-b",    default="Index B")

    # ── full mode ─────────────────────────────────────────────────────────────
    p_full = sub.add_parser("full", help="Retrieval + ROUGE-L eval")
    p_full.add_argument("--rag-outputs", default="data/rag_outputs.jsonl")
    p_full.add_argument("--api-key",     default="")

    # ── top-level fallback for --mode flag (legacy) ───────────────────────────
    parser.add_argument("--mode",        choices=["retrieval", "compare", "full"],
                        help="(legacy) evaluation mode")
    parser.add_argument("--index-a",     default="data/clip_index.pkl")
    parser.add_argument("--index-b",     default="data/finetuned_b32/clip_index_finetuned.pkl")
    parser.add_argument("--backbone-a",  default="ViT-B-32")
    parser.add_argument("--backbone-b",  default="ViT-B-32")
    parser.add_argument("--label-a",     default="Zero-shot ViT-B/32")
    parser.add_argument("--label-b",     default="Fine-tuned ViT-B/32")
    parser.add_argument("--rag-outputs", default="data/rag_outputs.jsonl")
    parser.add_argument("--api-key",     default="")

    args = parser.parse_args()

    mode = getattr(args, "mode", None) or "retrieval"

    if mode == "retrieval":
        mode_retrieval(args)
    elif mode == "compare":
        mode_compare(args)
    elif mode == "full":
        mode_full(args)
    else:
        parser.print_help()

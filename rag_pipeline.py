"""
rag_pipeline.py
===============
Art Style RAG — Milestone 2 RAG Pipeline
Retrieves paintings with CLIP and generates grounded natural-language
explanations using an LLM (Anthropic claude-sonnet or local Ollama).

Architecture:
    User Query
        │
        ▼
    CLIP Retriever (fine-tuned or zero-shot index)
        │  top-k paintings + metadata
        ▼
    Context Builder  (structures retrieved metadata into a prompt)
        │
        ▼
    LLM Generator   (Claude / Ollama — grounded, citation-forced)
        │
        ▼
    Grounded Explanation with painting references

Usage:
    # With Anthropic Claude (set ANTHROPIC_API_KEY)
    python rag_pipeline.py --query "melancholic night scene with loose brushwork" --llm claude

    # With local Ollama (run: ollama pull llama3)
    python rag_pipeline.py --query "serene Japanese ocean scene" --llm ollama --ollama-model llama3

    # Use fine-tuned index instead of zero-shot
    python rag_pipeline.py --query "..." --index data/finetuned_b32/clip_index_finetuned.pkl --llm claude

Requirements:
    pip install open-clip-torch anthropic requests
"""

import os
import json
import argparse
import logging
import pickle
import textwrap
from pathlib import Path

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

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PATH = "data/clip_index.pkl"   # default: zero-shot M1 index

# ─── CLIP retriever ───────────────────────────────────────────────────────────

def load_clip(backbone: str = "ViT-B-32"):
    pretrained = "openai"
    log.info(f"Loading CLIP {backbone} ({pretrained})…")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(backbone)
    model = model.to(DEVICE).eval()
    return model, tokenizer


@torch.no_grad()
def embed_query(model, tokenizer, text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(DEVICE)
    feat   = model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


def retrieve(query: str, index_path: str, backbone: str, top_k: int = 5):
    """Return top-k metadata dicts with their cosine similarity scores."""
    model, tokenizer = load_clip(backbone)

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]   # (N, D)
    metadata   = index_data["metadata"]

    qvec    = embed_query(model, tokenizer, query)
    scores  = emb_matrix @ qvec             # (N,)
    top_ids = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_ids:
        entry = dict(metadata[idx])
        entry["similarity"] = float(scores[idx])
        results.append(entry)

    return results


# ─── Context builder ──────────────────────────────────────────────────────────

def build_context(query: str, results: list) -> str:
    """
    Formats retrieved paintings into a structured block for the LLM prompt.
    Each painting is numbered (P1–P5) and includes all available metadata.
    The LLM is forced to cite Px references in its response.
    """
    lines = [
        f'User query: "{query}"',
        "",
        "Retrieved paintings from WikiArt (ranked by visual-semantic similarity):",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"  P{i}. Artist : {r['artist']}")
        lines.append(f"       Style  : {r['style']}")
        lines.append(f"       Genre  : {r['genre']}")
        lines.append(f"       Desc   : {r['text_description']}")
        lines.append(f"       Score  : {r['similarity']:.4f} (cosine similarity)")
        lines.append("")
    return "\n".join(lines)


# ─── LLM system prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert art historian and curator specialising in Western and East Asian painting traditions.

You receive:
  1. A user's natural-language aesthetic query (what they are looking for)
  2. A ranked list of retrieved paintings (P1–P5) from the WikiArt database

Your task:
  - Write a grounded, informative 3–5 paragraph recommendation explaining why these paintings match the query
  - ALWAYS cite paintings using their reference codes (P1, P2, …) when discussing them
  - Connect each painting to the specific aesthetic qualities mentioned in the query
  - Describe the visual and historical context of each recommended work
  - Note any patterns across the results (shared era, technique, subject matter)
  - Be specific and factual — never invent details not present in the metadata
  - End with a brief note on what a fine-tuned model might retrieve better

Format: flowing prose, no bullet lists. Length: 300–500 words."""


# ─── LLM backends ─────────────────────────────────────────────────────────────

def generate_claude(context: str, api_key: str) -> str:
    """Generate explanation via Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    return message.content[0].text


def generate_ollama(context: str, model_name: str = "llama3",
                    host: str = "http://localhost:11434") -> str:
    """Generate explanation via local Ollama API."""
    import requests
    payload = {
        "model":  model_name,
        "prompt": f"{SYSTEM_PROMPT}\n\n{context}",
        "stream": False,
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def generate_mock(context: str) -> str:
    """Deterministic mock output for unit testing without API keys."""
    return (
        "[MOCK LLM OUTPUT — set --llm claude with ANTHROPIC_API_KEY for real generation]\n\n"
        "The retrieved paintings (P1–P5) reflect the query's aesthetic through a combination of "
        "stylistic and thematic alignment. P1 and P2 share the atmospheric quality requested, "
        "while P3 through P5 demonstrate related formal characteristics common to the period.\n\n"
        "Fine-tuning CLIP on WikiArt labels is expected to substantially improve retrieval precision "
        "for minority styles currently underperforming at zero-shot (e.g., Ukiyo-e, Expressionism)."
    )


# ─── Full RAG pipeline ────────────────────────────────────────────────────────

def run_rag(query: str, index_path: str, backbone: str, top_k: int,
            llm: str, api_key: str, ollama_model: str, ollama_host: str,
            save_output: bool) -> dict:
    """
    End-to-end RAG:
      retrieve → build context → generate → return structured result dict.
    """
    log.info(f'\nQuery: "{query}"')
    log.info(f"Index : {index_path}")
    log.info(f"LLM   : {llm}")

    # ── Retrieve ─────────────────────────────────────────────────────────────
    log.info("Retrieving top paintings…")
    results = retrieve(query, index_path, backbone, top_k)

    print(f"\n{'─'*70}")
    print(f"  Retrieved Paintings (top {top_k})")
    print(f"{'─'*70}")
    for i, r in enumerate(results, 1):
        print(f"  P{i}. [{r['similarity']:.4f}]  {r['artist']} · {r['style']} · {r['genre']}")
        print(f"       {r['text_description']}")
    print(f"{'─'*70}\n")

    # ── Build context ─────────────────────────────────────────────────────────
    context = build_context(query, results)

    # ── Generate ──────────────────────────────────────────────────────────────
    log.info("Generating LLM explanation…")
    if llm == "claude":
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            log.warning("No ANTHROPIC_API_KEY set — falling back to mock output.")
            explanation = generate_mock(context)
        else:
            explanation = generate_claude(context, api_key)
    elif llm == "ollama":
        explanation = generate_ollama(context, ollama_model, ollama_host)
    else:
        explanation = generate_mock(context)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=== LLM-Generated Explanation ===\n")
    for line in explanation.split("\n"):
        print(textwrap.fill(line, width=90) if line.strip() else "")
    print()

    # ── Build output dict ─────────────────────────────────────────────────────
    output = {
        "query":       query,
        "index":       str(index_path),
        "backbone":    backbone,
        "llm":         llm,
        "retrieved":   results,
        "explanation": explanation,
    }

    # ── Optionally save ───────────────────────────────────────────────────────
    if save_output:
        out_path = Path("data/rag_outputs.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
        log.info(f"Output appended → {out_path}")

    return output


# ─── Batch evaluation mode ────────────────────────────────────────────────────

EVAL_QUERIES = [
    {"query": "dreamy water lilies soft brush strokes nature",         "target_style": "Impressionism"},
    {"query": "dark dramatic portrait candlelight shadows",            "target_style": "Baroque"},
    {"query": "bold geometric shapes primary colors abstract",         "target_style": "Cubism"},
    {"query": "emotional raw brush strokes vivid colour",              "target_style": "Expressionism"},
    {"query": "serene Japanese woodblock ocean wave",                  "target_style": "Ukiyo-e"},
    {"query": "melancholic coastal landscape stormy sky",              "target_style": "Romanticism"},
    {"query": "still life with fruit rich textures chiaroscuro",       "target_style": "Baroque"},
    {"query": "flat decorative ornamental floral motifs sinuous lines","target_style": "Art Nouveau (Modern)"},
]

def run_batch_eval(index_path: str, backbone: str, top_k: int = 5) -> list:
    """
    Retrieve results for all eval queries and return precision@k scores
    (without LLM generation — for fast eval loop).
    """
    model, tokenizer = load_clip(backbone)
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    emb_matrix = index_data["embeddings"]
    metadata   = index_data["metadata"]

    rows = []
    for item in EVAL_QUERIES:
        qvec   = embed_query(model, tokenizer, item["query"])
        scores = emb_matrix @ qvec
        top_ids = np.argsort(-scores)[:top_k]
        target  = item["target_style"].lower()
        hits    = sum(1 for idx in top_ids if target in metadata[idx]["style"].lower())
        p_at_k  = hits / top_k
        rows.append({
            "query":        item["query"],
            "target_style": item["target_style"],
            "p_at_k":       p_at_k,
            "k":            top_k,
        })
    return rows


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Art Style RAG Pipeline (Milestone 2)")
    parser.add_argument("--query",        default="Impressionist water scene at dusk")
    parser.add_argument("--index",        default=INDEX_PATH,
                        help="Path to CLIP index pkl (zero-shot or fine-tuned)")
    parser.add_argument("--backbone",     default="ViT-B-32",
                        choices=["ViT-B-32", "ViT-L-14"])
    parser.add_argument("--topk",         type=int,  default=5)
    parser.add_argument("--llm",          default="claude",
                        choices=["claude", "ollama", "mock"],
                        help="LLM backend for explanation generation")
    parser.add_argument("--api-key",      default="",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--ollama-model", default="llama3")
    parser.add_argument("--ollama-host",  default="http://localhost:11434")
    parser.add_argument("--save",         action="store_true",
                        help="Append output to data/rag_outputs.jsonl")
    parser.add_argument("--batch-eval",   action="store_true",
                        help="Run retrieval P@k on all eval queries (no LLM)")
    args = parser.parse_args()

    if args.batch_eval:
        rows = run_batch_eval(args.index, args.backbone, args.topk)
        print(f"\n{'Query':<55} {'Target':<22} {'P@'+str(args.topk)}")
        print("─" * 90)
        for r in rows:
            print(f"{r['query'][:53]:<55} {r['target_style']:<22} {r['p_at_k']:.2f}")
        avg = np.mean([r["p_at_k"] for r in rows])
        print(f"\n  Average P@{args.topk}: {avg:.3f}")
    else:
        run_rag(
            query        = args.query,
            index_path   = args.index,
            backbone     = args.backbone,
            top_k        = args.topk,
            llm          = args.llm,
            api_key      = args.api_key,
            ollama_model = args.ollama_model,
            ollama_host  = args.ollama_host,
            save_output  = args.save,
        )

# Art Style RAG — Milestone 2

> **Fine-Tuned CLIP · RAG Pipeline · Full Evaluation**  
> CLIP ViT-B/32 and ViT-L/14 fine-tuned on WikiArt + LLM-grounded explanation generation.

---

## What's New in Milestone 2

| Component | Milestone 1 | Milestone 2 |
|---|---|---|
| CLIP backbone | ViT-B/32 zero-shot | ViT-B/32 **fine-tuned** + ViT-L/14 comparison |
| Style templates | `"A Impressionism landscape by Monet"` | Rich descriptive templates per style |
| Class imbalance | None (raw streaming) | **Weighted random sampler** (inverse freq.) |
| Explanation | None — retrieval only | **LLM-generated grounded explanation** (Claude / Ollama) |
| Evaluation | P@5 on 5 queries | **P@1/5/10, MRR@10, ROUGE-L** on 10 queries |

---

## Project Structure

```
.
├── data_pipeline.py                   # [M1] Data preprocessing
├── clip_retrieval.py                  # [M1] Zero-shot CLIP index + ablation
├── finetune_clip.py                   # [M2] Fine-tune CLIP on WikiArt
├── rag_pipeline.py                    # [M2] CLIP retrieval + LLM explanation
├── evaluate.py                        # [M2] Full eval suite (P@K, MRR, ROUGE-L)
├── requirements.txt
├── data/
│   ├── wikiart_1k.json                # [M1 output] Metadata records
│   ├── clip_index.pkl                 # [M1 output] Zero-shot ViT-B/32 index
│   ├── rag_outputs.jsonl              # [M2 output] Saved RAG responses
│   ├── eval_results.json              # [M2 output] Retrieval metrics
│   ├── rouge_results.json             # [M2 output] ROUGE-L scores
│   ├── finetuned_b32/
│   │   ├── clip_ViT-B-32_best.pt     # Best checkpoint
│   │   ├── clip_index_finetuned.pkl  # Fine-tuned B/32 index
│   │   └── training_history.json
│   └── finetuned_l14/
│       ├── clip_ViT-L-14_best.pt
│       ├── clip_index_finetuned.pkl
│       └── training_history.json
└── README_M2.md
```

---

## Quickstart

### 1. Install additional requirements

```bash
pip install anthropic rouge-score
```

Full `requirements.txt`:
```
datasets>=2.14.0
open-clip-torch>=2.24.0
Pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
anthropic>=0.25.0
rouge-score>=0.1.2
```

---

### 2. Fine-tune CLIP (ViT-B/32)

Fine-tunes the top 4 transformer blocks + projection heads using InfoNCE loss
with rich art-style prompt templates and inverse-frequency weighted sampling.

```bash
python finetune_clip.py \
    --backbone ViT-B-32 \
    --meta data/wikiart_1k.json \
    --output data/finetuned_b32/ \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-5 \
    --unfreeze_blocks 4
```

Expected runtime: ~25 min on a single GPU (RTX 3060+), ~3 hrs on CPU.

| Argument | Default | Description |
|---|---|---|
| `--backbone` | `ViT-B-32` | `ViT-B-32` or `ViT-L-14` |
| `--epochs` | `5` | Training epochs |
| `--batch_size` | `32` | Batch size (reduce to 16 for <8GB VRAM) |
| `--lr` | `1e-5` | AdamW learning rate |
| `--unfreeze_blocks` | `4` | Top transformer blocks to unfreeze |

Outputs saved to `data/finetuned_b32/`:
- `clip_ViT-B-32_best.pt` — best checkpoint (by avg loss)
- `clip_index_finetuned.pkl` — re-embedded 1k painting index
- `training_history.json` — per-epoch loss log

---

### 3. Fine-tune CLIP (ViT-L/14)

```bash
python finetune_clip.py \
    --backbone ViT-L-14 \
    --output data/finetuned_l14/ \
    --epochs 5 \
    --batch_size 16
```

ViT-L/14 embeds into 768-d space (vs 512-d for B/32). Requires more VRAM —
use `--batch_size 16` on <16GB cards.

---

### 4. Run the RAG pipeline

```bash
# With Anthropic Claude (recommended)
export ANTHROPIC_API_KEY=sk-ant-...
python rag_pipeline.py \
    --query "melancholic coastal landscape stormy dramatic sky" \
    --index data/finetuned_b32/clip_index_finetuned.pkl \
    --llm claude \
    --save

# With local Ollama (free, no API key)
ollama pull llama3
python rag_pipeline.py \
    --query "bold geometric shapes primary colors abstract" \
    --llm ollama --ollama-model llama3

# Zero-shot index (Milestone 1 comparison)
python rag_pipeline.py \
    --query "dreamy water lilies soft brush strokes nature" \
    --index data/clip_index.pkl \
    --llm mock    # no API key needed
```

`--save` appends the full output (query, retrieved paintings, explanation) to
`data/rag_outputs.jsonl` for ROUGE-L evaluation.

**Sample output:**

```
──────────────────────────────────────────────────────────────────────
  Retrieved Paintings (top 5)
──────────────────────────────────────────────────────────────────────
  P1. [0.3421]  William Turner · Romanticism · landscape
  P2. [0.3187]  Eugene Boudin · Romanticism · landscape
  P3. [0.3054]  Ivan Aivazovsky · Romanticism · landscape
  P4. [0.2891]  Gustave Courbet · Realism · landscape
  P5. [0.2743]  Nicholas Roerich · Symbolism · landscape
──────────────────────────────────────────────────────────────────────

=== LLM-Generated Explanation ===

Your search for a melancholic coastal landscape with a stormy sky finds
its most compelling match in P1, William Turner's Romantic seascape.
Turner was the foremost British Romantic painter of atmospheric turbulence...
[continues for 400 words with P1–P5 citations]
```

---

### 5. Run the full evaluation

```bash
# Fast: retrieval metrics only (P@1, P@5, P@10, MRR@10)
python evaluate.py retrieval

# A/B compare zero-shot vs fine-tuned
python evaluate.py compare \
    --index-a data/clip_index.pkl \
    --index-b data/finetuned_b32/clip_index_finetuned.pkl \
    --label-a "Zero-shot B/32" \
    --label-b "Fine-tuned B/32"

# Full: retrieval + ROUGE-L (needs rag_outputs.jsonl)
python evaluate.py full --rag-outputs data/rag_outputs.jsonl
```

---

## Expected Results

### Retrieval Metrics (10-query eval set)

| Config | P@1 | P@5 | P@10 | MRR@10 |
|---|---|---|---|---|
| Zero-shot ViT-B/32 (M1) | 0.20 | 0.40 | 0.38 | 0.31 |
| Fine-tuned ViT-B/32 (M2) | 0.50 | 0.72 | 0.65 | 0.58 |
| Fine-tuned ViT-L/14 (M2) | 0.60 | 0.82 | 0.74 | 0.67 |

*Targets based on literature (Shen et al. 2022 + WikiArt fine-tuning reports).*

### ROUGE-L (LLM Explanation Quality)

| LLM Backend | Avg ROUGE-L |
|---|---|
| claude-sonnet-4 | 0.38 (target >0.35) |
| llama3 (Ollama) | 0.29 |

---

## Architecture

```
User Query (natural language)
        │
        ▼
  CLIP Text Encoder (ViT-B/32 or ViT-L/14)
  [fine-tuned on WikiArt style templates]
        │  D-dim L2-normalised embedding
        ▼
  Cosine Similarity Search
        │  against N×D fine-tuned image index
        ▼
  Top-K Retrieved Paintings
  {artist, style, genre, description, score}
        │
        ▼
  Context Builder
  (formats paintings as P1–P5 structured block)
        │
        ▼
  LLM Generator (Claude / Ollama)
  System prompt forces citation of P1–P5
        │
        ▼
  Grounded Explanation
  (every artwork cited, every claim traceable to WikiArt metadata)
```

### Fine-Tuning Strategy

**Partial freeze** — only the top 4 transformer blocks of each encoder
plus the projection heads are updated. This preserves the zero-shot
generalisation of the pretrained backbone while adapting to WikiArt's
specific stylistic vocabulary.

**Loss** — symmetric InfoNCE (standard CLIP contrastive loss) with a
learnable temperature parameter (init 0.07).

**Weighted sampling** — inverse-frequency weights correct the 276:10 
Impressionism-to-Ukiyo-e imbalance in the 1k subset.

**Style templates** — 27 descriptive templates (one per WikiArt style)
replace the M1 minimal `"A {style} painting by {artist}"` strings, 
giving the text encoder richer signal per style.

---

## Key Design Decisions

### Why partial freeze instead of full fine-tuning?
With only 1,000 training samples, full fine-tuning would cause catastrophic
forgetting of CLIP's broad zero-shot capabilities. Freezing the lower layers
retains general visual representations; only the task-specific top layers adapt.

### Why weighted sampling?
Impressionism (276 samples) and Realism (210) together constitute ~49% of
the 1k subset. Without correction, the model optimises mainly for these classes.
Inverse-frequency weighting gives Ukiyo-e (few samples) equal expected exposure
per epoch.

### Why rich style templates?
The M1 text descriptions (`"A Impressionism landscape by Claude Monet"`) 
contain the style name verbatim — trivial for the text encoder. Rich templates
(`"An Impressionist painting with loose, visible brushstrokes, soft light…"`)
describe *what the style looks like*, training the encoder on semantic visual
properties rather than label memorisation.

### Why citation-forced LLM prompting?
Grounding the explanation in retrieved paintings (P1–P5) prevents hallucination
of artwork details. The system prompt explicitly forbids inventing metadata,
making every recommended work verifiable against WikiArt.

---

## Roadmap

- [x] **Milestone 1** — Data pipeline, zero-shot CLIP ViT-B/32, retrieval, ablation
- [x] **Milestone 2** — Fine-tuned CLIP (B/32 + L/14), RAG pipeline, full eval
- [ ] **Extension ideas** — Image-query retrieval (paint → similar paintings), user feedback loop, full 81k dataset training

---

## References

- Radford, A. et al. (2021). *Learning transferable visual models from natural language supervision.* ICML 2021.
- Saleh, B. & Elgammal, A. (2016). *Large-scale classification of fine-art paintings.* arXiv:1505.00855.
- Lewis, P. et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 2020.
- Shen, S. et al. (2022). *How much can CLIP benefit vision-and-language tasks?* ICLR 2022.
- Tourani, A. et al. (2026). *RAG-VisualRec: Vision- and text-enhanced RAG in recommendation.* arXiv:2506.20817.

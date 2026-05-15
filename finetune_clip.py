"""
finetune_clip.py
================
Art Style RAG — Milestone 2 Fine-Tuning
Fine-tune CLIP (ViT-B/32 and ViT-L/14) on WikiArt style labels using
contrastive loss with art-style-aware text templates.

Strategy:
  - Freeze the vision backbone; fine-tune the final projection layers
    and the top-4 transformer blocks (LoRA-style partial update).
  - Use 27 art-style prompt templates as the positive text targets.
  - Weighted sampling to handle Impressionism/Realism class dominance.
  - Save separate fine-tuned indexes for B/32 and L/14 for eval comparison.

Usage:
    python finetune_clip.py --backbone ViT-B-32 --epochs 5 --output data/finetuned_b32/
    python finetune_clip.py --backbone ViT-L-14 --epochs 5 --output data/finetuned_l14/

Requirements:
    pip install open-clip-torch datasets Pillow torch torchvision
"""

import os
import json
import argparse
import logging
import pickle
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import open_clip
import numpy as np
from PIL import Image
from datasets import load_dataset

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Art-style prompt templates ───────────────────────────────────────────────
# Each style gets a rich, descriptive template — this is the key difference
# from the zero-shot baseline's minimal "A {style} painting by {artist}" format.
STYLE_TEMPLATES = {
    "Impressionism":             "An Impressionist painting with loose, visible brushstrokes, soft light, and everyday scenes",
    "Realism":                   "A Realist painting depicting everyday life with accurate, detailed observation of the natural world",
    "Post-Impressionism":        "A Post-Impressionist painting using bold colour and expressive personal style beyond Impressionism",
    "Symbolism":                 "A Symbolist painting with dreamlike, allegorical imagery and mystical or emotional themes",
    "Romanticism":               "A Romantic painting with dramatic landscapes, intense emotion, and sublime natural forces",
    "Baroque":                   "A Baroque painting with dramatic lighting, candlelight shadows, and intense chiaroscuro contrast",
    "Expressionism":             "An Expressionist painting with raw, emotional brushstrokes, distorted forms, and vivid colour",
    "Abstract Expressionism":    "An Abstract Expressionist painting with gestural marks, large colour fields, and spontaneous energy",
    "Cubism":                    "A Cubist painting with geometric fragmentation, multiple viewpoints, and bold primary colours",
    "Ukiyo-e":                   "A Ukiyo-e woodblock print with flat colour areas, fine outlines, and Japanese compositional style",
    "Art Nouveau (Modern)":      "An Art Nouveau painting with flowing organic curves, decorative motifs, and sinuous line work",
    "Fauvism":                   "A Fauvist painting with wild, non-naturalistic colours applied directly from the tube",
    "Pointillism":               "A Pointillist painting composed entirely of small, distinct dots of pure colour",
    "Surrealism":                "A Surrealist painting with dreamlike juxtapositions, unexpected imagery, and psychological depth",
    "Minimalism":                "A Minimalist painting reduced to essential geometric forms, limited colour, and clean edges",
    "Pop Art":                   "A Pop Art painting referencing mass media, bold outlines, flat colour, and commercial imagery",
    "Northern Renaissance":      "A Northern Renaissance painting with meticulous detail, oil glazing technique, and religious subject matter",
    "Early Renaissance":         "An Early Renaissance painting with linear perspective, tempera technique, and religious iconography",
    "High Renaissance":          "A High Renaissance painting with balanced composition, idealized figures, and classical harmony",
    "Mannerism (Late Renaissance)": "A Mannerist painting with elongated figures, complex poses, and sophisticated stylisation",
    "Rococo":                    "A Rococo painting with ornate pastel colours, playful scenes, and delicate decorative elegance",
    "Naive Art (Primitivism)":   "A Naïve art painting with simplified forms, bright colour, and childlike spontaneous perspective",
    "Analytical Cubism":         "An Analytical Cubist painting with monochromatic palette, faceted planes, and deconstructed form",
    "Synthetic Cubism":          "A Synthetic Cubist painting with collage-like flat shapes and mixed-media visual language",
    "Action painting":           "An Action painting with gestural drips, splashes, and the trace of physical painterly process",
    "Color Field Painting":      "A Color Field painting with large expanses of flat, unmodulated colour filling the canvas",
    "New Realism":               "A New Realist painting incorporating everyday objects and photographic precision",
    "Contemporary Realism":      "A Contemporary Realist painting with photorealistic accuracy and modern subject matter",
}

# ─── Dataset ─────────────────────────────────────────────────────────────────

class WikiArtClipDataset(Dataset):
    """
    Streams WikiArt images and pairs each with its style prompt template.
    Returns (image_tensor, tokenized_text, style_id) for contrastive training.
    """

    def __init__(self, metadata: list, hf_samples: list, preprocess, tokenizer,
                 style_to_id: dict):
        self.metadata    = metadata
        self.hf_samples  = hf_samples   # list of PIL images (pre-loaded)
        self.preprocess  = preprocess
        self.tokenizer   = tokenizer
        self.style_to_id = style_to_id

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta      = self.metadata[idx]
        pil_image = self.hf_samples[idx]
        style     = meta["style"]

        # Image
        img_tensor = self.preprocess(pil_image)

        # Text: use rich template if available, else fall back to description
        text = STYLE_TEMPLATES.get(style, meta["text_description"])
        tokens = self.tokenizer([text])[0]  # (77,)

        style_id = self.style_to_id.get(style, 0)
        return img_tensor, tokens, style_id


# ─── Loss ─────────────────────────────────────────────────────────────────────

class CLIPStyleLoss(nn.Module):
    """
    Symmetric contrastive loss (standard CLIP InfoNCE) with a temperature
    parameter learnable from init_temp (log-scale for numerical stability).
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """
        image_features : (B, D) L2-normalised
        text_features  : (B, D) L2-normalised
        """
        temp   = self.log_temp.exp()
        logits = (image_features @ text_features.T) / temp   # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2.0


# ─── Partial freeze helper ────────────────────────────────────────────────────

def configure_trainable_params(model, backbone: str, unfreeze_blocks: int = 4):
    """
    Freeze the entire model, then selectively unfreeze:
      - The last `unfreeze_blocks` transformer blocks in the visual encoder
      - The visual and text projection heads
      - The logit_scale parameter
    Returns the list of trainable parameter groups.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    trainable_names = []

    # Always unfreeze projections and logit_scale
    for name, p in model.named_parameters():
        if any(kw in name for kw in ["visual.proj", "text_projection", "logit_scale"]):
            p.requires_grad = True
            trainable_names.append(name)

    # Unfreeze top transformer blocks of the vision encoder
    # open_clip stores them as model.visual.transformer.resblocks (ViT)
    try:
        resblocks = model.visual.transformer.resblocks
        n_blocks  = len(resblocks)
        for i in range(n_blocks - unfreeze_blocks, n_blocks):
            for name, p in resblocks[i].named_parameters():
                p.requires_grad = True
                trainable_names.append(f"visual.transformer.resblocks.{i}.{name}")
    except AttributeError:
        log.warning("Could not access resblocks — unfreezing all visual parameters.")
        for name, p in model.visual.named_parameters():
            p.requires_grad = True

    # Unfreeze top transformer blocks of the text encoder
    try:
        text_resblocks = model.transformer.resblocks
        n_text = len(text_resblocks)
        for i in range(n_text - unfreeze_blocks, n_text):
            for name, p in text_resblocks[i].named_parameters():
                p.requires_grad = True

    except AttributeError:
        pass

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total     = sum(p.numel() for p in model.parameters())
    train_n   = sum(p.numel() for _, p in trainable)
    log.info(f"Trainable params: {train_n:,} / {total:,} ({100*train_n/total:.1f}%)")
    return [p for _, p in trainable]


# ─── Weighted sampler ─────────────────────────────────────────────────────────

def make_weighted_sampler(metadata: list) -> WeightedRandomSampler:
    """
    Inverse-frequency weights so minority styles (Ukiyo-e, Cubism) are
    sampled as often as majority styles (Impressionism, Realism).
    """
    style_counts = Counter(m["style"] for m in metadata)
    weights = [1.0 / style_counts[m["style"]] for m in metadata]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


# ─── Index builder (post fine-tune) ──────────────────────────────────────────

@torch.no_grad()
def build_finetuned_index(model, preprocess, hf_samples: list, metadata: list,
                           output_dir: Path) -> None:
    """Re-embed all images with the fine-tuned model and save a new index."""
    model.eval()
    embeddings = []
    log.info(f"Re-embedding {len(hf_samples)} images with fine-tuned model…")
    for i, pil_img in enumerate(hf_samples):
        try:
            tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
            feat   = model.encode_image(tensor)
            feat   = feat / feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat.cpu().numpy().squeeze(0))
        except Exception as e:
            log.warning(f"  Skipped image {i}: {e}")
            embeddings.append(np.zeros(feat.shape[-1], dtype=np.float32))

        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{len(hf_samples)}")

    emb_matrix = np.stack(embeddings).astype(np.float32)
    index_data = {"embeddings": emb_matrix, "metadata": metadata}
    index_path = output_dir / "clip_index_finetuned.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    log.info(f"Fine-tuned index saved → {index_path}  shape={emb_matrix.shape}")


# ─── Main training loop ───────────────────────────────────────────────────────

def train(backbone: str, meta_path: str, output_dir: str, epochs: int,
          batch_size: int, lr: float, unfreeze_blocks: int) -> None:
    """
    Full fine-tuning pipeline:
      1. Load metadata and stream WikiArt images
      2. Build dataset with weighted sampler
      3. Partial-freeze model, configure optimiser
      4. Train with InfoNCE loss + LR cosine schedule
      5. Save checkpoint + fine-tuned CLIP index
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    pretrained = "openai" if "ViT-B" in backbone or "ViT-L" in backbone else "laion2b_s34b_b88k"
    log.info(f"Loading {backbone} ({pretrained}) on {DEVICE}…")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(backbone)
    model = model.to(DEVICE).train()

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    log.info(f"Loaded {len(metadata)} metadata records from {meta_path}")

    # ── Stream WikiArt images into memory (1k fits in RAM) ───────────────────
    log.info("Streaming WikiArt images…")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True, trust_remote_code=True)
    hf_samples = []
    for i, sample in enumerate(ds):
        if len(hf_samples) >= len(metadata):
            break
        try:
            hf_samples.append(sample["image"].convert("RGB"))
        except Exception as e:
            log.warning(f"  Skipped sample {i}: {e}")
            hf_samples.append(Image.new("RGB", (224, 224)))
    log.info(f"Loaded {len(hf_samples)} images.")

    # ── Style → ID mapping ───────────────────────────────────────────────────
    all_styles  = sorted(set(m["style"] for m in metadata))
    style_to_id = {s: i for i, s in enumerate(all_styles)}
    log.info(f"Styles in 1k subset: {all_styles}")

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    dataset = WikiArtClipDataset(metadata, hf_samples, preprocess, tokenizer, style_to_id)
    sampler = make_weighted_sampler(metadata)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         num_workers=0, pin_memory=(DEVICE == "cuda"))

    # ── Optimiser + LR schedule ──────────────────────────────────────────────
    trainable_params = configure_trainable_params(model, backbone, unfreeze_blocks)
    loss_fn   = CLIPStyleLoss(init_temp=0.07).to(DEVICE)
    optimiser = torch.optim.AdamW(
        [{"params": trainable_params}, {"params": loss_fn.parameters()}],
        lr=lr, weight_decay=0.01
    )
    total_steps  = epochs * len(loader)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"\nStarting fine-tuning: {epochs} epochs × {len(loader)} steps = {total_steps} total")
    best_loss = float("inf")
    history   = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        model.train()

        for step, (imgs, tokens, _style_ids) in enumerate(loader, 1):
            imgs   = imgs.to(DEVICE)
            tokens = tokens.to(DEVICE)

            # Forward
            img_feat  = model.encode_image(imgs)
            text_feat = model.encode_text(tokens)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            loss = loss_fn(img_feat, text_feat)

            # Backward
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 20 == 0:
                log.info(f"  Epoch {epoch}/{epochs}  Step {step}/{len(loader)}"
                         f"  Loss={loss.item():.4f}  LR={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        log.info(f"Epoch {epoch}/{epochs} complete — avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")
        history.append({"epoch": epoch, "avg_loss": avg_loss})

        # Save checkpoint
        ckpt_path = out / f"clip_{backbone.replace('/', '_')}_epoch{epoch}.pt"
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "loss_fn_state": loss_fn.state_dict(),
            "avg_loss":     avg_loss,
        }, ckpt_path)
        log.info(f"  Checkpoint saved → {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = out / f"clip_{backbone.replace('/', '_')}_best.pt"
            torch.save(model.state_dict(), best_path)
            log.info(f"  ✓ New best model saved → {best_path}")

    # ── Build fine-tuned index ────────────────────────────────────────────────
    log.info("\nBuilding fine-tuned embedding index…")
    best_state = torch.load(out / f"clip_{backbone.replace('/', '_')}_best.pt", map_location=DEVICE)
    model.load_state_dict(best_state)
    build_finetuned_index(model, preprocess, hf_samples, metadata, out)

    # ── Save training history ─────────────────────────────────────────────────
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\nFine-tuning complete. Best loss: {best_loss:.4f}")
    log.info(f"All outputs saved to: {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on WikiArt for Art Style RAG")
    parser.add_argument("--backbone",        default="ViT-B-32",
                        choices=["ViT-B-32", "ViT-L-14"],
                        help="CLIP backbone architecture")
    parser.add_argument("--meta",            default="data/wikiart_1k.json",
                        help="Metadata JSON from data_pipeline.py")
    parser.add_argument("--output",          default="data/finetuned_b32/",
                        help="Directory for checkpoints and fine-tuned index")
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--unfreeze_blocks", type=int,   default=4,
                        help="Number of top transformer blocks to unfreeze")
    args = parser.parse_args()

    train(
        backbone        = args.backbone,
        meta_path       = args.meta,
        output_dir      = args.output,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        unfreeze_blocks = args.unfreeze_blocks,
    )

# Class-Channel Neighbour Transformer

An optional post-processing block that can be injected at the end of
PointTransformerV3 via the Hydra model config.  When the config key is absent
the model behaves identically to before — zero code-path changes, no weight
differences.

---

## Motivation

PTv3 processes all feature channels jointly: its encoder/decoder attention
mixes class logits, Gaussian geometry, colour, and opacity together in a single
high-dimensional representation.  This is intentional — cross-channel context
helps the network learn good representations — but it means the final per-class
logits are produced by a general-purpose transformation that has no explicit
awareness of class-level spatial consistency.

The **Class-Channel Neighbour Transformer** adds a targeted second pass:

> *For each class `c` independently, look at what nearby Gaussian splats
> predicted for class `c`, and decide whether to adjust the current splat's
> class-`c` logit.*

No information flows from class `c′ ≠ c` into class `c`'s update — the
channels are processed with strict independence.  Crucially, the weights are
**shared across all classes** (same transformation applied to every class's
feature vector), so the module is invariant to the ordering of classes.  This
is important because class ordering in PTv3 is sorted by per-scene frequency
and therefore changes between scenes.

This is inspired by cost-aggregation approaches in 2-D open-vocabulary
segmentation (e.g. CAT-Seg, NeurIPS 2023), where the class-logit volume is
treated as a structured "cost" to be refined by spatial aggregation before the
final prediction.

---

## Pipeline position

```
Input features [N, C+11]          (C class logits + 11 Gaussian properties)
        │
        ├──── saved as input_logits [N, C] (if enable_skip or skip_to_output)
        ▼
   PTv3 encoder
        │
        ▼
   PTv3 decoder  →  point.feat [N, dec_channels[0]]
        │               = [N, C]             (embedding_dim=1, standard)
        │               = [N, C·embedding_dim] (embedding_dim>1, rich)
        │  (enable_skip residual added here if configured; only when embedding_dim=1)
        ▼
ClassChannelNeighborTransformer
        │
        ▼
point.feat  [N, C]                 ← refined logits
        │  (skip_to_output residual added here if configured)
        ▼
point.feat  [N, C]                 ← final output, identical interface to no-CCNT
        │
        ▼
   Neighbour voting → point-cloud predictions
```

---

## Architecture

### Standard mode (`use_embed: true`, `embedding_dim: 1`)

The decoder outputs `[N, C]` scalar logits.  Each logit is lifted to a
`d_head`-dimensional vector, processed by the transformer, then projected back
to a scalar delta and added as a residual.

```
Input: point.feat [N, C]
       │
  ┌────┘  (save as orig_feat for residual)
  │  shared Linear(1 → d_head)     embed scalar logit → d_head vector
  │  applied per class independently
  ▼
x: [N, C, d_head]
  │
  └─── transformer blocks (see below) ───▶ x: [N, C, d_head]
                                                │
                                         shared Linear(d_head → 1)  [zero-init]
                                                ▼
                                          delta: [N, C]
                                                │
                                          + orig_feat                (internal residual)
                                                ▼
                                    Output: point.feat [N, C]
```

Identity at init: `out_proj` zero-init → delta = 0 → output = PTv3 logits unchanged.

### Rich-embedding mode (`use_embed: true`, `embedding_dim > 1`)

The decoder outputs `[N, C·embedding_dim]` rich per-class embeddings.  These
are reshaped and projected up to `d_head` before the transformer.  No internal
residual is applied (the per-class embedding space has no direct correspondence
to the output logit space); use `skip_to_output: true` for a full-model
residual over the input logits.

```
Input: point.feat [N, C·E]       (E = embedding_dim)
       │
  reshape → [N, C, E]
       │
  shared Linear(E → d_head)      upscale to attention space
       ▼
x: [N, C, d_head]
  │
  └─── transformer blocks ───▶ x: [N, C, d_head]
                                      │
                               shared Linear(d_head → 1)  [zero-init]
                                      ▼
                           Output: point.feat [N, C]     (+ skip_to_output if set)
```

Zero-init at start → uniform logits; well-defined CE gradients; learns quickly
from the rich decoder features.

### Direct-embedding mode (`use_embed: false`)

The decoder outputs `[N, C·embedding_dim]` and the transformer operates
directly in that space (`d_head = embedding_dim`).  Requires `embedding_dim ≥ 8`
for flash-attn.

```
Input: point.feat [N, C·D]       (D = embedding_dim = d_head)
       │
  reshape → [N, C, D]            no embed, use decoder space directly
       │
  └─── transformer blocks ───▶ x: [N, C, D]
                                      │
                               shared Linear(D → 1)  [zero-init]
                                      ▼
                           Output: point.feat [N, C]     (+ skip_to_output if set)
```

### Transformer block (shared across all modes)

```
┌───────────────────────────────────────────────┐
│  repeat num_layers times                       │
│                                               │
│  LayerNorm(d_head) per (N, C)                 │
│        │                                      │
│  Virtual-batch serialised attention            │
│  • class c → rows [c·N, (c+1)·N)              │
│    in a [C·N, d_head] tensor                  │
│  • PTv3 serialisation order replicated C×     │
│  • patch_size points per window               │
│  • shared QKV and proj weights                │  ← class-invariant
│  • flash attention (if enable_flash: true)    │
│  • reshape back → [N, C, d_head]             │
│        │                                      │
│  + residual                                   │
│        │                                      │
│  LayerNorm(d_head) per (N, C)                 │
│        │                                      │
│  shared Linear FFN (per token)                │  d_head → d_head·mlp_ratio → d_head
│        │  (fc2 zero-init)                     │
│  + residual                                   │
└───────────────────────────────────────────────┘
```

### Key properties

| Property | How it is achieved |
|---|---|
| Per-class independence | C classes packed as a virtual batch; each class attends only to its own neighbours |
| Class invariance (weight sharing) | Same QKV/FFN/proj weights applied to every class's feature vector |
| Permutation invariance | No positional encoding; spatial locality from PTv3's serialisation order |
| Identity at init | `out_proj` and all `fc2` weights are zero-initialised |
| Reuses PTv3 infrastructure | Subclasses `SerializedAttention`; inherits flash-attn path, padding, serialisation order |

---

## Config parameters

### `neighbor_transformer` block

```yaml
neighbor_transformer:
  use_embed: true    # true (default): use Linear(embedding_dim, d_head) before attention
                     # false: decoder output is used directly as d_head-dim features

  d_head: 16         # Attention space dimension per class.
                     # Used when use_embed=true; ignored when use_embed=false.
                     # flash-attn requires head_dim = d_head // num_heads ≥ 8.

  num_heads: 1       # Attention heads within d_head space.
                     # head_dim = d_head // num_heads; must be ≥ 8 for flash-attn.

  num_layers: 1      # Stacked transformer blocks. 1–2 is usually sufficient.

  mlp_ratio: 2       # FFN hidden-dim multiplier: hidden = d_head * mlp_ratio.

  enable_flash: true # Per-CCNT flash-attn override. Useful when d_head is small
                     # (e.g. use_embed=false with embedding_dim=4 < 8 minimum).

  patch_size: 1024   # Points per attention window (PTv3 serialisation order).
                     # Defaults to dec_patch_size[0] from decoder config.

  order_index: 0     # Serialisation curve to use (0 = first curve in PTv3 config).
```

### Top-level model parameters

```yaml
embedding_dim: 1     # Per-class embedding depth from decoder.
                     # dec_channels[0] = embedding_dim * class_subset (injected at runtime).
                     # embedding_dim=1: standard scalar logits.
                     # embedding_dim>1: rich per-class embeddings; requires enable_skip: false.

enable_skip: false   # Add input logits to decoder output BEFORE CCNT.
                     # Only compatible with embedding_dim=1.

skip_to_output: true # Add input logits to CCNT output AFTER CCNT.
                     # Compatible with any embedding_dim.
                     # Recommended when embedding_dim>1 (no internal residual in that mode).
```

Omitting the `neighbor_transformer` key entirely disables the module.

---

## Example configs

### `medium_inter` — scalar logits, standard embed (`embedding_dim=1`, `use_embed=true`)

```yaml
embedding_dim: 1          # default; dec_channels[0] = class_subset
enable_skip: true         # residual before CCNT (compatible at embedding_dim=1)
neighbor_transformer:
  use_embed: true
  d_head: 16
  num_heads: 1
  num_layers: 1
  mlp_ratio: 2
```

### `larger_embed` — rich decoder embeddings + upscale (`embedding_dim=4`, `use_embed=true`)

```yaml
embedding_dim: 4          # dec_channels[0] = 4 * class_subset (e.g. 208 for 52 classes)
enable_skip: false        # incompatible with embedding_dim > 1
skip_to_output: true      # full-model residual after CCNT
neighbor_transformer:
  use_embed: true          # Linear(4, 16): upscale 4-dim embedding to 16-dim attention space
  d_head: 16               # flash-attn works (16 ≥ 8 minimum)
  enable_flash: true
  num_heads: 1
  num_layers: 1
  mlp_ratio: 2
```

Channel flow (class_subset=52): decoder → `[N, 208]` → reshape `[N, 52, 4]` → `Linear(4,16)` → `[N, 52, 16]` → attention → `[N, 52]` logits → `+ input_logits`.

---

## Memory and compute

With typical settings (N = 400 k, C = 52, patch_size = 1024):

| Mode | d_head | Virtual feat tensor | Flash-attn |
|---|---|---|---|
| `use_embed=true`, `embedding_dim=1` | 16 | [20.8M, 16] ~1.3 GB fp32 | yes (head_dim=16) |
| `use_embed=true`, `embedding_dim=4` | 16 | [20.8M, 16] ~1.3 GB fp32 | yes (head_dim=16) |
| `use_embed=false`, `embedding_dim=8` | 8 | [20.8M, 8] ~0.7 GB fp32 | yes (head_dim=8, minimum) |
| `use_embed=false`, `embedding_dim=4` | 4 | [20.8M, 4] ~0.3 GB fp32 | **no** (head_dim=4 < 8) |

The rich-embed modes (`embedding_dim>1`) also widen the decoder's finest stage
(e.g. `[N, 208]` instead of `[N, 52]`), adding some decoder overhead, but the
CCNT attention cost itself is determined only by `d_head`.

The module adds roughly **10–15 % overhead** to a single forward pass and
**~5–8 GB additional VRAM** during training (forward + gradients).  Both scale
linearly with `num_layers`.

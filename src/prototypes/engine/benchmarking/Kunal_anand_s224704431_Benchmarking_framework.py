# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Audio Benchmark — **BCResNetLite** & **PatchTransformerTiny**
#
# This notebook shows how to integrate **two different modules** (architectures) into an audio benchmarking framework using **different variable names** from your friend’s code and **detailed comments**.
#
# ### Architectures
# - **BCResNetLite** — tiny residual CNN for log-mel spectrograms  
# - **PatchTransformerTiny** — small Transformer encoder over spectrogram **patches**
#
# ### Assumptions
# - Input tensors are log-mel spectrograms shaped **(batch, 1, n_mels, time)**
# - Number of classes is configurable via `n_classes`
# - Only dependency is **PyTorch**
#
# We’ll first define utility blocks, then both models, add a **registry** (factory pattern), and finally a **tiny benchmark harness** to sanity-check wiring on random data. Replace the dummy bits with your real pipeline later.
#

# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Device helper for later (CUDA if available)
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# ## Utility Blocks
#
# We keep models tidy by reusing small building blocks:
#
# - `ConvBNAct`: 2D convolution → BatchNorm → activation (SiLU by default).  
#   Good for stems/heads.
# - `DepthwiseSeparable`: Depthwise conv + Pointwise (1×1) conv.  
#   Efficient and widely used in mobile/compact CNNs.
#
# > Note: we implement `DepthwiseSeparable` so it accepts **tuple stride** `(h, w)`.  
# > This ensures the residual block can downsample the **main path** consistently.
#

# %%
class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm2d → activation (SiLU by default)."""
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        act: nn.Module | None = None,
    ):
        super().__init__()
        if p is None:
            p = k // 2  # "same-ish" padding for odd kernels
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = act if act is not None else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """
    Depthwise convolution followed by pointwise (1x1) convolution.
    Accepts stride as int or (h, w) tuple — important for downsampling.
    """
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int | Tuple[int, int] = 1):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s, padding=pad, groups=c_in, bias=False
        )
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)



# %% [markdown]
# ## BCResNetLite (Tiny Residual CNN)
#
# A compact ResNet-style CNN for spectrograms:
# - Uses our `DepthwiseSeparable` block for efficiency
# - Downsamples **inside the first conv** of the block so shapes match the skip path
# - Global average over (freq, time) → classifier head
#
# > The stride logic avoids residual shape mismatches like  
# > “size of tensor a (T1) must match size of tensor b (T2)”.
#

# %%
class BCResBlock(nn.Module):
    """
    Two depthwise-separable convs + residual connection.
    The FIRST conv uses the block stride so main path matches skip path size.
    """
    def __init__(self, c_in: int, c_out: int, stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        s_h, s_w = stride

        # Main path: apply stride here for downsampling
        self.conv1 = DepthwiseSeparable(c_in, c_out, k=3, s=(s_h, s_w))
        self.conv2 = DepthwiseSeparable(c_out, c_out, k=3, s=1)

        # Skip path: downsample if needed, and project channels if they differ
        self.pool = (
            nn.AvgPool2d(kernel_size=(s_h, s_w), stride=(s_h, s_w))
            if (s_h > 1 or s_w > 1) else nn.Identity()
        )
        self.proj = (
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            if c_in != c_out else nn.Identity()
        )

        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.proj(self.pool(x))
        out = self.bn(out + skip)
        return self.act(out)


class BCResNetLite(nn.Module):
    """
    Small CNN for audio spectrogram classification.
    Input: (B, 1, n_mels, time) → logits (B, num_labels)
    """
    def __init__(self, n_mels: int, num_labels: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        # Stem reduces both dims (H/2, W/2)
        self.stem = ConvBNAct(1, c1, k=5, s=2)

        # Stages: control how freq/time are reduced
        self.stage1 = BCResBlock(c1, c1, stride=(1, 2))  # downsample TIME only
        self.stage2 = BCResBlock(c1, c2, stride=(2, 2))  # downsample both
        self.stage3 = BCResBlock(c2, c3, stride=(2, 2))  # downsample both

        # Head
        self.head_norm = nn.LayerNorm(c3)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c3, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Global average pool over (freq, time)
        x = x.mean(dim=[2, 3])              # (B, C)
        x = self.head_norm(x)
        x = self.dropout(x)
        return self.classifier(x)



# %% [markdown]
# ##  PatchTransformerTiny (Patch-based Transformer)
#
# A tiny Transformer encoder on spectrogram **patches**:
# - `PatchEmbed` is a Conv2d with `stride=patch_size` → sequences of patch tokens
# - Prepend a learnable **[CLS]** token for pooled representation
# - A stack of minimal Transformer encoder layers (pre-norm)
#
# > Positional encodings are omitted for simplicity but can be added later.
#

# %%
class PatchEmbed(nn.Module):
    """Conv patchify: (B, 1, H, W) → (B, N_patches, C)."""
    def __init__(self, in_ch: int, embed_dim: int, patch: Tuple[int, int] = (4, 4)):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                              # (B, C, H/P, W/P)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)              # (B, Hp*Wp, C)
        return self.norm(x)


class TransformerEncoderLayer(nn.Module):
    """Minimal pre-norm Transformer encoder layer (MHA + MLP)."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, d_model), nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MLP with residual
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class PatchTransformerTiny(nn.Module):
    """Tiny Transformer for spectrogram patches with a learnable [CLS] token."""
    def __init__(
        self,
        n_mels: int,
        num_labels: int,
        embed_dim: int = 96,
        depth: int = 4,
        n_heads: int = 3,
        patch: Tuple[int, int] = (4, 4),
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.patchify = PatchEmbed(1, embed_dim, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, mlp_ratio=4.0, p_drop=p_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patchify(x)                         # (B, N, C)
        cls_tok = self.cls.expand(B, -1, -1)         # (B, 1, C)
        x = torch.cat([cls_tok, x], dim=1)           # prepend CLS → (B, 1+N, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]                                # pooled rep
        return self.head(cls)                        # (B, num_labels)



# %% [markdown]
# ## Builders & Registry (Different Variable Names)
#
# To make integration easy, we expose **factory functions** and a central **registry**.
# These names are **different** from your friend’s to avoid conflicts.
#
# - `make_bcresnet_lite(cfg)`  
# - `make_patchformer_tiny(cfg)`  
# - `ARCH_REGISTRY` maps a string key → builder function
#

# %%
@dataclass
class NetArgs:
    n_mels: int = 64
    n_classes: int = 10

def make_bcresnet_lite(cfg: NetArgs) -> nn.Module:
    """Factory for BCResNetLite."""
    return BCResNetLite(n_mels=cfg.n_mels, num_labels=cfg.n_classes, width=32, dropout=0.1)

def make_patchformer_tiny(cfg: NetArgs) -> nn.Module:
    """Factory for PatchTransformerTiny."""
    return PatchTransformerTiny(n_mels=cfg.n_mels, num_labels=cfg.n_classes, embed_dim=96, depth=4, n_heads=3, patch=(4, 4), p_drop=0.1)

ARCH_REGISTRY: Dict[str, Callable[[NetArgs], nn.Module]] = {
    "BCResNetLite": make_bcresnet_lite,
    "PatchTransformerTiny": make_patchformer_tiny,
}


# %% [markdown]
# ##  Minimal Benchmark Harness
#
# A tiny training/evaluation loop with a **dummy dataset**:
# - `RandomSpectrograms`: returns random mel-spectrograms and random labels
# - `train_one_epoch`: one pass of cross-entropy training
# - `evaluate_top1`: simple top-1 accuracy
#
# > Replace this section with your real dataset/loader and trainer later.
#

# %%
class RandomSpectrograms(torch.utils.data.Dataset):
    """Dummy dataset to test end-to-end wiring; replace with your real dataset."""
    def __init__(self, n_items: int, n_mels: int, time_steps: int, n_classes: int):
        self.n_items = n_items
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_classes = n_classes

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int):
        x = torch.randn(1, self.n_mels, self.time_steps)  # (1, n_mels, T)
        y = torch.randint(0, self.n_classes, (1,)).item()
        return x, y


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> float:
    """Single epoch on the toy dataset (for sanity checking)."""
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def evaluate_top1(model: nn.Module, loader, device: torch.device) -> float:
    """Compute top-1 accuracy on the toy validation set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / max(1, total)


def quick_benchmark(
    arch_key: str,
    cfg: NetArgs,
    device: str = DEVICE_DEFAULT,
) -> None:
    """
    Tiny train/eval pass to ensure the selected model is wired correctly.
    Prints average loss and toy accuracy.
    """
    device_t = torch.device(device)
    model = ARCH_REGISTRY[arch_key](cfg).to(device_t)

    # Toy loaders
    train_ds = RandomSpectrograms(n_items=128, n_mels=cfg.n_mels, time_steps=256, n_classes=cfg.n_classes)
    val_ds   = RandomSpectrograms(n_items=64,  n_mels=cfg.n_mels, time_steps=256, n_classes=cfg.n_classes)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=32)

    # Modest optimizer/lr for stability on random data
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    avg_loss = train_one_epoch(model, train_ld, optim, device_t)
    acc = evaluate_top1(model, val_ld, device_t)
    print(f"[SanityCheck] {arch_key}: loss={avg_loss:.3f}, acc={acc:.1f}%")



# %% [markdown]
# ## Run Sanity Check
#
# Create a config and run **both** models from the registry.  
# On random data, accuracy is meaningless; we just verify the end-to-end path.
#

# %%
cfg = NetArgs(n_mels=64, n_classes=35)  # e.g., SpeechCommands ~35 classes
for key in ARCH_REGISTRY.keys():
    quick_benchmark(key, cfg)


# %% [markdown]
# ## Extra Shape Probe (Optional)
#
# If you want to confirm output sizes manually for **BCResNetLite**:
#

# %%
with torch.no_grad():
    model = BCResNetLite(n_mels=64, num_labels=35)
    dummy = torch.randn(2, 1, 64, 256)  # (B, 1, n_mels, time)
    out = model(dummy)
    print("Logits shape:", out.shape)  # Expect: (2, 35)


# %%
# Under Development
# Robust visualization utilities (Matplotlib-only; no Seaborn/SciPy required)

import math
from typing import Optional, Union, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _to_percent_series(s: pd.Series) -> pd.Series:
    """Return % values: if data is 0..1 scale → convert to 0..100; otherwise assume already percent."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().between(0, 1).all():
        return s * 100.0
    return s

def _annotate_bars(ax, orient: str = "v", fmt: str = "{:.1f}"):
    """Add numeric labels on bars (vertical or horizontal)."""
    for p in ax.patches:
        if orient == "v":
            val = p.get_height()
            if math.isnan(val): continue
            ax.text(p.get_x() + p.get_width()/2, val, fmt.format(val),
                    ha="center", va="bottom", fontsize=9)
        else:
            val = p.get_width()
            if math.isnan(val): continue
            ax.text(val, p.get_y() + p.get_height()/2, fmt.format(val),
                    ha="left", va="center", fontsize=9)

def _mpl_barplot(ax, x_labels, y_values, *, color=None, orient="v"):
    """Minimal Matplotlib bar plot with nice defaults."""
    if orient == "v":
        ax.bar(x_labels, y_values, color=color)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        ax.barh(x_labels, y_values, color=color)

def visualize_results(
    results_df: Optional[pd.DataFrame],
    aug_data: Optional[Union[pd.DataFrame, Sequence[dict]]] = None,
    *,
    exp_col: str = "Experiment",
    acc_col: str = "Test Accuracy",
    f1_col: str = "F1 Score",
    time_col: str = "Training Time (min)",
    model_col: str = "Model",
):
    """
    Visualize experiment metrics in a 2x2 dashboard and (optionally) augmentation impact.

    results_df columns:
      - Experiment (str)
      - Test Accuracy (float; 0..1 or 0..100)
      - F1 Score (float; 0..1 or 0..100)
      - Training Time (min) (float)
      - Model (str)

    aug_data (optional) columns:
      - Augmentation (str), Accuracy (float; 0..1 or 0..100), Augmentation Type (str)
    """
    if results_df is None or len(results_df) == 0:
        print("No results available to visualize.")
        return

    df = results_df.copy()

    have_acc = acc_col in df.columns and exp_col in df.columns
    have_f1  = f1_col  in df.columns and exp_col in df.columns
    have_tm  = time_col in df.columns and exp_col in df.columns
    have_mod = model_col in df.columns and acc_col in df.columns

    if have_acc: df["_acc_pct_"] = _to_percent_series(df[acc_col])
    if have_f1:  df["_f1_pct_"]  = _to_percent_series(df[f1_col])

    # Sort experiments by accuracy if available
    if have_acc:
        df_sorted = df.sort_values("_acc_pct_", ascending=False)
        exp_order = df_sorted[exp_col].tolist()
    else:
        df_sorted = df.copy()
        exp_order = df[exp_col].tolist() if exp_col in df.columns else []

    # 2x2 dashboard
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax11, ax12, ax21, ax22 = axes.ravel()

    # 1) Test Accuracy by Experiment
    if have_acc:
        _mpl_barplot(ax11, exp_order, df.set_index(exp_col).loc[exp_order]["_acc_pct_"].values)
        ax11.set_title("Test Accuracy by Experiment")
        ax11.set_xlabel("Experiment"); ax11.set_ylabel("Accuracy (%)"); ax11.set_ylim(0, 100)
        _annotate_bars(ax11, orient="v", fmt="{:.1f}")
    else:
        ax11.axis("off"); ax11.text(0.5, 0.5, f"Missing '{acc_col}' / '{exp_col}'", ha="center", va="center")

    # 2) F1 Score by Experiment
    if have_f1:
        _mpl_barplot(ax12, exp_order, df.set_index(exp_col).loc[exp_order]["_f1_pct_"].values, color="#90CAF9")
        ax12.set_title("F1 Score by Experiment")
        ax12.set_xlabel("Experiment"); ax12.set_ylabel("F1 (%)"); ax12.set_ylim(0, 100)
        _annotate_bars(ax12, orient="v", fmt="{:.1f}")
    else:
        ax12.axis("off"); ax12.text(0.5, 0.5, f"Missing '{f1_col}' / '{exp_col}'", ha="center", va="center")

    # 3) Training Time by Experiment (minutes)
    if have_tm:
        _mpl_barplot(ax21, exp_order, df.set_index(exp_col).loc[exp_order][time_col].values, color="#A5D6A7")
        ax21.set_title("Training Time by Experiment (minutes)")
        ax21.set_xlabel("Experiment"); ax21.set_ylabel("Minutes")
        _annotate_bars(ax21, orient="v", fmt="{:.1f}")
    else:
        ax21.axis("off"); ax21.text(0.5, 0.5, f"Missing '{time_col}' / '{exp_col}'", ha="center", va="center")

    # 4) Average Accuracy by Model
    if have_mod:
        model_df = (
            df[[model_col, "_acc_pct_"]]
            .dropna()
            .groupby(model_col, as_index=False)["_acc_pct_"]
            .mean()
            .sort_values("_acc_pct_", ascending=False)
        )
        _mpl_barplot(ax22, model_df[model_col].astype(str).tolist(), model_df["_acc_pct_"].values, color="#FFCC80")
        ax22.set_title("Average Accuracy by Model")
        ax22.set_xlabel("Model"); ax22.set_ylabel("Accuracy (%)"); ax22.set_ylim(0, 100)
        _annotate_bars(ax22, orient="v", fmt="{:.1f}")
    else:
        ax22.axis("off"); ax22.text(0.5, 0.5, f"Missing '{model_col}' / '{acc_col}'", ha="center", va="center")

    plt.show()

    # ----- Augmentation Impact (optional) -----
    if aug_data is not None:
        if isinstance(aug_data, pd.DataFrame):
            aug_df = aug_data.copy()
        else:
            aug_df = pd.DataFrame(list(aug_data))

        need = {"Augmentation", "Accuracy", "Augmentation Type"}
        if need.issubset(aug_df.columns):
            aug_df["_acc_pct_"] = _to_percent_series(aug_df["Accuracy"])

            # Grouped bars per augmentation type (Matplotlib)
            types = sorted(aug_df["Augmentation Type"].unique())
            augs  = list(aug_df["Augmentation"].unique())
            x_idx = np.arange(len(augs))
            width = 0.8 / max(1, len(types))

            plt.figure(figsize=(12, 6))
            for i, t in enumerate(types):
                sub = aug_df[aug_df["Augmentation Type"] == t]
                vals = [float(sub[sub["Augmentation"] == a]["_acc_pct_"].mean()) if a in set(sub["Augmentation"]) else 0.0 for a in augs]
                plt.bar(x_idx + i*width, vals, width=width, label=t)

            plt.xticks(x_idx + width*(len(types)-1)/2, augs, rotation=45, ha="right")
            plt.title("Accuracy by Augmentation Type")
            plt.xlabel("Augmentation"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
            plt.legend(title="Type"); plt.tight_layout(); plt.show()
        else:
            missing = ", ".join(sorted(need - set(aug_df.columns)))
            print(f"[augmentations] Skipped: missing columns → {missing}")

# Auto-run if results_df exists
try:
    visualize_results(results_df, aug_data if 'aug_data' in globals() else None)
except NameError:
    print("Define `results_df` (and optional `aug_data`) then call: visualize_results(results_df, aug_data)")


# %%
# --- Sample data to drive the visualizations ---
# Replace these with your real metrics when you have them.

import pandas as pd

# Results across multiple experiments / models
results_df = pd.DataFrame([
    {"Experiment": "BCResNetLite_lr3e-4_bs16",  "Test Accuracy": 0.74, "F1 Score": 0.72, "Training Time (min)": 7.5,  "Model": "BCResNetLite"},
    {"Experiment": "BCResNetLite_lr1e-3_bs32",  "Test Accuracy": 0.77, "F1 Score": 0.75, "Training Time (min)": 6.8,  "Model": "BCResNetLite"},
    {"Experiment": "PatchFormerTiny_lr3e-4",    "Test Accuracy": 0.79, "F1 Score": 0.78, "Training Time (min)": 9.2,  "Model": "PatchTransformerTiny"},
    {"Experiment": "PatchFormerTiny_lr1e-3",    "Test Accuracy": 0.81, "F1 Score": 0.80, "Training Time (min)": 8.9,  "Model": "PatchTransformerTiny"},
    {"Experiment": "BCResNetLite_aug-specaug",  "Test Accuracy": 0.83, "F1 Score": 0.82, "Training Time (min)": 7.9,  "Model": "BCResNetLite"},
    {"Experiment": "PatchFormerTiny_aug-mixup", "Test Accuracy": 0.85, "F1 Score": 0.84, "Training Time (min)": 9.6,  "Model": "PatchTransformerTiny"},
])

# Optional: augmentation impact data (shown in the second figure)
aug_data = [
    {"Augmentation": "Time Mask",   "Accuracy": 0.80, "Augmentation Type": "SpecAug"},
    {"Augmentation": "Freq Mask",   "Accuracy": 0.82, "Augmentation Type": "SpecAug"},
    {"Augmentation": "Mixup",       "Accuracy": 0.84, "Augmentation Type": "Mixing"},
    {"Augmentation": "Noise",       "Accuracy": 0.78, "Augmentation Type": "Classical"},
    {"Augmentation": "Pitch Shift", "Accuracy": 0.79, "Augmentation Type": "Classical"},
    {"Augmentation": "Time Stretch","Accuracy": 0.81, "Augmentation Type": "Classical"},
]

# --- Render the visuals ---
visualize_results(results_df, aug_data)


# %%

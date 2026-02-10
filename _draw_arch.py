"""Generate architecture diagram for the GPU fraud detection pipeline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colour palette ──────────────────────────────────────────────────
C_BG       = "#161b22"
C_BORDER   = "#30363d"
C_GREEN    = "#238636"
C_BLUE     = "#1f6feb"
C_PURPLE   = "#8957e5"
C_ORANGE   = "#d29922"
C_RED      = "#f85149"
C_TEAL     = "#3fb950"
C_CYAN     = "#58a6ff"
C_TEXT     = "#e6edf3"
C_SUBTEXT  = "#8b949e"
C_ARROW    = "#58a6ff"

# ── Helper: rounded box ────────────────────────────────────────────
def box(x, y, w, h, color, label, sublabel=None, fontsize=11, icon=None):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor="#48505a",
        linewidth=1.3,
        alpha=0.92,
    )
    ax.add_patch(rect)
    text = f"{icon}  {label}" if icon else label
    ty = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C_TEXT, family="monospace")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel, ha="center", va="center",
                fontsize=8.5, color=C_SUBTEXT, family="monospace")

def section_box(x, y, w, h, title, color):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2",
        facecolor="#0d1117",
        edgecolor=color,
        linewidth=1.8,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(rect)
    ax.text(x + 0.3, y + h - 0.32, title, fontsize=9, fontweight="bold",
            color=color, family="monospace", alpha=0.9)

def arrow(x1, y1, x2, y2, label=None, color=C_ARROW):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.8,
            connectionstyle="arc3,rad=0.0",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.22, label, ha="center", va="center",
                fontsize=7.5, color=C_SUBTEXT, family="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none"))

def arrow_curved(x1, y1, x2, y2, label=None, color=C_ARROW, rad=0.3):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.6,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )

# ── Title ───────────────────────────────────────────────────────────
ax.text(9, 12.55, "GPU-Accelerated Fraud Detection Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=C_TEXT, family="monospace")
ax.text(9, 12.15, "NVIDIA RAPIDS  ·  XGBoost gpu_hist  ·  FastAPI  ·  H100",
        ha="center", va="center", fontsize=10, color=C_SUBTEXT, family="monospace")

# ── Section: Docker / GPU Runtime ───────────────────────────────────
section_box(0.3, 0.3, 17.4, 11.55, "Docker  ·  nvidia runtime  ·  CUDA 12.0  ·  RAPIDS 24.02", C_BORDER)

# ══════════════════════════════════════════════════════════════════════
# ROW 1 — Data Generation
# ══════════════════════════════════════════════════════════════════════
section_box(0.7, 8.8, 7.8, 2.7, "Stage 1: Data Generation  (src/data/)", C_GREEN)

box(1.1, 9.6, 3.2, 1.4, "#1a3024", "FraudDataGenerator", "generator.py", fontsize=10)
box(4.8, 9.6, 3.3, 1.4, "#1a3024", "TransactionSchema", "schema.py  ·  16 cols", fontsize=10)

# Details inside data gen
ax.text(2.7, 9.25, "CuPy arrays → cuDF DataFrame", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")
ax.text(6.45, 9.25, "10 merchant cats · 4 txn types", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")

arrow(4.3, 10.3, 4.8, 10.3, "schema")

# ══════════════════════════════════════════════════════════════════════
# ROW 1 — Config & Metrics (right side)
# ══════════════════════════════════════════════════════════════════════
section_box(9.1, 8.8, 8.2, 2.7, "Utils  (src/utils/)", C_ORANGE)

box(9.5, 9.6, 3.5, 1.4, "#2d2208", "PipelineConfig", "config.py  ·  dataclasses", fontsize=10)
box(13.5, 9.6, 3.4, 1.4, "#2d2208", "BenchmarkResults", "metrics.py  ·  timing", fontsize=10)

ax.text(11.25, 9.25, "from_env() overrides", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")
ax.text(15.2, 9.25, "timed_operation() ctx mgr", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")

# ══════════════════════════════════════════════════════════════════════
# PARQUET storage
# ══════════════════════════════════════════════════════════════════════
box(1.1, 7.6, 3.2, 0.85, "#1c1c2e", "Parquet Files", "data/ partitioned 1M rows", fontsize=9)

arrow(2.7, 9.6, 2.7, 8.45, "cuDF →\nParquet", color=C_TEAL)

# ══════════════════════════════════════════════════════════════════════
# ROW 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════
section_box(0.7, 5.1, 7.8, 2.3, "Stage 2: Feature Engineering  (src/features/)", C_BLUE)

box(1.1, 5.5, 3.3, 1.4, "#0d2240", "FeatureEngineer", "engineering.py", fontsize=10)

ax.text(5.4, 6.65, "user_avg_amount\nvelocity_1h/6h/24h\nmerchant_risk_score\nlog_amount · percentiles",
        ha="left", va="top", fontsize=8, color=C_SUBTEXT, family="monospace", linespacing=1.4)

arrow(2.7, 7.6, 2.7, 6.9, "load", color=C_TEAL)

# Feature parquet output
box(1.1, 3.9, 3.2, 0.85, "#1c1c2e", "Feature Parquet", "20+ engineered features", fontsize=9)
arrow(2.7, 5.5, 2.7, 4.75, "save", color=C_TEAL)

# ══════════════════════════════════════════════════════════════════════
# ROW 2 right — Benchmarking
# ══════════════════════════════════════════════════════════════════════
section_box(9.1, 5.1, 8.2, 2.3, "Benchmarking  (scripts/benchmark.py)", C_PURPLE)

box(9.5, 5.5, 3.3, 1.4, "#271746", "CPU vs GPU", "5-stage comparison", fontsize=10)
box(13.3, 5.5, 3.7, 1.4, "#271746", "Rich Summary", "speedup multipliers", fontsize=10)

ax.text(11.15, 5.2, "data · agg · sort · train · infer", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")

arrow(12.8, 6.2, 13.3, 6.2, "→")

# ══════════════════════════════════════════════════════════════════════
# ROW 3 — Training
# ══════════════════════════════════════════════════════════════════════
section_box(0.7, 1.3, 7.8, 2.35, "Stage 3: Training  (src/training/)", C_RED)

box(1.1, 1.65, 3.3, 1.45, "#3b1219", "XGBoostTrainer", "xgboost_gpu.py", fontsize=10)
box(4.9, 1.65, 3.2, 1.45, "#3b1219", "ModelEvaluator", "evaluation.py", fontsize=10)

ax.text(2.75, 1.38, "gpu_hist · early stop · DMatrix", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")
ax.text(6.5, 1.38, "AUC-ROC · PR · threshold", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")

arrow(2.7, 3.9, 2.7, 3.1, "load", color=C_TEAL)
arrow(4.4, 2.4, 4.9, 2.4, "eval")

# ══════════════════════════════════════════════════════════════════════
# ROW 3 right — Inference
# ══════════════════════════════════════════════════════════════════════
section_box(9.1, 1.3, 8.2, 2.35, "Stage 4: Inference  (src/inference/)", C_CYAN)

box(9.5, 1.65, 3.3, 1.45, "#0a2540", "FraudPredictor", "predictor.py", fontsize=10)
box(13.3, 1.65, 3.8, 1.45, "#0a2540", "FastAPI Server", "server.py  ·  :8000", fontsize=10)

ax.text(11.15, 1.38, "batch GPU inference", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")
ax.text(15.2, 1.38, "/predict · /batch · /health", ha="center", fontsize=7.5,
        color=C_SUBTEXT, family="monospace")

arrow(12.8, 2.4, 13.3, 2.4, "serves")

# ══════════════════════════════════════════════════════════════════════
# MODEL artifact — connecting training → inference
# ══════════════════════════════════════════════════════════════════════
box(5.2, 4.0, 3.5, 0.72, "#2d2208", "Model Artifact", "xgboost_fraud.json", fontsize=9)

arrow(4.4, 2.8, 5.2, 4.2, "save", color=C_ORANGE)
arrow(8.7, 4.36, 11.15, 3.1, "load", color=C_ORANGE)

# ══════════════════════════════════════════════════════════════════════
# External client arrow
# ══════════════════════════════════════════════════════════════════════
ax.annotate(
    "",
    xy=(17.05, 2.4), xytext=(17.6, 2.4),
    arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=2.2),
)
ax.text(17.85, 2.4, "Client\n(curl)", ha="center", va="center",
        fontsize=8, color=C_TEAL, family="monospace", fontweight="bold")

# ══════════════════════════════════════════════════════════════════════
# Pipeline CLI arrow (top → bottom)
# ══════════════════════════════════════════════════════════════════════
ax.text(8.55, 7.0, "scripts/\nrun_pipeline.py\n(Click CLI)", ha="center", va="center",
        fontsize=8, color=C_PURPLE, family="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c1030", edgecolor=C_PURPLE, lw=1.2))

arrow_curved(8.55, 6.5, 4.4, 6.2, color=C_PURPLE, rad=0.15)
arrow_curved(8.55, 6.5, 4.4, 2.8, color=C_PURPLE, rad=0.25)

# ── GPU badge ───────────────────────────────────────────────────────
gpu_rect = FancyBboxPatch(
    (14.5, 10.85), 2.8, 0.55,
    boxstyle="round,pad=0.12",
    facecolor=C_GREEN, edgecolor="#3fb950",
    linewidth=1.5, alpha=0.9,
)
ax.add_patch(gpu_rect)
ax.text(15.9, 11.13, "H100 80GB  ·  CUDA", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#ffffff", family="monospace")

# ── Legend ───────────────────────────────────────────────────────────
legend_y = 0.55
ax.text(1.0, legend_y, "Data flow ─── ", color=C_TEAL, fontsize=8, family="monospace")
ax.text(4.0, legend_y, "Model artifact ─── ", color=C_ORANGE, fontsize=8, family="monospace")
ax.text(7.8, legend_y, "CLI orchestration ─── ", color=C_PURPLE, fontsize=8, family="monospace")

plt.tight_layout(pad=0.5)
plt.savefig("/home/ubuntu/gpu_spark_app/arch.png", dpi=180, facecolor="#0d1117",
            edgecolor="none", bbox_inches="tight")
print("Saved arch.png")

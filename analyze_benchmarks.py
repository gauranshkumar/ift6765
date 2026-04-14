"""
analyze_benchmarks.py

Computes code-similarity metrics (exact match, BLEU, CodeBLEU, embedding cosine
similarity) between base / finetuned predictions and ground-truth PlantUML, then
produces publication-ready figures.

Usage (on the cluster):
    python analyze_benchmarks.py \
        --base         output/benchmark_results_base.parquet \
        --finetuned    output/benchmark_results_finetuned.parquet \
        --ground-truth output/qwen_lora_test_split.parquet \
        --output-dir   output/analysis

Dependencies (install with pip if missing):
    pip install sacrebleu codebleu sentence-transformers scikit-learn matplotlib seaborn
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise(code: str) -> str:
    """Lower-case, collapse whitespace — used for exact-match and BLEU."""
    if not isinstance(code, str):
        return ""
    # strip @startuml / @enduml wrappers so we compare content only
    code = re.sub(r"@startuml[^\n]*\n?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"@enduml[^\n]*", "", code, flags=re.IGNORECASE)
    return " ".join(code.lower().split())


def extract_uml_body(code: str) -> str:
    """Return inner content between @startuml and @enduml (normalised)."""
    return normalise(code)


# ─────────────────────────────────────────────────────────────────────────────
# Metric: exact match
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(pred: str, ref: str) -> float:
    return float(normalise(pred) == normalise(ref))


# ─────────────────────────────────────────────────────────────────────────────
# Metric: sentence BLEU (sacrebleu)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu_scores(preds: list[str], refs: list[str]) -> list[float]:
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        scores = []
        for p, r in zip(preds, refs):
            p_n = normalise(p)
            r_n = normalise(r)
            score = bleu.sentence_score(p_n, [r_n])
            scores.append(score.score / 100.0)   # 0-1
        return scores
    except ImportError:
        log.warning("sacrebleu not installed — skipping BLEU. pip install sacrebleu")
        return [float("nan")] * len(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Metric: CodeBLEU
# ─────────────────────────────────────────────────────────────────────────────

def compute_codebleu_scores(preds: list[str], refs: list[str]) -> list[float]:
    """
    Uses the `codebleu` package (pip install codebleu).
    Falls back to token-level BLEU on plaintext if unavailable.
    PlantUML has no dedicated grammar, so we use 'python' tokeniser as a proxy
    (it handles identifiers / punctuation well enough).
    """
    try:
        from codebleu import calc_codebleu
        scores = []
        for p, r in zip(preds, refs):
            if not p.strip() or not r.strip():
                scores.append(0.0)
                continue
            try:
                result = calc_codebleu(
                    references=[[r]],
                    predictions=[p],
                    lang="python",         # closest proxy for PlantUML keywords
                    weights=(0.25, 0.25, 0.25, 0.25),
                )
                scores.append(result["codebleu"])
            except Exception:
                scores.append(float("nan"))
        return scores
    except ImportError:
        log.warning("codebleu not installed — falling back to BLEU. pip install codebleu")
        return compute_bleu_scores(preds, refs)


# ─────────────────────────────────────────────────────────────────────────────
# Metric: embedding cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_embedding_similarity(preds: list[str], refs: list[str]) -> list[float]:
    """
    Uses sentence-transformers (all-MiniLM-L6-v2) if available,
    otherwise falls back to TF-IDF cosine similarity.
    """
    # --- sentence-transformers path ---
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos

        log.info("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts_p = [normalise(p) or "<empty>" for p in preds]
        texts_r = [normalise(r) or "<empty>" for r in refs]
        emb_p = model.encode(texts_p, show_progress_bar=True, batch_size=64)
        emb_r = model.encode(texts_r, show_progress_bar=True, batch_size=64)
        sims = [
            float(sk_cos([ep], [er])[0][0])
            for ep, er in zip(emb_p, emb_r)
        ]
        return sims

    except ImportError:
        log.warning("sentence-transformers not installed — using TF-IDF cosine similarity.")

    # --- TF-IDF fallback ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos

        corpus = [normalise(t) or "<empty>" for t in preds + refs]
        vec = TfidfVectorizer().fit(corpus)
        vp = vec.transform([normalise(p) or "<empty>" for p in preds])
        vr = vec.transform([normalise(r) or "<empty>" for r in refs])
        sims = [float(sk_cos(vp[i], vr[i])[0][0]) for i in range(len(preds))]
        return sims
    except ImportError:
        log.warning("scikit-learn not installed — skipping embedding similarity.")
        return [float("nan")] * len(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Token-level F1 (recall/precision on PlantUML tokens)
# ─────────────────────────────────────────────────────────────────────────────

def token_f1(pred: str, ref: str) -> tuple[float, float, float]:
    pred_toks = set(normalise(pred).split())
    ref_toks  = set(normalise(ref).split())
    if not pred_toks or not ref_toks:
        return 0.0, 0.0, 0.0
    tp = len(pred_toks & ref_toks)
    precision = tp / len(pred_toks)
    recall    = tp / len(ref_toks)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# Per-row metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    log.info(f"Computing metrics for: {label}  (n={len(df)})")

    preds = df["uml_pred"].fillna("").tolist()
    refs  = df["uml_code_gt"].fillna("").tolist()

    em     = [exact_match(p, r) for p, r in zip(preds, refs)]
    bleu   = compute_bleu_scores(preds, refs)
    cb     = compute_codebleu_scores(preds, refs)
    emb    = compute_embedding_similarity(preds, refs)
    tf1    = [token_f1(p, r) for p, r in zip(preds, refs)]

    result = df[["uml_code_gt", "uml_pred", "uml_valid", "uml_error", "llm_failed"]].copy()
    result["model"]         = label
    result["exact_match"]   = em
    result["bleu"]          = bleu
    result["codebleu"]      = cb
    result["emb_sim"]       = emb
    result["token_prec"]    = [t[0] for t in tf1]
    result["token_recall"]  = [t[1] for t in tf1]
    result["token_f1"]      = [t[2] for t in tf1]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {"base": "#6B8EAD", "finetuned": "#E07B54"}

def _save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Saved → {out}")


def plot_validity_bar(base_df: pd.DataFrame, ft_df: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    labels   = ["LLM failures", "Invalid", "Valid"]
    b_vals   = [
        base_df["llm_failed"].sum(),
        (~base_df["uml_valid"] & ~base_df["llm_failed"]).sum(),
        base_df["uml_valid"].sum(),
    ]
    ft_vals  = [
        ft_df["llm_failed"].sum(),
        (~ft_df["uml_valid"] & ~ft_df["llm_failed"]).sum(),
        ft_df["uml_valid"].sum(),
    ]
    n = len(base_df)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_b  = ax.bar(x - w/2, [v/n*100 for v in b_vals],  w, label="Base",      color=PALETTE["base"])
    bars_ft = ax.bar(x + w/2, [v/n*100 for v in ft_vals], w, label="Finetuned", color=PALETTE["finetuned"])

    for bar in list(bars_b) + list(bars_ft):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("PlantUML Validity: Base vs Finetuned")
    ax.legend()
    ax.set_ylim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig1_validity.png")
    plt.close(fig)


def plot_metric_bars(combined: pd.DataFrame, out: Path):
    """Bar chart comparing mean metric values for base vs finetuned."""
    import matplotlib.pyplot as plt

    metrics = {
        "Exact Match":       "exact_match",
        "BLEU":              "bleu",
        "CodeBLEU":          "codebleu",
        "Embedding Sim.":    "emb_sim",
        "Token F1":          "token_f1",
    }

    # drop all-NaN metrics
    valid_metrics = {k: v for k, v in metrics.items()
                     if not combined[v].isna().all()}

    n_m = len(valid_metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(3.2 * n_m, 4.5), sharey=False)
    if n_m == 1:
        axes = [axes]

    for ax, (label, col) in zip(axes, valid_metrics.items()):
        means = combined.groupby("model")[col].mean()
        b_val  = means.get("base",      0.0)
        ft_val = means.get("finetuned", 0.0)
        bars = ax.bar(["Base", "Finetuned"], [b_val, ft_val],
                      color=[PALETTE["base"], PALETTE["finetuned"]], width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, min(1.15, max(b_val, ft_val) * 1.3 + 0.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Code Similarity Metrics: Base vs Finetuned", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig2_metrics_bar.png")
    plt.close(fig)


def plot_metric_distributions(combined: pd.DataFrame, out: Path):
    """Histogram plots showing score distributions (robust to constant/near-constant data)."""
    import matplotlib.pyplot as plt

    metrics = {
        "BLEU":           "bleu",
        "CodeBLEU":       "codebleu",
        "Embedding Sim.": "emb_sim",
        "Token F1":       "token_f1",
    }
    valid_metrics = {k: v for k, v in metrics.items()
                     if not combined[v].isna().all()}

    n_m = len(valid_metrics)
    if n_m == 0:
        return
    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4.5))
    if n_m == 1:
        axes = [axes]

    for ax, (label, col) in zip(axes, valid_metrics.items()):
        for model, color in PALETTE.items():
            data = combined.loc[combined["model"] == model, col].dropna()
            if len(data) == 0:
                continue
            std = data.std()
            if std > 1e-6:
                # Try KDE first
                try:
                    data.plot.kde(ax=ax, label=model.capitalize(), color=color, linewidth=2)
                except Exception:
                    ax.hist(data, bins=20, alpha=0.5, color=color,
                            label=model.capitalize(), density=True)
            else:
                # Constant data — just draw a vertical line at the single value
                ax.axvline(data.iloc[0], color=color, linewidth=2,
                           label=f"{model.capitalize()} (={data.iloc[0]:.3f})")
            ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Score Distributions: Base vs Finetuned", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig3_distributions.png")
    plt.close(fig)


def plot_scatter_emb_vs_bleu(combined: pd.DataFrame, out: Path):
    """Scatter: embedding similarity vs BLEU, coloured by model."""
    import matplotlib.pyplot as plt

    if combined["emb_sim"].isna().all() or combined["bleu"].isna().all():
        log.warning("Skipping scatter plot — metrics not available.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for model, color in PALETTE.items():
        sub = combined[combined["model"] == model]
        ax.scatter(sub["bleu"], sub["emb_sim"],
                   alpha=0.35, s=18, color=color, label=model.capitalize())
    ax.set_xlabel("BLEU")
    ax.set_ylabel("Embedding Similarity")
    ax.set_title("BLEU vs Embedding Similarity")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig4_scatter_bleu_emb.png")
    plt.close(fig)


def plot_valid_and_correct(combined: pd.DataFrame, out: Path):
    """
    Stacked bar: for each model show what fraction is
    invalid | valid but low-similarity | valid and high-similarity
    (threshold = median CodeBLEU of finetuned valid predictions).
    """
    import matplotlib.pyplot as plt

    # pick best available metric
    for col in ("codebleu", "bleu", "token_f1", "emb_sim"):
        if not combined[col].isna().all():
            sim_col = col
            break
    else:
        log.warning("No similarity metric available for stacked bar.")
        return

    # threshold = median of valid finetuned predictions
    valid_ft = combined.loc[
        (combined["model"] == "finetuned") & combined["uml_valid"], sim_col
    ].dropna()
    threshold = valid_ft.median() if len(valid_ft) else 0.5
    log.info(f"Correctness threshold ({sim_col}): {threshold:.3f}")

    categories = ["Invalid", "Valid but Imprecise", "Valid & Correct"]
    colors     = ["#d9534f", "#f0ad4e", "#5cb85c"]
    results    = {}

    for model in ("base", "finetuned"):
        sub = combined[combined["model"] == model]
        n   = len(sub)
        invalid = (~sub["uml_valid"]).sum()
        valid   = sub[sub["uml_valid"]]
        correct = (valid[sim_col].fillna(0) >= threshold).sum()
        imprecise = len(valid) - correct
        results[model] = [invalid/n*100, imprecise/n*100, correct/n*100]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)
    bottoms = np.zeros(2)
    models = list(results.keys())
    for i, (cat, color) in enumerate(zip(categories, colors)):
        vals = [results[m][i] for m in models]
        bars = ax.bar(x, vals, bottom=bottoms, color=color, label=cat, width=0.5)
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + h/2,
                        f"{h:.1f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models])
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Validity & Correctness (threshold={threshold:.2f} {sim_col})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig5_valid_and_correct.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(combined: pd.DataFrame):
    metrics = ["exact_match", "bleu", "codebleu", "emb_sim",
               "token_prec", "token_recall", "token_f1"]
    available = [m for m in metrics if not combined[m].isna().all()]

    rows = []
    for model in ("base", "finetuned"):
        sub = combined[combined["model"] == model]
        n   = len(sub)
        row = {
            "model":        model,
            "n":            n,
            "valid_%":      f"{sub['uml_valid'].mean()*100:.1f}",
        }
        for m in available:
            row[m] = f"{sub[m].mean():.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    print("\n" + "=" * 80)
    print("METRIC SUMMARY")
    print("=" * 80)
    print(summary.T.to_string())
    print("=" * 80 + "\n")

    # delta row
    print("Δ (finetuned − base):")
    base_row = combined[combined["model"] == "base"]
    ft_row   = combined[combined["model"] == "finetuned"]
    for m in available:
        delta = ft_row[m].mean() - base_row[m].mean()
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {m:<18} {arrow} {delta:+.4f}")
    print()

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark similarity analysis")
    parser.add_argument("--base",          required=True, help="benchmark_results_base.parquet")
    parser.add_argument("--finetuned",     required=True, help="benchmark_results_finetuned.parquet")
    parser.add_argument("--ground-truth",  required=True,
                        help="qwen_lora_test_split.parquet with uml_code column")
    parser.add_argument("--output-dir", default="./analysis_output",
                        help="Directory to save figures and CSV (default: ./analysis_output)")
    parser.add_argument("--no-plots",   action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── load ground truth ─────────────────────────────────────────────────────
    log.info(f"Loading ground truth:        {args.ground_truth}")
    gt_df = pd.read_parquet(args.ground_truth)
    log.info(f"Ground truth: {len(gt_df)} rows | columns: {gt_df.columns.tolist()}")
    if "uml_code" not in gt_df.columns:
        raise ValueError(f"'uml_code' column not found in ground truth. Available: {gt_df.columns.tolist()}")
    gt_codes = gt_df["uml_code"].fillna("").reset_index(drop=True)

    # ── load benchmarks and inject ground truth ───────────────────────────────
    log.info(f"Loading base benchmark:      {args.base}")
    base_df = pd.read_parquet(args.base).reset_index(drop=True)
    log.info(f"Loading finetuned benchmark: {args.finetuned}")
    ft_df   = pd.read_parquet(args.finetuned).reset_index(drop=True)

    log.info(f"Base:      {len(base_df)} rows | columns: {base_df.columns.tolist()}")
    log.info(f"Finetuned: {len(ft_df)} rows | columns: {ft_df.columns.tolist()}")

    # Overwrite uml_code_gt from the authoritative ground truth file
    for df, name in [(base_df, "base"), (ft_df, "finetuned")]:
        if len(df) != len(gt_codes):
            raise ValueError(
                f"{name} benchmark has {len(df)} rows but ground truth has {len(gt_codes)} rows. "
                "They must be the same test split in the same order."
            )
        df["uml_code_gt"] = gt_codes.values

    # Sanity-check: show a sample prediction vs ground truth
    sample_pred = base_df["uml_pred"].iloc[0][:120] if "uml_pred" in base_df.columns else ""
    sample_gt   = base_df["uml_code_gt"].iloc[0][:120]
    log.info(f"Sample GT  : {sample_gt!r}")
    log.info(f"Sample pred: {sample_pred!r}")

    # ── metrics ───────────────────────────────────────────────────────────────
    base_metrics = compute_all_metrics(base_df, "base")
    ft_metrics   = compute_all_metrics(ft_df,   "finetuned")
    combined     = pd.concat([base_metrics, ft_metrics], ignore_index=True)

    # save detailed CSV
    csv_path = out / "metrics_detail.csv"
    combined.drop(columns=["uml_code_gt", "uml_pred"], errors="ignore").to_csv(csv_path, index=False)
    log.info(f"Detailed metrics saved → {csv_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    print_summary_table(combined)

    if args.no_plots:
        return

    # ── figures ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless / cluster safe

        plot_validity_bar(base_df, ft_df, out)
        plot_metric_bars(combined, out)
        plot_metric_distributions(combined, out)
        plot_scatter_emb_vs_bleu(combined, out)
        plot_valid_and_correct(combined, out)
        log.info(f"All figures saved to {out}/")
    except ImportError:
        log.warning("matplotlib not installed — skipping figures. pip install matplotlib")


if __name__ == "__main__":
    main()

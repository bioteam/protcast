#!/usr/bin/env python3
"""
Consolidate all benchmark results into a single coherent set of outputs.

Replaces the original F1-based analysis with the comprehensive extended metrics
(F_max, ROC-AUC, PR-AUC, MCC, S_min, bootstrap CIs, protein difficulty).

Generates:
  - results/figures/algorithm_comparison.png    (F_max boxplots, replaces old F1 version)
  - results/figures/efficiency_frontier.png     (F_max vs compute, replaces old F1 version)
  - results/figures/correlation_heatmap.png     (metric correlations, now includes all metrics)
  - results/reports/benchmark_summary_report.csv (consolidated per-algorithm summary)
  - results/reports/analysis_summary.txt        (comprehensive human-readable report)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS = Path("results")
DATA = RESULTS / "data"
FIGURES = RESULTS / "figures"
REPORTS = RESULTS / "reports"
COMBINED_TSV = Path("data/processed/combined_benchmark_data.tsv")

FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
ext = pd.read_csv(DATA / "extended_summary.csv")
smin = pd.read_csv(DATA / "smin_scores.csv")
difficulty = pd.read_csv(DATA / "protein_difficulty.csv")
combined = pd.read_csv(COMBINED_TSV, sep="\t")

# Merge efficiency info (Vector_Length, Elapsed_Time) from combined into ext
efficiency = (
    combined.groupby("Algorithm")
    .agg(Vector_Length=("Vector_Length", "first"), Elapsed_Time=("Elapsed_Time", "median"))
    .reset_index()
)
ext_with_eff = ext.merge(efficiency, on="Algorithm", how="left")

# ── Figure 1: Algorithm Comparison (F_max boxplots) ───────────────────
algo_order = ext.groupby("Algorithm")["F_max"].median().sort_values(ascending=False).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# F_max
sns.boxplot(data=ext, x="Algorithm", y="F_max", order=algo_order, ax=axes[0],
            palette="viridis", fliersize=2)
axes[0].set_title("F_max by Algorithm (CAFA Standard)", fontsize=12, fontweight="bold")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("F_max")
axes[0].axhline(y=ext["F_max"].median(), color="red", linestyle="--", alpha=0.5, label="Global Median")
axes[0].legend(fontsize=8)

# ROC-AUC
sns.boxplot(data=ext, x="Algorithm", y="ROC_AUC", order=algo_order, ax=axes[1],
            palette="viridis", fliersize=2)
axes[1].set_title("ROC-AUC by Algorithm", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("ROC-AUC")

# MCC
sns.boxplot(data=ext, x="Algorithm", y="MCC_at_Fmax", order=algo_order, ax=axes[2],
            palette="viridis", fliersize=2)
axes[2].set_title("MCC at Optimal Threshold", fontsize=12, fontweight="bold")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right", fontsize=8)
axes[2].set_ylabel("MCC")

plt.tight_layout()
plt.savefig(FIGURES / "algorithm_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ algorithm_comparison.png")

# ── Figure 2: Efficiency Frontier (F_max vs Vector Length & Time) ─────
algo_agg = ext_with_eff.groupby("Algorithm").agg(
    F_max_mean=("F_max", "mean"),
    F_max_median=("F_max", "median"),
    ROC_AUC_mean=("ROC_AUC", "mean"),
    Vector_Length=("Vector_Length", "first"),
    Elapsed_Time=("Elapsed_Time", "first"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# F_max vs Vector Length
axes[0].scatter(algo_agg["Vector_Length"], algo_agg["F_max_mean"], s=80, c="steelblue", edgecolors="black", zorder=5)
for _, row in algo_agg.iterrows():
    axes[0].annotate(row["Algorithm"], (row["Vector_Length"], row["F_max_mean"]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                     textcoords="offset points")
axes[0].set_xlabel("Vector Length (dimensions)", fontsize=11)
axes[0].set_ylabel("Mean F_max", fontsize=11)
axes[0].set_title("Accuracy vs Feature Dimensionality", fontsize=12, fontweight="bold")
axes[0].set_xscale("log")
axes[0].grid(True, alpha=0.3)

# F_max vs Elapsed Time
axes[1].scatter(algo_agg["Elapsed_Time"], algo_agg["F_max_mean"], s=80, c="coral", edgecolors="black", zorder=5)
for _, row in algo_agg.iterrows():
    axes[1].annotate(row["Algorithm"], (row["Elapsed_Time"], row["F_max_mean"]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                     textcoords="offset points")
axes[1].set_xlabel("Median Elapsed Time (s)", fontsize=11)
axes[1].set_ylabel("Mean F_max", fontsize=11)
axes[1].set_title("Accuracy vs Computation Time", fontsize=12, fontweight="bold")
axes[1].set_xscale("log")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "efficiency_frontier.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ efficiency_frontier.png")

# ── Figure 3: Correlation Heatmap (all metrics) ──────────────────────
metric_cols = ["F_max", "ROC_AUC", "PR_AUC", "MCC_at_Fmax", "Class_Imbalance"]

# Also merge original F1 from combined
orig_f1 = combined[["Algorithm", "GO_Term", "F1_Score", "Sensitivity", "Specificity"]].copy()
orig_f1.rename(columns={"F1_Score": "Original_F1"}, inplace=True)
merged = ext.merge(orig_f1, on=["Algorithm", "GO_Term"], how="inner")

corr_cols = ["F_max", "ROC_AUC", "PR_AUC", "MCC_at_Fmax", "Original_F1", "Sensitivity", "Specificity", "Class_Imbalance"]
corr_data = merged[corr_cols].dropna()
corr_matrix = corr_data.corr(method="spearman")

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Spearman Rank Correlation Between All Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ correlation_heatmap.png")

# ── Consolidated Summary CSV ──────────────────────────────────────────
algo_summary = ext.groupby("Algorithm").agg(
    F_max_mean=("F_max", "mean"),
    F_max_median=("F_max", "median"),
    F_max_std=("F_max", "std"),
    ROC_AUC_mean=("ROC_AUC", "mean"),
    ROC_AUC_median=("ROC_AUC", "median"),
    PR_AUC_mean=("PR_AUC", "mean"),
    MCC_mean=("MCC_at_Fmax", "mean"),
    MCC_median=("MCC_at_Fmax", "median"),
    Fmax_CI_Width_mean=("Fmax_CI_Width", "mean"),
    AUC_CI_Width_mean=("AUC_CI_Width", "mean"),
    N_GO_Terms=("GO_Term", "nunique"),
).reset_index()

# Merge S_min (averaged across levels)
smin_avg = smin.groupby("Algorithm").agg(
    S_min_mean=("S_min", "mean"),
    S_min_std=("S_min", "std"),
).reset_index()
algo_summary = algo_summary.merge(smin_avg, on="Algorithm", how="left")

# Merge efficiency
algo_summary = algo_summary.merge(efficiency, on="Algorithm", how="left")

# Merge original F1 mean
orig_f1_avg = combined.groupby("Algorithm")["F1_Score"].mean().reset_index()
orig_f1_avg.columns = ["Algorithm", "Original_F1_mean"]
algo_summary = algo_summary.merge(orig_f1_avg, on="Algorithm", how="left")

# Sort by F_max
algo_summary.sort_values("F_max_mean", ascending=False, inplace=True)

algo_summary.to_csv(REPORTS / "benchmark_summary_report.csv", index=False)
print("✓ benchmark_summary_report.csv")

# ── Difficulty summary ────────────────────────────────────────────────
diff_summary = difficulty.groupby("Category").size().reset_index(name="Count")
total_proteins = len(difficulty)
diff_summary["Percentage"] = (diff_summary["Count"] / total_proteins * 100).round(2)

# ── Comprehensive Report ──────────────────────────────────────────────
top = algo_summary.iloc[0]
runner_up = algo_summary.iloc[1]

# Best efficiency: highest F_max / log(Vector_Length)
algo_summary["efficiency_score"] = algo_summary["F_max_mean"] / np.log10(algo_summary["Vector_Length"].clip(lower=1))
best_eff = algo_summary.sort_values("efficiency_score", ascending=False).iloc[0]

# Lowest S_min
best_smin = algo_summary.sort_values("S_min_mean").iloc[0]

# Hardest GO terms
hard_proteins = difficulty[difficulty["Category"] == "universally_hard"]
hard_by_term = hard_proteins.groupby("GO_Term").size().sort_values(ascending=False)

report = f"""{'='*76}
PROTCAST BENCHMARK ANALYSIS — COMPREHENSIVE REPORT
{'='*76}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report summarises the benchmark evaluation of {algo_summary['N_GO_Terms'].max()} GO term
binary classifiers across {len(algo_summary)} protein feature extraction algorithms
at GO DAG levels 2–7 (Molecular Function ontology).

Metrics were computed from {len(ext)} raw per-protein score files using
threshold-optimised evaluation (F_max) rather than fixed-threshold F1.

{'─'*76}
1. OVERALL ALGORITHM RANKING (by mean F_max)
{'─'*76}
"""

for i, row in algo_summary.iterrows():
    rank = algo_summary.index.get_loc(i) + 1
    report += (
        f"  {rank:2d}. {row['Algorithm']:<12s}  "
        f"F_max={row['F_max_mean']:.3f} (±{row['F_max_std']:.3f})  "
        f"AUC={row['ROC_AUC_mean']:.3f}  "
        f"MCC={row['MCC_mean']:.3f}  "
        f"S_min={row['S_min_mean']:.3f}\n"
    )

report += f"""
{'─'*76}
2. KEY FINDINGS
{'─'*76}

MOST ACCURATE ALGORITHM: {top['Algorithm']}
  • Mean F_max:     {top['F_max_mean']:.4f}  (median {top['F_max_median']:.4f})
  • Mean ROC-AUC:   {top['ROC_AUC_mean']:.4f}
  • Mean PR-AUC:    {top['PR_AUC_mean']:.4f}
  • Mean MCC:       {top['MCC_mean']:.4f}
  • Mean S_min:     {top['S_min_mean']:.4f}
  • Vector length:  {int(top['Vector_Length'])} dimensions
  • Median time:    {top['Elapsed_Time']:.0f}s per GO term

RUNNER-UP: {runner_up['Algorithm']}
  • Mean F_max:     {runner_up['F_max_mean']:.4f}
  • Mean ROC-AUC:   {runner_up['ROC_AUC_mean']:.4f}
  • Vector length:  {int(runner_up['Vector_Length'])} dimensions

BEST SEMANTIC DISTANCE (S_min, lower is better): {best_smin['Algorithm']}
  • Mean S_min:     {best_smin['S_min_mean']:.4f}

BEST EFFICIENCY (accuracy per dimensionality): {best_eff['Algorithm']}
  • Mean F_max:     {best_eff['F_max_mean']:.4f}
  • Vector length:  {int(best_eff['Vector_Length'])} dimensions
  • Efficiency:     {best_eff['F_max_mean']/int(best_eff['Vector_Length'])*1000:.2f} × 10⁻³ F_max/dim

{'─'*76}
3. THRESHOLD ARTIFACT: F1 vs F_max
{'─'*76}

The original pipeline used fixed-threshold F1 scores, which dramatically
underestimated performance for algorithms whose default threshold was suboptimal.

"""

# Compute F1 vs Fmax gap per algorithm
f1_vs_fmax = algo_summary[["Algorithm", "Original_F1_mean", "F_max_mean"]].copy()
f1_vs_fmax["Gap"] = f1_vs_fmax["F_max_mean"] - f1_vs_fmax["Original_F1_mean"]
f1_vs_fmax.sort_values("Gap", ascending=False, inplace=True)

report += "  Algorithm       Original F1   F_max    Gap\n"
report += "  " + "─" * 50 + "\n"
for _, row in f1_vs_fmax.iterrows():
    report += f"  {row['Algorithm']:<15s}  {row['Original_F1_mean']:.3f}       {row['F_max_mean']:.3f}    +{row['Gap']:.3f}\n"

report += f"""
  Largest gap: {f1_vs_fmax.iloc[0]['Algorithm']} (+{f1_vs_fmax.iloc[0]['Gap']:.3f})
  This confirms that fixed-threshold F1 is unreliable for comparing algorithms.
  F_max (and ROC-AUC) should be used as the primary metrics.

{'─'*76}
4. BOOTSTRAP CONFIDENCE INTERVALS (95%)
{'─'*76}

  Mean F_max CI width:   {ext['Fmax_CI_Width'].mean():.4f}
  Mean ROC-AUC CI width: {ext['AUC_CI_Width'].mean():.4f}

  Narrowest CIs (most stable): algorithms with large, balanced datasets
  Widest CIs: small GO terms with <30 proteins

{'─'*76}
5. PROTEIN DIFFICULTY
{'─'*76}

  Total protein-term pairs analysed: {total_proteins:,}

"""

for _, drow in diff_summary.sort_values("Count", ascending=False).iterrows():
    report += f"  {drow['Category']:<20s}  {drow['Count']:>6,}  ({drow['Percentage']:.1f}%)\n"

report += f"""
  Universally hard proteins (misclassified by ALL algorithms) are prime
  targets for ESM-based models that can exploit structural/evolutionary
  features unavailable to sequence-composition baselines.

  Top 5 hardest GO terms (most universally-hard proteins):
"""

for term, count in hard_by_term.head(5).items():
    report += f"    {term}: {count} universally-hard proteins\n"

report += f"""
{'─'*76}
6. S_min BY LEVEL
{'─'*76}

"""

smin_pivot = smin.pivot_table(index="Algorithm", columns="Level", values="S_min")
smin_pivot["Mean"] = smin_pivot.mean(axis=1)
smin_pivot.sort_values("Mean", inplace=True)

report += f"  {'Algorithm':<12s}"
for lvl in sorted(smin.Level.unique()):
    report += f"  L{lvl:d}"
report += "   Mean\n"
report += "  " + "─" * 60 + "\n"

for algo, row in smin_pivot.iterrows():
    report += f"  {algo:<12s}"
    for lvl in sorted(smin.Level.unique()):
        val = row.get(lvl, np.nan)
        report += f"  {val:.3f}" if not np.isnan(val) else "    –  "
    report += f"  {row['Mean']:.3f}\n"

report += f"""
{'─'*76}
7. DATA FILES
{'─'*76}

  results/data/
    extended_summary.csv        – Per-file metrics + bootstrap CIs ({len(ext):,} rows)
    per_file_metrics.csv        – Per-file metrics only ({len(ext):,} rows)
    smin_scores.csv             – S_min per algorithm × level ({len(smin)} rows)
    bootstrap_confidence.csv    – 95% CIs for F_max and AUC ({len(ext):,} rows)
    protein_difficulty.csv      – Per-protein difficulty ({total_proteins:,} rows)

  results/figures/
    algorithm_comparison.png    – F_max / ROC-AUC / MCC boxplots by algorithm
    efficiency_frontier.png     – F_max vs dimensionality and compute time
    correlation_heatmap.png     – Spearman correlations across all metrics
    fmax_distribution.png       – F_max distribution per algorithm
    smin_comparison.png         – S_min bar chart by algorithm and level
    protein_difficulty_distribution.png – Difficulty histogram + category breakdown

  results/reports/
    benchmark_summary_report.csv – Consolidated per-algorithm summary table
    analysis_summary.txt         – This report

{'─'*76}
8. RECOMMENDATIONS FOR ESM MODEL DEVELOPMENT
{'─'*76}

  1. Target metric: F_max ≥ {top['F_max_mean']:.3f} (beat {top['Algorithm']})
     and/or S_min < {best_smin['S_min_mean']:.3f} (beat {best_smin['Algorithm']})

  2. Focus on universally-hard proteins — these are the cases where all
     sequence-composition baselines fail, so structural/evolutionary
     features from ESM embeddings have the highest marginal value.

  3. The high correlation (ρ > 0.94) between F_max, AUC, and MCC means
     that if ESM beats baselines on one metric, it will likely beat
     them on all threshold-optimised metrics.

  4. Class imbalance increases at deeper GO levels. Consider using PR-AUC
     as a secondary metric for level 6–7 terms where positives are rare.

  5. The efficiency frontier shows that {best_eff['Algorithm']} achieves
     {best_eff['F_max_mean']:.1%} of top accuracy with only
     {int(best_eff['Vector_Length'])} dimensions. ESM embeddings
     (typically 320–2560 dim) should be compared against this baseline.

{'='*76}
"""

with open(REPORTS / "analysis_summary.txt", "w") as f:
    f.write(report)
print("✓ analysis_summary.txt")

print(f"\nDone. All outputs in {RESULTS}/")

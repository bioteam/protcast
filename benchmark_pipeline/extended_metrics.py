#!/usr/bin/env python3
"""
Extended Benchmark Metrics for ProtCast GO Term Prediction

Computes 8 additional metrics from raw per-protein scores files:
1. ROC-AUC (threshold-independent ranking quality)
2. PR-AUC / Average Precision (better for imbalanced data)
3. F_max (CAFA standard: max F1 across all thresholds)
4. MCC at optimal threshold (chance-corrected accuracy)
5. Class imbalance ratio per GO term
6. S_min (CAFA semantic distance using Information Content)
7. Bootstrap 95% confidence intervals for F_max and ROC-AUC
8. Per-protein difficulty scores (cross-algorithm agreement)

Usage:
    python extended_metrics.py \\
        --scores-dir ../ProtCast_stats \\
        --obo ../ProtCast_stats/2025-10-10/go.obo \\
        --gaf ../ProtCast_stats/2025-10-10/filtered_goa_uniprot_all_noiea.gaf \\
        --output results \\
        --workers 4 \\
        --skip-bootstrap
"""

import argparse
import logging
import re
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# 1. ScoresFileLoader
# ---------------------------------------------------------------------------

class ScoresFileLoader:
    """Discovers and loads *_scores.tsv files from ProtCast_stats/level-{2..7}/."""

    def __init__(self, scores_dir, levels=None):
        self.scores_dir = Path(scores_dir)
        self.levels = levels or [2, 3, 4, 5, 6, 7]
        self.index = {}  # (level, go_term, algorithm) -> Path

    def build_index(self):
        """Scan all level directories and index scores files."""
        pattern = re.compile(r'^(GO:\d+)_(\w+)_scores\.tsv$')
        for level in self.levels:
            level_dir = self.scores_dir / f'level-{level}'
            if not level_dir.exists():
                logger.warning(f"Directory not found: {level_dir}")
                continue
            for f in level_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    go_term, algorithm = m.group(1), m.group(2)
                    self.index[(level, go_term, algorithm)] = f

        logger.info(f"Indexed {len(self.index)} scores files across {len(self.levels)} levels")
        return self.index

    def load_scores_file(self, path):
        """Load a single scores file. Returns (y_true, scores, protein_ids)."""
        labels, proteins, scores = [], [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                label_str, protein_id, score_str = parts[0], parts[1], parts[2]
                is_positive = 1 if label_str.startswith('GO:') else 0
                labels.append(is_positive)
                proteins.append(protein_id)
                scores.append(float(score_str))
        return np.array(labels), np.array(scores), proteins

    def get_go_terms_for_level(self, level):
        """Return set of GO terms available at a given level."""
        return {go for (lv, go, _) in self.index if lv == level}

    def get_algorithms_for_level_term(self, level, go_term):
        """Return list of algorithms available for a given level+term."""
        return [algo for (lv, go, algo) in self.index if lv == level and go == go_term]


# ---------------------------------------------------------------------------
# 2. GOOntologyParser (lightweight, no goatools dependency)
# ---------------------------------------------------------------------------

class GOOntologyParser:
    """Parses GO OBO + GAF files to compute Information Content for MF terms."""

    def __init__(self):
        self.parents = {}       # go_id -> set of parent go_ids
        self.children = defaultdict(set)
        self.mf_terms = set()
        self.ic_values = {}

    def parse_obo(self, obo_path):
        """Parse OBO file, extract MF terms and is_a relationships."""
        logger.info(f"Parsing OBO file: {obo_path}")
        current = None
        with open(obo_path) as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    current = {'parents': set(), 'is_obsolete': False}
                elif line == '[Typedef]':
                    current = None
                elif line == '' and current is not None:
                    if (current.get('namespace') == 'molecular_function'
                            and not current['is_obsolete']
                            and 'id' in current):
                        go_id = current['id']
                        self.mf_terms.add(go_id)
                        self.parents[go_id] = current['parents']
                        for parent in current['parents']:
                            self.children[parent].add(go_id)
                    current = None
                elif current is not None:
                    if line.startswith('id: '):
                        current['id'] = line[4:]
                    elif line.startswith('namespace: '):
                        current['namespace'] = line[11:]
                    elif line.startswith('is_obsolete: true'):
                        current['is_obsolete'] = True
                    elif line.startswith('is_a: '):
                        parent_id = line.split()[1]
                        current['parents'].add(parent_id)

        logger.info(f"Parsed {len(self.mf_terms)} MF terms")

    def parse_gaf_and_compute_ic(self, gaf_path):
        """Parse GAF file, propagate annotations, compute IC values."""
        logger.info(f"Parsing GAF file: {gaf_path}")

        # Step 1: Collect direct annotations (protein -> GO term) for MF only
        direct_annotations = defaultdict(set)  # go_term -> set of protein_ids
        total_proteins = set()

        with open(gaf_path) as f:
            for line in f:
                if line.startswith('!'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                aspect = parts[8]
                if aspect != 'F':
                    continue
                protein_id = parts[1]
                go_id = parts[4]
                if go_id in self.mf_terms:
                    direct_annotations[go_id].add(protein_id)
                    total_proteins.add(protein_id)

        n_total = len(total_proteins)
        logger.info(f"Found {sum(len(v) for v in direct_annotations.values())} MF annotations "
                     f"for {len(direct_annotations)} terms across {n_total} proteins")

        # Step 2: Propagate protein sets bottom-up (true path rule)
        # Each term's propagated set = its own proteins + all descendants' proteins
        propagated = {}  # go_term -> set of protein_ids

        def get_propagated(term):
            if term in propagated:
                return propagated[term]
            proteins = set(direct_annotations.get(term, set()))
            for child in self.children.get(term, set()):
                if child in self.mf_terms:
                    proteins |= get_propagated(child)
            propagated[term] = proteins
            return proteins

        # Find root MF terms (no parents within MF, or parent is GO:0003674)
        for term in self.mf_terms:
            get_propagated(term)

        # Step 3: Compute IC
        for term in self.mf_terms:
            count = len(propagated.get(term, set()))
            if count > 0 and n_total > 0:
                freq = count / n_total
                self.ic_values[term] = -math.log2(freq)
            else:
                self.ic_values[term] = 0.0

        # Root term (GO:0003674) should have IC ~= 0
        n_with_ic = sum(1 for v in self.ic_values.values() if v > 0)
        max_ic = max(self.ic_values.values()) if self.ic_values else 0
        logger.info(f"Computed IC for {n_with_ic} terms (max IC = {max_ic:.2f})")


# ---------------------------------------------------------------------------
# 3. PerFileMetrics (ROC-AUC, PR-AUC, F_max, MCC, class imbalance)
# ---------------------------------------------------------------------------

def compute_fmax(y_true, scores):
    """Compute F_max (maximum F1 across all thresholds) and optimal threshold."""
    # Use unique scores as thresholds for exact computation
    thresholds = np.unique(scores)
    if len(thresholds) > 200:
        thresholds = np.linspace(scores.min(), scores.max(), 201)

    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_f1, best_thresh


class PerFileMetrics:
    """Computes per-scores-file metrics (1-5)."""

    def compute_all(self, loader):
        """Compute ROC-AUC, PR-AUC, F_max, MCC, class imbalance for every file."""
        rows = []
        total = len(loader.index)
        for i, ((level, go_term, algorithm), path) in enumerate(sorted(loader.index.items())):
            if (i + 1) % 500 == 0:
                logger.info(f"  Processing file {i+1}/{total}...")

            y_true, scores, proteins = loader.load_scores_file(path)
            n_pos = int(y_true.sum())
            n_neg = len(y_true) - n_pos
            n_total = len(y_true)

            # Skip degenerate cases (all one class)
            if n_pos == 0 or n_neg == 0:
                continue

            # ROC-AUC
            try:
                roc = roc_auc_score(y_true, scores)
            except ValueError:
                roc = np.nan

            # PR-AUC
            try:
                pr = average_precision_score(y_true, scores)
            except ValueError:
                pr = np.nan

            # F_max
            fmax, fmax_thresh = compute_fmax(y_true, scores)

            # MCC at F_max threshold
            y_pred_at_fmax = (scores >= fmax_thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_at_fmax)

            rows.append({
                'Level': level,
                'GO_Term': go_term,
                'Algorithm': algorithm,
                'ROC_AUC': round(roc, 6),
                'PR_AUC': round(pr, 6),
                'F_max': round(fmax, 6),
                'F_max_Threshold': round(fmax_thresh, 4),
                'MCC_at_Fmax': round(mcc, 6),
                'Class_Imbalance': round(n_pos / n_total, 4),
                'N_Positive': n_pos,
                'N_Negative': n_neg,
                'N_Total': n_total,
            })

        df = pd.DataFrame(rows)
        logger.info(f"Computed per-file metrics for {len(df)} files")
        return df


# ---------------------------------------------------------------------------
# 4. SminCalculator (CAFA semantic distance)
# ---------------------------------------------------------------------------

class SminCalculator:
    """Computes S_min using IC-weighted remaining uncertainty and misinformation."""

    def __init__(self, ontology, loader):
        self.ontology = ontology
        self.loader = loader

    def compute_smin_for_algo_level(self, level, algorithm):
        """Compute S_min for one algorithm at one level."""
        # Collect all GO terms this algorithm has scores for at this level
        go_terms = []
        data = {}  # go_term -> (y_true, scores, proteins)
        for (lv, go, algo), path in self.loader.index.items():
            if lv == level and algo == algorithm:
                ic = self.ontology.ic_values.get(go, 0.0)
                if ic <= 0:
                    continue
                y_true, scores, proteins = self.loader.load_scores_file(path)
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    continue
                go_terms.append(go)
                data[go] = (y_true, scores, proteins)

        if len(go_terms) < 2:
            return None

        # Collect all unique proteins across all GO terms
        all_proteins = set()
        for go in go_terms:
            all_proteins.update(data[go][2])
        protein_list = sorted(all_proteins)
        protein_idx = {p: i for i, p in enumerate(protein_list)}
        n_proteins = len(protein_list)
        n_terms = len(go_terms)

        # Build matrices: truth[protein, term] and score[protein, term]
        truth = np.full((n_proteins, n_terms), -1, dtype=np.int8)  # -1 = not evaluated
        score_matrix = np.full((n_proteins, n_terms), np.nan)

        for j, go in enumerate(go_terms):
            y_true, scores, proteins = data[go]
            for k, pid in enumerate(proteins):
                idx = protein_idx[pid]
                truth[idx, j] = y_true[k]
                score_matrix[idx, j] = scores[k]

        ic_array = np.array([self.ontology.ic_values.get(go, 0.0) for go in go_terms])

        # Sweep thresholds
        all_scores = score_matrix[~np.isnan(score_matrix)]
        if len(all_scores) == 0:
            return None
        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 101)

        best_s, best_thresh = float('inf'), thresholds[0]

        for tau in thresholds:
            total_ru, total_mi = 0.0, 0.0
            n_evaluated = 0

            for i in range(n_proteins):
                ru_i, mi_i = 0.0, 0.0
                has_any = False
                for j in range(n_terms):
                    if truth[i, j] < 0:
                        continue
                    has_any = True
                    predicted_pos = (not np.isnan(score_matrix[i, j])
                                     and score_matrix[i, j] >= tau)
                    actual_pos = truth[i, j] == 1

                    if actual_pos and not predicted_pos:
                        ru_i += ic_array[j]  # missed annotation
                    elif predicted_pos and not actual_pos:
                        mi_i += ic_array[j]  # false annotation

                if has_any:
                    total_ru += ru_i
                    total_mi += mi_i
                    n_evaluated += 1

            if n_evaluated > 0:
                mean_ru = total_ru / n_evaluated
                mean_mi = total_mi / n_evaluated
                s_tau = math.sqrt(mean_ru ** 2 + mean_mi ** 2)
                if s_tau < best_s:
                    best_s = s_tau
                    best_thresh = tau

        return {
            'Level': level,
            'Algorithm': algorithm,
            'S_min': round(best_s, 6),
            'S_min_Threshold': round(best_thresh, 4),
            'N_GO_Terms': n_terms,
            'N_Proteins': n_proteins,
        }

    def compute_all(self):
        """Compute S_min for all algorithm × level combinations."""
        # Get unique (level, algorithm) pairs
        combos = set()
        for (level, go_term, algorithm) in self.loader.index:
            combos.add((level, algorithm))

        rows = []
        total = len(combos)
        for i, (level, algorithm) in enumerate(sorted(combos)):
            if (i + 1) % 10 == 0:
                logger.info(f"  S_min: {i+1}/{total} ({algorithm} level-{level})...")
            result = self.compute_smin_for_algo_level(level, algorithm)
            if result:
                rows.append(result)

        df = pd.DataFrame(rows)
        logger.info(f"Computed S_min for {len(df)} algorithm-level combinations")
        return df


# ---------------------------------------------------------------------------
# 5. BootstrapCI
# ---------------------------------------------------------------------------

def _bootstrap_one_file(args):
    """Worker function for parallel bootstrap."""
    path_str, n_iter, seed = args
    path = Path(path_str)

    # Parse level, go_term, algorithm from path
    level_match = re.search(r'level-(\d+)', str(path.parent))
    level = int(level_match.group(1)) if level_match else 0
    name_match = re.match(r'^(GO:\d+)_(\w+)_scores\.tsv$', path.name)
    if not name_match:
        return None
    go_term, algorithm = name_match.group(1), name_match.group(2)

    # Load file
    labels, scores = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            labels.append(1 if parts[0].startswith('GO:') else 0)
            scores.append(float(parts[2]))
    y_true = np.array(labels)
    scores = np.array(scores)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    rng = np.random.default_rng(seed)
    fmax_samples, auc_samples = [], []

    for _ in range(n_iter):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_boot = y_true[idx]
        s_boot = scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        fm, _ = compute_fmax(y_boot, s_boot)
        fmax_samples.append(fm)
        try:
            auc_samples.append(roc_auc_score(y_boot, s_boot))
        except ValueError:
            pass

    if len(fmax_samples) < 10:
        return None

    return {
        'Level': level,
        'GO_Term': go_term,
        'Algorithm': algorithm,
        'Fmax_CI_Lower': round(np.percentile(fmax_samples, 2.5), 6),
        'Fmax_CI_Upper': round(np.percentile(fmax_samples, 97.5), 6),
        'Fmax_CI_Width': round(np.percentile(fmax_samples, 97.5) - np.percentile(fmax_samples, 2.5), 6),
        'AUC_CI_Lower': round(np.percentile(auc_samples, 2.5), 6) if auc_samples else np.nan,
        'AUC_CI_Upper': round(np.percentile(auc_samples, 97.5), 6) if auc_samples else np.nan,
        'AUC_CI_Width': round(np.percentile(auc_samples, 97.5) - np.percentile(auc_samples, 2.5), 6) if auc_samples else np.nan,
        'N_Valid_Iterations': len(fmax_samples),
    }


class BootstrapCI:
    """Computes bootstrap 95% confidence intervals for F_max and ROC-AUC."""

    def compute_all(self, loader, n_iterations=1000, n_workers=4, seed=42):
        """Run bootstrap across all scores files."""
        args_list = [
            (str(path), n_iterations, seed + i)
            for i, path in enumerate(sorted(loader.index.values()))
        ]

        logger.info(f"Running bootstrap ({n_iterations} iterations, {n_workers} workers, "
                     f"{len(args_list)} files)...")

        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(_bootstrap_one_file, args_list)
        else:
            results = [_bootstrap_one_file(a) for a in args_list]

        rows = [r for r in results if r is not None]
        df = pd.DataFrame(rows)
        logger.info(f"Bootstrap completed for {len(df)} files")
        return df


# ---------------------------------------------------------------------------
# 6. ProteinDifficulty
# ---------------------------------------------------------------------------

class ProteinDifficulty:
    """Computes per-protein difficulty scores across algorithms."""

    def compute_all(self, loader, per_file_df):
        """For each protein, count how many algorithms classify it correctly."""
        # Build lookup: (level, go_term, algorithm) -> optimal threshold
        threshold_lookup = {}
        for _, row in per_file_df.iterrows():
            key = (row['Level'], row['GO_Term'], row['Algorithm'])
            threshold_lookup[key] = row['F_max_Threshold']

        # Group by (level, go_term)
        level_term_algos = defaultdict(list)
        for (level, go_term, algorithm) in loader.index:
            level_term_algos[(level, go_term)].append(algorithm)

        rows = []
        total = len(level_term_algos)
        for i, ((level, go_term), algorithms) in enumerate(sorted(level_term_algos.items())):
            if (i + 1) % 50 == 0:
                logger.info(f"  Protein difficulty: {i+1}/{total} terms...")

            # For each protein, track correct/total across algorithms
            protein_correct = defaultdict(int)
            protein_total = defaultdict(int)
            protein_label = {}

            for algo in algorithms:
                key = (level, go_term, algo)
                thresh = threshold_lookup.get(key)
                if thresh is None:
                    continue

                path = loader.index.get(key)
                if path is None:
                    continue

                y_true, scores, proteins = loader.load_scores_file(path)
                y_pred = (scores >= thresh).astype(int)
                correct = (y_pred == y_true)

                for j, pid in enumerate(proteins):
                    protein_correct[pid] += int(correct[j])
                    protein_total[pid] += 1
                    protein_label[pid] = int(y_true[j])

            for pid in protein_correct:
                n_corr = protein_correct[pid]
                n_tot = protein_total[pid]
                difficulty = 1.0 - (n_corr / n_tot) if n_tot > 0 else 0.0

                if difficulty == 1.0:
                    category = 'universally_hard'
                elif difficulty >= 0.75:
                    category = 'hard'
                elif difficulty >= 0.25:
                    category = 'medium'
                else:
                    category = 'easy'

                rows.append({
                    'Level': level,
                    'GO_Term': go_term,
                    'Protein_ID': pid,
                    'Is_Positive': protein_label[pid],
                    'N_Correct': n_corr,
                    'N_Algorithms': n_tot,
                    'Difficulty_Score': round(difficulty, 4),
                    'Category': category,
                })

        df = pd.DataFrame(rows)
        logger.info(f"Computed difficulty for {len(df)} protein-term pairs")
        return df


# ---------------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------------

def plot_smin_comparison(smin_df, output_path):
    """Bar chart of S_min by algorithm, averaged across levels."""
    if smin_df.empty:
        return
    avg = smin_df.groupby('Algorithm')['S_min'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))
    avg.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Mean S_min (lower is better)')
    ax.set_title('CAFA S_min by Algorithm (averaged across GO levels)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved S_min comparison plot to {output_path}")


def plot_fmax_distribution(per_file_df, output_path):
    """Boxplots of F_max distribution by algorithm."""
    if per_file_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    order = per_file_df.groupby('Algorithm')['F_max'].mean().sort_values(ascending=False).index
    sns.boxplot(data=per_file_df, x='Algorithm', y='F_max', order=order, ax=ax,
                palette='viridis', showmeans=True,
                meanprops={'marker': '^', 'markerfacecolor': 'red', 'markersize': 8})
    ax.set_title('F_max Distribution Across GO Terms by Algorithm')
    ax.set_ylabel('F_max')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F_max distribution plot to {output_path}")


def plot_protein_difficulty(difficulty_df, output_path):
    """Histogram of protein difficulty scores with category breakdown."""
    if difficulty_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    axes[0].hist(difficulty_df['Difficulty_Score'], bins=50, color='steelblue',
                 edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Difficulty Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Protein Difficulty Scores')

    # Right: category pie chart
    cats = difficulty_df['Category'].value_counts()
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c',
              'universally_hard': '#8e44ad'}
    cat_colors = [colors.get(c, 'gray') for c in cats.index]
    axes[1].pie(cats.values, labels=cats.index, autopct='%1.1f%%', colors=cat_colors)
    axes[1].set_title('Protein Difficulty Categories')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved protein difficulty plot to {output_path}")


# ---------------------------------------------------------------------------
# 8. Report Generation
# ---------------------------------------------------------------------------

def generate_report(per_file_df, smin_df, difficulty_df, bootstrap_df, output_path):
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXTENDED BENCHMARK METRICS REPORT")
    lines.append("=" * 70)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Per-file metrics summary
    lines.append("\n" + "-" * 70)
    lines.append("1. PER-FILE METRICS SUMMARY")
    lines.append("-" * 70)

    if not per_file_df.empty:
        algo_summary = per_file_df.groupby('Algorithm').agg({
            'ROC_AUC': 'mean',
            'PR_AUC': 'mean',
            'F_max': 'mean',
            'MCC_at_Fmax': 'mean',
        }).round(4).sort_values('F_max', ascending=False)

        lines.append("\nMean metrics by algorithm (sorted by F_max):\n")
        lines.append(f"{'Algorithm':<15} {'ROC-AUC':>10} {'PR-AUC':>10} {'F_max':>10} {'MCC':>10}")
        lines.append("-" * 55)
        for algo, row in algo_summary.iterrows():
            lines.append(f"{algo:<15} {row['ROC_AUC']:>10.4f} {row['PR_AUC']:>10.4f} "
                         f"{row['F_max']:>10.4f} {row['MCC_at_Fmax']:>10.4f}")

        best_algo = algo_summary['F_max'].idxmax()
        best_fmax = algo_summary.loc[best_algo, 'F_max']
        lines.append(f"\nBest algorithm by F_max: {best_algo} ({best_fmax:.4f})")

    # Class imbalance
    lines.append("\n" + "-" * 70)
    lines.append("2. CLASS IMBALANCE")
    lines.append("-" * 70)
    if not per_file_df.empty:
        imb = per_file_df.groupby('GO_Term')['Class_Imbalance'].first()
        lines.append(f"\nClass imbalance (proportion positive) across {len(imb)} GO terms:")
        lines.append(f"  Mean: {imb.mean():.4f}")
        lines.append(f"  Min:  {imb.min():.4f}")
        lines.append(f"  Max:  {imb.max():.4f}")
        lines.append(f"  Std:  {imb.std():.4f}")

    # S_min
    lines.append("\n" + "-" * 70)
    lines.append("3. S_min (CAFA SEMANTIC DISTANCE)")
    lines.append("-" * 70)
    if smin_df is not None and not smin_df.empty:
        smin_avg = smin_df.groupby('Algorithm')['S_min'].mean().sort_values()
        lines.append("\nMean S_min by algorithm (lower = better):\n")
        for algo, val in smin_avg.items():
            lines.append(f"  {algo:<15} {val:.4f}")
        lines.append(f"\nBest algorithm by S_min: {smin_avg.idxmin()} ({smin_avg.min():.4f})")
    else:
        lines.append("\nS_min not computed.")

    # Bootstrap CIs
    lines.append("\n" + "-" * 70)
    lines.append("4. BOOTSTRAP CONFIDENCE INTERVALS")
    lines.append("-" * 70)
    if bootstrap_df is not None and not bootstrap_df.empty:
        ci_summary = bootstrap_df.groupby('Algorithm').agg({
            'Fmax_CI_Width': 'mean',
            'AUC_CI_Width': 'mean',
        }).round(4)
        lines.append(f"\nMean 95% CI width by algorithm:\n")
        lines.append(f"{'Algorithm':<15} {'F_max CI Width':>15} {'AUC CI Width':>15}")
        lines.append("-" * 45)
        for algo, row in ci_summary.sort_values('Fmax_CI_Width').iterrows():
            lines.append(f"{algo:<15} {row['Fmax_CI_Width']:>15.4f} {row['AUC_CI_Width']:>15.4f}")
    else:
        lines.append("\nBootstrap CIs not computed.")

    # Protein difficulty
    lines.append("\n" + "-" * 70)
    lines.append("5. PROTEIN DIFFICULTY")
    lines.append("-" * 70)
    if not difficulty_df.empty:
        cats = difficulty_df['Category'].value_counts()
        total = len(difficulty_df)
        lines.append(f"\nDifficulty distribution ({total} protein-term pairs):")
        for cat in ['easy', 'medium', 'hard', 'universally_hard']:
            count = cats.get(cat, 0)
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"  {cat:<20} {count:>6} ({pct:.1f}%)")

        # Top universally hard proteins
        hard = difficulty_df[difficulty_df['Category'] == 'universally_hard']
        if len(hard) > 0:
            lines.append(f"\n{len(hard)} universally hard protein-term pairs "
                         f"(misclassified by ALL algorithms)")

    lines.append("\n" + "=" * 70)

    report_text = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    logger.info(f"Saved report to {output_path}")


# ---------------------------------------------------------------------------
# 9. Main Runner
# ---------------------------------------------------------------------------

class ExtendedMetricsRunner:
    """Orchestrates all extended metric computations."""

    def __init__(self, scores_dir, obo_path, gaf_path, output_dir,
                 n_workers=4, bootstrap_iterations=1000, skip_bootstrap=False,
                 skip_smin=False):
        self.scores_dir = Path(scores_dir)
        self.obo_path = Path(obo_path)
        self.gaf_path = Path(gaf_path)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.bootstrap_iterations = bootstrap_iterations
        self.skip_bootstrap = skip_bootstrap
        self.skip_smin = skip_smin

        # Create output directories
        self.data_dir = self.output_dir / 'data'
        self.figures_dir = self.output_dir / 'figures'
        self.reports_dir = self.output_dir / 'reports'
        for d in [self.data_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute all extended metrics."""
        print("\n" + "=" * 70)
        print("EXTENDED BENCHMARK METRICS PIPELINE")
        print("=" * 70)

        # Step 1: Index scores files
        print("\n[1/8] Indexing scores files...")
        loader = ScoresFileLoader(self.scores_dir)
        loader.build_index()

        # Step 2: Per-file metrics
        print("\n[2/8] Computing per-file metrics (ROC-AUC, PR-AUC, F_max, MCC, imbalance)...")
        pfm = PerFileMetrics()
        per_file_df = pfm.compute_all(loader)
        per_file_df.to_csv(self.data_dir / 'per_file_metrics.csv', index=False)
        print(f"  Saved {len(per_file_df)} rows to per_file_metrics.csv")

        # Step 3: S_min
        smin_df = None
        if not self.skip_smin:
            print("\n[3/8] Computing S_min (parsing ontology, propagating IC, sweeping thresholds)...")
            ontology = GOOntologyParser()
            ontology.parse_obo(self.obo_path)
            ontology.parse_gaf_and_compute_ic(self.gaf_path)
            smin_calc = SminCalculator(ontology, loader)
            smin_df = smin_calc.compute_all()
            smin_df.to_csv(self.data_dir / 'smin_scores.csv', index=False)
            print(f"  Saved {len(smin_df)} rows to smin_scores.csv")
        else:
            print("\n[3/8] Skipping S_min (--skip-smin)")

        # Step 4: Bootstrap CIs
        bootstrap_df = None
        if not self.skip_bootstrap:
            print(f"\n[4/8] Computing bootstrap CIs ({self.bootstrap_iterations} iterations, "
                  f"{self.n_workers} workers)...")
            bci = BootstrapCI()
            bootstrap_df = bci.compute_all(loader, self.bootstrap_iterations,
                                           self.n_workers, seed=42)
            bootstrap_df.to_csv(self.data_dir / 'bootstrap_confidence.csv', index=False)
            print(f"  Saved {len(bootstrap_df)} rows to bootstrap_confidence.csv")
        else:
            print("\n[4/8] Skipping bootstrap (--skip-bootstrap)")

        # Step 5: Protein difficulty
        print("\n[5/8] Computing per-protein difficulty scores...")
        pd_calc = ProteinDifficulty()
        difficulty_df = pd_calc.compute_all(loader, per_file_df)
        difficulty_df.to_csv(self.data_dir / 'protein_difficulty.csv', index=False)
        print(f"  Saved {len(difficulty_df)} rows to protein_difficulty.csv")

        # Step 6: Merged summary
        print("\n[6/8] Creating extended summary...")
        extended = per_file_df.copy()
        if bootstrap_df is not None and not bootstrap_df.empty:
            extended = extended.merge(
                bootstrap_df[['Level', 'GO_Term', 'Algorithm',
                              'Fmax_CI_Lower', 'Fmax_CI_Upper', 'Fmax_CI_Width',
                              'AUC_CI_Lower', 'AUC_CI_Upper', 'AUC_CI_Width']],
                on=['Level', 'GO_Term', 'Algorithm'],
                how='left'
            )
        extended.to_csv(self.data_dir / 'extended_summary.csv', index=False)
        print(f"  Saved {len(extended)} rows to extended_summary.csv")

        # Step 7: Visualizations
        print("\n[7/8] Generating visualizations...")
        plot_fmax_distribution(per_file_df, self.figures_dir / 'fmax_distribution.png')
        if smin_df is not None and not smin_df.empty:
            plot_smin_comparison(smin_df, self.figures_dir / 'smin_comparison.png')
        plot_protein_difficulty(difficulty_df, self.figures_dir / 'protein_difficulty_distribution.png')

        # Step 8: Report
        print("\n[8/8] Generating report...")
        generate_report(per_file_df, smin_df, difficulty_df, bootstrap_df,
                        self.reports_dir / 'extended_metrics_report.txt')

        print("\n" + "=" * 70)
        print("EXTENDED METRICS PIPELINE COMPLETE!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute extended benchmark metrics from raw scores files')
    parser.add_argument('--scores-dir', type=str, required=True,
                        help='Path to ProtCast_stats directory')
    parser.add_argument('--obo', type=str, required=True,
                        help='Path to go.obo file')
    parser.add_argument('--gaf', type=str, required=True,
                        help='Path to filtered GAF file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for bootstrap (default: 4)')
    parser.add_argument('--bootstrap-iterations', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip bootstrap CI computation (slow)')
    parser.add_argument('--skip-smin', action='store_true',
                        help='Skip S_min computation')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    runner = ExtendedMetricsRunner(
        scores_dir=args.scores_dir,
        obo_path=args.obo,
        gaf_path=args.gaf,
        output_dir=args.output,
        n_workers=args.workers,
        bootstrap_iterations=args.bootstrap_iterations,
        skip_bootstrap=args.skip_bootstrap,
        skip_smin=args.skip_smin,
    )
    runner.run()


if __name__ == '__main__':
    main()

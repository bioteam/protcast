"""compare_order_esm_vs_order_combined.py

Two-way CAFA comparison of *order-embedding* neural networks for protein
function prediction, holding the network geometry fixed and varying only the
input feature representation:

  1. Order NN (ESM-C)             – MultiLabelClassifier with OrderEmbeddingLayer
                                    trained on raw ESM-C embeddings. The GO-DAG
                                    order-violation loss forces predicted labels
                                    to respect the ontology hierarchy
                                    ("one true path").
  2. Order NN (ESM-C + PseKRAAC)  – the SAME order-embedding network trained on a
                                    concatenation of ESM-C embeddings and
                                    classical PseKRAAC feature vectors. The two
                                    feature blocks are standardised separately
                                    (fit on the training split only) before
                                    concatenation, so the high-dimensional ESM
                                    block does not numerically dominate the order
                                    projection head.

Primary question: once the labels are tied into the GO DAG by an order
embedding, do classical PseKRAAC descriptors still carry information that
complements ESM-C — i.e. can NN(ESM + Order) be outperformed by
NN(ESM + Order + PseKRAAC)? This is the order-embedding analogue of the earlier
flat-NN and KNN ESM-vs-combined comparisons.

Both arms are trained and evaluated on the *same* protein set, the *same*
train/val split (controlled by --seed), the *same* order hyper-parameters, and
the *same* GO-DAG edges, so any Fmax/Smin delta is attributable to the feature
representation, not the geometry, the data, or the split.

Inputs (not modified):
    - Pre-computed ESM embeddings (.pkl files) in the -d/--input_dir directory.
    - Serialized ProtCastDataset (.bin file) with sequences + GO annotations.

Saved to the output directory (-o/--output_dir, default: order_esm_vs_combined):
    - {name}_order_esm_vs_combined_results.json   All results (updated per model)
    - Order NN (ESM):       {name}_order_esm_comparison_multilabel.keras
    - Order NN (combined):  {name}_order_combined_comparison_multilabel.keras
    - GOEncoder files for each model
    - {name}_order_combined_scalers.pkl  (ESM + FV StandardScalers, plus the
                                          train-protein-id list used to fit them,
                                          for reproducible inference)

    If the results JSON already exists with all requested models completed, the
    script loads and prints without retraining. Individual models already present
    are skipped on re-run (allows resuming interrupted runs).

Example usage:

python3 scripts/compare_order_esm_vs_order_combined.py \\
    -d mf_go_terms-level-8 \\
    -p ProtCastDataset.bin \\
    -o order_esm_vs_combined \\
    --feature_algorithms PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8 \\
    --order-variant soft \\
    --seed 42 \\
    -v
"""

import os

# Enable TensorFlow determinism BEFORE any TF import (which happens transitively
# below via the classifier modules). Together these flags make GPU training
# deterministic for a given seed: identical Fmax across re-runs of the same
# seed. This eliminates within-seed (training) variance — so a single run per
# seed is statistically sufficient, and all observed variance is attributable
# to the train/val split (between-seed variance). Costs ~5–10% throughput.
# Mirrors compare_knn_vs_multilabel.py so the two environments are identical.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import re
import gc
import json
import time
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from protcast.model.multilabel_classifier import MultiLabelClassifier
from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_flat_embeddings(input_dir, verbose=False):
    """Load pre-computed ESM embeddings into flat protein-centric dicts.

    Identical to load_flat_embeddings() in compare_knn_vs_multilabel.py and
    compare_knn_esm_vs_knn_combined.py: one entry per protein, GO ids normalised
    to canonical GO:XXXXXXX form (filenames use underscores, the DAG uses colons).

    Returns
    -------
    protein_embeddings : dict
        {protein_id: np.ndarray} mapping each protein to its ESM embedding.
    protein_go_terms : dict
        {protein_id: set[str]} mapping each protein to its GO term annotations.
    go_ids : list[str]
        Ordered list of GO term IDs found in the embedding files.
    """
    protein_embeddings = {}
    protein_go_terms = defaultdict(set)
    go_ids = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".pkl"):
            continue
        match = re.match(r"(GO_\d+)", filename)
        if not match:
            continue
        go_id = match.group(1).replace("_", ":", 1)
        go_ids.append(go_id)
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "rb") as f:
            embeddings_dict = pickle.load(f)
        for pid, embedding in embeddings_dict.items():
            if pid not in protein_embeddings:
                protein_embeddings[pid] = embedding
            protein_go_terms[pid].add(go_id)
    if verbose:
        print(f"Loaded {len(protein_embeddings)} proteins, {len(go_ids)} GO terms")
    return protein_embeddings, dict(protein_go_terms), go_ids


def compute_classical_feature_vectors(sequences, feature_algorithms, verbose=False):
    """Compute the concatenated classical FV block for every protein.

    Identical in spirit to the same-named helper in
    compare_knn_esm_vs_knn_combined.py. Each algorithm is run separately and the
    per-algorithm encodings are concatenated; only proteins that have a valid
    encoding under *every* requested algorithm are kept (the intersection).

    Parameters
    ----------
    sequences : dict
        {protein_id: str} mapping. Order is not preserved by the calculator,
        so we look entries up by pid afterwards.
    feature_algorithms : list[str]
        Algorithms passed one-by-one to Calculator.get_feature_vectors,
        e.g. ["PseKRAAC_type_7", "PseKRAAC_type_3B", "PseKRAAC_type_8"].

    Returns
    -------
    fv_dict : dict
        {protein_id: np.ndarray} — only protein ids valid under every algorithm.
    """
    from protein_feature_vectors import Calculator
    fv = Calculator(verbose=verbose)
    algo_encodings = {}
    for algo in feature_algorithms:
        fv.get_feature_vectors(algo, pdict=sequences)
        if fv.encodings is None:
            raise ValueError(f"No {algo} encodings generated")
        algo_encodings[algo] = fv.encodings

    valid_pids = [
        pid for pid in sequences
        if all(pid in enc.index for enc in algo_encodings.values())
    ]
    if verbose:
        skipped = len(sequences) - len(valid_pids)
        if skipped:
            print(f"Classical FV: dropped {skipped} proteins missing in ≥1 encoding")

    fv_dict = {}
    for pid in valid_pids:
        parts = [
            algo_encodings[algo].loc[pid].values.astype(np.float32)
            for algo in feature_algorithms
        ]
        fv_dict[pid] = np.concatenate(parts).astype(np.float32)
    return fv_dict


def get_train_pids(protein_ids, validation_split, random_state):
    """Reproduce the exact train split that MultiLabelClassifier.train_model uses.

    MultiLabelClassifier sorts protein ids inside prepare_data() and then passes
    the matching X / y rows to sklearn.train_test_split with the supplied
    random_state. We mirror that here so the combined-feature scalers are fit on
    *exactly* the proteins the network will later train on — no validation
    leakage.

    Invariant: if the classifier's sort order ever drifts from `sorted()`, this
    silently fits scalers on the wrong split. We feed the classifier a
    pre-filtered dict whose sorted keys are `protein_ids`, which is what keeps
    the two splits in agreement.
    """
    from sklearn.model_selection import train_test_split
    sorted_pids = sorted(protein_ids)
    train_pids, _ = train_test_split(
        sorted_pids, test_size=validation_split, random_state=random_state,
    )
    return train_pids


def build_combined_embeddings(protein_embeddings, fv_dict, train_pids, verbose=False):
    """Concatenate scaled ESM and scaled classical FVs into one vector per pid.

    Scalers are fit on the train pids only (no validation leakage) and then used
    to transform every protein. We return the resulting per-pid dict plus the
    fitted scalers so they can be persisted for inference.

    Standardising the two blocks *separately* matters more for the order model
    than it did for the flat model: the order projection head feeds a
    Dense(order_dim, softplus), and an unscaled PseKRAAC block (counts/fractions
    on a different scale than ESM activations) would dominate that projection and
    distort the learned order geometry.
    """
    common_pids = sorted(set(protein_embeddings) & set(fv_dict))
    if not common_pids:
        raise ValueError("No proteins have both ESM and classical FVs")

    esm_dim = next(iter(protein_embeddings.values())).shape[0]
    fv_dim = next(iter(fv_dict.values())).shape[0]

    train_pid_set = set(train_pids)
    train_only = [p for p in common_pids if p in train_pid_set]
    if not train_only:
        raise ValueError("No training proteins survived the FV intersection")

    esm_train = np.vstack([protein_embeddings[p] for p in train_only]).astype(np.float32)
    fv_train = np.vstack([fv_dict[p] for p in train_only]).astype(np.float32)

    esm_scaler = StandardScaler().fit(esm_train)
    fv_scaler = StandardScaler().fit(fv_train)

    combined = {}
    for pid in common_pids:
        esm_s = esm_scaler.transform(protein_embeddings[pid].reshape(1, -1).astype(np.float32))
        fv_s = fv_scaler.transform(fv_dict[pid].reshape(1, -1).astype(np.float32))
        fv_s = np.nan_to_num(fv_s, nan=0.0, posinf=0.0, neginf=0.0)
        combined[pid] = np.concatenate([esm_s.ravel(), fv_s.ravel()]).astype(np.float32)

    if verbose:
        print(
            f"Combined feature dim: {esm_dim + fv_dim} "
            f"(ESM: {esm_dim} + FV: {fv_dim})  over {len(common_pids)} proteins"
        )
    return combined, esm_scaler, fv_scaler, common_pids


# ──────────────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────────────

def train_order_multilabel(
    protein_embeddings, protein_go_terms, go_ids,
    config, name, variant_tag, seed, go_dag, order_variant, use_mlflow,
    verbose=False,
):
    """Fit an order-embedding MultiLabelClassifier and return a result dict.

    Always runs with USE_ORDER_EMBEDDINGS=True (the geometry is the constant in
    this comparison — only the feature representation changes). The config is
    copied so flags never persist between runs or get written back to disk.

    Parameters
    ----------
    variant_tag : str
        "esm" or "combined" — distinguishes the two model files / MLflow runs.
    order_variant : str
        "soft" (softplus violation, default) or "hard" (ReLU). Controls the
        order-violation loss formulation; see protcast.model.order_embeddings.
    """
    run_config = dict(config)
    run_config["USE_ORDER_EMBEDDINGS"] = True
    run_config["ORDER_VARIANT"] = order_variant

    classifier = MultiLabelClassifier(
        verbose=verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=run_config,
        id=f"{name}_order_{variant_tag}_comparison",
        use_mlflow=use_mlflow,
        go_dag=go_dag,
        random_state=seed,
        # Always standardize inputs (train-fold-fit), matching
        # scan_individual_features.py, so ESM and ESM+FV arms are preprocessed
        # identically and no arm silently runs unscaled on mean_max_std.
        scale_features=True,
    )
    classifier.run()

    # Predict on the held-out validation set once; reuse for all metrics.
    y_pred = classifier.model.predict(classifier.X_val, verbose=0)
    fmax, fmax_threshold = calculate_fmax(classifier.y_val, y_pred)
    smin, smin_threshold = calculate_smin(classifier.y_val, y_pred)

    depth_metrics = classifier.compute_depth_metrics(classifier.y_val, y_pred)
    freq_metrics = classifier.compute_frequency_metrics(classifier.y_val, y_pred)

    classifier.save_model()

    result = {
        "fmax": float(fmax),
        "fmax_threshold": float(fmax_threshold),
        "smin": float(smin),
        "smin_threshold": float(smin_threshold),
        "best_loss": float(min(classifier.history.history["val_loss"])),
        "epochs": len(classifier.history.history["loss"]),
        "training_time": round(classifier.training_time, 2),
        "vector_length": int(classifier.vector_length),
        "order_variant": order_variant,
        "depth_metrics": {str(k): v for k, v in depth_metrics.items()},
        "frequency_metrics": freq_metrics,
        "status": "ok",
    }
    del classifier
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────────────────────────────────────

def _nan_safe(value, fmt=".4f"):
    try:
        if np.isnan(float(value)):
            return "---"
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "---"


def _delta_str(a, b):
    try:
        d = float(a) - float(b)
        return f"{d:>+9.4f}"
    except (TypeError, ValueError):
        return f"{'---':>9}"


def print_results(results):
    """Print three sections: overall, depth breakdown, frequency breakdown.

    The ESM-only order model is the baseline; the combined model's delta versus
    that baseline is the headline number for the hypothesis.
    """
    esm = results.get("order_esm", {})
    comb = results.get("order_combined", {})
    esm_fmax = esm.get("fmax")

    sep = "=" * 100
    thin = "-" * 100
    print("\n" + sep)
    print("ORDER NN (ESM-C) vs ORDER NN (ESM-C + PseKRAAC)  (CAFA Metrics)")
    print(sep)
    print(
        f"Level : {results.get('level', '?')}   "
        f"Seed  : {results['seed']}   "
        f"ESM dim : {results.get('esm_dim', '?')}   "
        f"Combined dim : {results.get('combined_dim', '?')}   "
        f"Order : {results.get('order_variant', '?')}"
    )
    print(f"Features : {', '.join(results.get('feature_algorithms', []))}")

    # ── Overall ───────────────────────────────────────────────────────────────
    print()
    print("── OVERALL METRICS ──")
    hdr = (f"{'Model':<28} {'Fmax':>8} {'Thr':>6} {'Smin':>8} {'Epochs':>7} "
           f"{'Time':>8}   {'Δ vs ESM':>10}")
    print(hdr)
    print(thin)

    def _row(label, r, is_baseline=False):
        fmax = _nan_safe(r.get("fmax"))
        thr = _nan_safe(r.get("fmax_threshold"), ".2f")
        smin = _nan_safe(r.get("smin"))
        t = _nan_safe(r.get("training_time"), ".1f")
        ep = r.get("epochs")
        ep_s = f"{ep:>7d}" if isinstance(ep, int) else f"{'---':>7}"
        delta = f"{'---':>10}" if is_baseline else _delta_str(r.get("fmax"), esm_fmax)
        print(f"{label:<28} {fmax:>8} {thr:>6} {smin:>8} {ep_s} {t:>7}s   {delta}")

    if esm:
        _row("Order NN (ESM-C)", esm, is_baseline=True)
    if comb:
        _row("Order NN (ESM-C + PseKRAAC)", comb)

    # ── Depth breakdown ───────────────────────────────────────────────────────
    print()
    print("── DEPTH BREAKDOWN  (higher depth = more specific GO terms) ──")
    all_depths = set()
    for r in (esm, comb):
        if r:
            all_depths.update(int(d) for d in r.get("depth_metrics", {}))

    if all_depths:
        hdr = (f"{'Depth':>5}  {'N terms':>7}  {'Avg ann':>8}  "
               f"{'ESM':>8}  {'Combined':>9}  {'Δ vs ESM':>10}")
        print(hdr)
        print("-" * len(hdr))
        for depth in sorted(all_depths):
            ds = str(depth)
            esm_d = esm.get("depth_metrics", {}).get(ds, {})
            comb_d = comb.get("depth_metrics", {}).get(ds, {})
            meta = esm_d or comb_d
            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")
            esm_f = esm_d.get("fmax")
            comb_f = comb_d.get("fmax")
            print(
                f"{depth:>5}  {n_terms:>7}  {avg_ann:>8}  "
                f"{_nan_safe(esm_f):>8}  {_nan_safe(comb_f):>9}  "
                f"{_delta_str(comb_f, esm_f)}"
            )
    else:
        print("  (no depth metrics — go_dag not available)")

    # ── Frequency breakdown ───────────────────────────────────────────────────
    print()
    print("── FREQUENCY BREAKDOWN  (by training annotation count) ──")
    bucket_labels = [
        ("rare_lt50", "Rare  (<50)     "),
        ("medium_50_500", "Medium (50–500) "),
        ("common_gt500", "Common (>500)   "),
    ]
    any_freq = any(r.get("frequency_metrics") for r in (esm, comb) if r)
    if any_freq:
        hdr = (f"{'Bucket':<18}  {'N terms':>7}  {'Avg ann':>8}  "
               f"{'ESM':>8}  {'Combined':>9}  {'Δ vs ESM':>10}")
        print(hdr)
        print("-" * len(hdr))
        for bucket, label in bucket_labels:
            esm_b = esm.get("frequency_metrics", {}).get(bucket, {})
            comb_b = comb.get("frequency_metrics", {}).get(bucket, {})
            meta = esm_b or comb_b
            if not meta:
                continue
            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")
            esm_f = esm_b.get("fmax")
            comb_f = comb_b.get("fmax")
            print(
                f"{label:<18}  {n_terms:>7}  {avg_ann:>8}  "
                f"{_nan_safe(esm_f):>8}  {_nan_safe(comb_f):>9}  "
                f"{_delta_str(comb_f, esm_f)}"
            )
    else:
        print("  (no frequency metrics)")

    print()
    print(f"Total elapsed time: {results.get('elapsed', '?')}s")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare order-embedding NN on ESM-C vs ESM-C + PseKRAAC feature "
            "vectors using CAFA Fmax/Smin with depth and frequency breakdowns. "
            "Both arms are order-embedding NNs; only the input features change."
        )
    )
    parser.add_argument("-d", "--input_dir", required=True,
                        help="Directory of pre-computed ESM embedding .pkl files")
    parser.add_argument("-p", "--protcast_dataset", required=True,
                        help="Path to serialised ProtCastDataset (.bin)")
    parser.add_argument("--feature_algorithms", nargs="+",
                        default=["PseKRAAC_type_7", "PseKRAAC_type_3B", "PseKRAAC_type_8"],
                        help=("Classical FV algorithms to concatenate with ESM "
                              "(default: PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8)"))
    parser.add_argument("--order-variant", choices=["soft", "hard"], default="soft",
                        help=(
                            "Order-violation loss variant for BOTH arms. 'soft' "
                            "(default) uses softplus violations and keeps gradients "
                            "alive even when the child already dominates the parent "
                            "(Lai & Hockenmaier 2017); 'hard' uses ReLU violations."
                        ))
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42)")
    parser.add_argument("-o", "--output_dir", default="order_esm_vs_combined",
                        help="Directory for output files (default: order_esm_vs_combined)")
    parser.add_argument("--use_mlflow", action="store_true",
                        help="Log each model run to MLflow")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    start = time.time()

    # Seed Python, NumPy, and TF RNGs so the determinism flags actually take
    # effect. Combined with TF_DETERMINISTIC_OPS at the top of this file, this
    # makes the same --seed reproduce the same Fmax bit-for-bit across re-runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    except Exception:
        pass

    # Resolve to absolute paths now — os.chdir() later would silently break any
    # relative paths used after the working directory changes.
    input_dir = os.path.abspath(args.input_dir)
    protcast_dataset = os.path.abspath(args.protcast_dataset)
    name = os.path.basename(input_dir.rstrip("/"))

    level_match = re.search(r"level-(\d+)", name)
    level = int(level_match.group(1)) if level_match else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    results_file = f"{name}_order_esm_vs_combined_results.json"

    # ── Resume support ─────────────────────────────────────────────────────
    results = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    expected = {"order_esm", "order_combined"}
    if results is not None:
        done = {k for k in expected if results.get(k, {}).get("status") == "ok"}
        if done == expected:
            print(f"All models already completed in {results_file}")
            print_results(results)
            return
        remaining = expected - done
        print(f"Resuming: completed={sorted(done)}, remaining={sorted(remaining)}")
    else:
        results = {
            "seed": args.seed,
            "level": level,
            "feature_algorithms": args.feature_algorithms,
            "order_variant": args.order_variant,
            "esm_dim": None,
            "combined_dim": None,
        }

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.isdir(input_dir):
        print(f"Error: embeddings directory not found: {input_dir}")
        return
    if not [f for f in os.listdir(input_dir) if f.endswith(".pkl")]:
        print(f"Error: no .pkl embedding files found in {input_dir}")
        return
    if not os.path.exists(protcast_dataset):
        print(f"Error: ProtCast dataset not found: {protcast_dataset}")
        return

    # ── Load ESM embeddings + dataset ──────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    try:
        dataset = ProtCastDataset.load_serialized_file(protcast_dataset)
    except Exception as e:
        print(f"Error loading ProtCastDataset: {e}")
        return

    try:
        protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
            input_dir, args.verbose,
        )
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
    if not protein_embeddings:
        print("Error: no embeddings loaded")
        return

    go_dag = getattr(dataset, "annotated_dag", None)
    if go_dag is None:
        # The order-violation loss needs DAG edges; without a go_dag the order
        # model degenerates to a flat sigmoid (empty edge set). Warn loudly.
        print(
            "Warning: ProtCastDataset has no annotated_dag — the order-violation "
            "loss will have zero edges and depth metrics will be unavailable. "
            "The 'order' models will effectively train as flat models."
        )

    # ── Compute PseKRAAC FVs once for the intersection of pids ─────────────
    # Both arms must train on the SAME protein set; otherwise the train/val
    # split (driven by sorted pid order) would differ and the delta would no
    # longer be attributable to the feature representation alone.
    sequences = {
        pid: dataset.proteins[pid].sequence
        for pid in protein_embeddings
        if pid in dataset.proteins
    }
    missing_seq = len(protein_embeddings) - len(sequences)
    if missing_seq:
        print(f"Warning: {missing_seq} embedded proteins absent from ProtCastDataset")

    print("Computing classical (PseKRAAC) feature vectors...")
    try:
        fv_dict = compute_classical_feature_vectors(
            sequences, args.feature_algorithms, verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error computing classical feature vectors: {e}")
        return

    # Restrict the protein set to the intersection so both arms train on
    # identical data. The ESM-only arm therefore drops any pids that lack a
    # valid classical FV — this is intentional, to keep the split identical.
    common_pids = sorted(set(protein_embeddings) & set(fv_dict) & set(protein_go_terms))
    if not common_pids:
        print("Error: empty intersection of ESM / FV / annotated proteins")
        return
    if args.verbose:
        print(f"Common protein set (used for both arms): {len(common_pids)}")

    protein_embeddings = {p: protein_embeddings[p] for p in common_pids}
    protein_go_terms = {p: protein_go_terms[p] for p in common_pids}

    if results.get("esm_dim") is None:
        results["esm_dim"] = int(next(iter(protein_embeddings.values())).shape[0])

    # ── Arm 1: Order NN on ESM-C only ──────────────────────────────────────
    if results.get("order_esm", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print(f"MODEL 1 / 2: ORDER NN (ESM-C)  [{args.order_variant}]")
        print("=" * 60)
        try:
            results["order_esm"] = train_order_multilabel(
                protein_embeddings, protein_go_terms, go_ids,
                config, name, "esm", args.seed, go_dag,
                args.order_variant, args.use_mlflow, args.verbose,
            )
            r = results["order_esm"]
            print(
                f"Order NN ESM — Fmax: {r['fmax']:.4f}  "
                f"Smin: {r['smin']:.4f}  Epochs: {r['epochs']}  "
                f"Time: {r['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: Order NN ESM — {e}")
            results["order_esm"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # ── Arm 2: Order NN on (scaled ESM-C ⊕ scaled PseKRAAC) ────────────────
    if results.get("order_combined", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print(f"MODEL 2 / 2: ORDER NN (ESM-C + {', '.join(args.feature_algorithms)})  "
              f"[{args.order_variant}]")
        print("=" * 60)
        try:
            validation_split = config.get("VALIDATION_SPLIT", 0.2)
            train_pids = get_train_pids(common_pids, validation_split, args.seed)
            combined_embeddings, esm_scaler, fv_scaler, _ = build_combined_embeddings(
                protein_embeddings, fv_dict, train_pids, verbose=args.verbose,
            )

            # Persist scalers (+ the train pids used to fit them) so inference on
            # new proteins is exactly reproducible.
            scalers_path = f"{name}_order_combined_scalers.pkl"
            with open(scalers_path, "wb") as f:
                pickle.dump(
                    {
                        "esm_scaler": esm_scaler,
                        "fv_scaler": fv_scaler,
                        "feature_algorithms": args.feature_algorithms,
                        "train_pids": train_pids,
                    },
                    f,
                )
            if args.verbose:
                print(f"Saved combined scalers to {scalers_path}")

            results["combined_dim"] = int(
                next(iter(combined_embeddings.values())).shape[0]
            )

            results["order_combined"] = train_order_multilabel(
                combined_embeddings, protein_go_terms, go_ids,
                config, name, "combined", args.seed, go_dag,
                args.order_variant, args.use_mlflow, args.verbose,
            )
            r = results["order_combined"]
            print(
                f"Order NN Combined — Fmax: {r['fmax']:.4f}  "
                f"Smin: {r['smin']:.4f}  Epochs: {r['epochs']}  "
                f"Time: {r['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: Order NN Combined — {e}")
            results["order_combined"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed"] = round(time.time() - start)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print_results(results)


if __name__ == "__main__":
    main()

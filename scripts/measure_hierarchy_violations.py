"""measure_hierarchy_violations.py

Quantify how much "hierarchy signal is on the table" for the trained models in a
compare_knn_vs_multilabel.py results directory, across every arm present
(KNN, flat NN, order NN), and report — per arm — how often their predictions
violate the GO DAG true-path rule (a child term scored above its parent, which
is impossible for a consistent annotation) and what the true-path rule recovers.

Why compare arms:
  * KNN has ZERO hierarchy awareness yet is the strongest model here. If even
    KNN is already consistent, the task is intrinsically hierarchy-consistent
    and geometric/structural hierarchy methods are moot.
  * The order NN is *designed* to be consistent; this shows whether its geometry
    actually buys the consistency it sacrifices accuracy for.

For each arm we print:
  1. Continuous violations  — % of (protein x parent-child-edge) pairs with
     score(child) > score(parent), and the mean overshoot.
  2. Binary inconsistencies — at the Fmax threshold, % where child is called
     positive while parent is not (a hard contradiction).
  3. delta Fmax from the true-path rule (raise parents to their descendants'
     max, to a fixpoint) — the accuracy a free, geometry-less fix would recover.

Self-validation: each arm's recomputed validation Fmax is printed next to the
value in the results JSON. A mismatch means the reconstruction (protein set,
seed split, column order, train-fold scaler) is wrong and that arm's numbers
must NOT be trusted — a loud warning is printed and the arm is skipped.

Example (Frontera, in the TF container, from the repo root):

    python3 scripts/measure_hierarchy_violations.py \\
        -d $DATADIR/mf_go_terms-level-6-mean_max_std \\
        -p $DATADIR/ProtCastDataset.bin \\
        -o $WORK/ProtCast_results/orderfix-smoke-L6-s42 \\
        --seed 42 -v
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make `import protcast` resolve to THIS checkout (with the order-embedding fix),
# not a stale pip-installed copy — the same trap that bit the earlier runs. The
# scripts dir is added too, for the driver's shared embedding loader.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../scripts
_ROOT = os.path.dirname(_HERE)                              # repo root
sys.path.insert(0, _HERE)                                   # compare_knn_vs_multilabel
sys.path.insert(0, _ROOT)                                   # protcast (repo, first)
from compare_knn_vs_multilabel import load_flat_embeddings  # noqa: E402

from protcast.model.multilabel_classifier import GOEncoder  # noqa: E402
from protcast.model.knn_classifier import KNNClassifier  # noqa: E402
from protcast.model.stats.utils import calculate_fmax  # noqa: E402
from protcast.preprocessing.go_dag_edges import extract_dag_edges  # noqa: E402
from protcast.preprocessing.protcast_dataset import ProtCastDataset  # noqa: E402
from protcast.config.model_config import ConfigManager  # noqa: E402


def true_path_correct(y_pred, edges, max_iter=100):
    """Enforce score(parent) >= score(child) by upward max-propagation.

    Raising parents (rather than lowering children) preserves the specific-term
    recall Fmax rewards. Iterates to a fixpoint so corrections propagate through
    multi-level chains represented as a series of direct edges.
    """
    corrected = y_pred.copy()
    parent_idx = edges[:, 0]
    child_idx = edges[:, 1]
    for _ in range(max_iter):
        updated = corrected.copy()
        np.maximum.at(updated, (slice(None), parent_idx), corrected[:, child_idx])
        if np.array_equal(updated, corrected):
            break
        corrected = updated
    return corrected


def violation_report(label, y_pred, y_val, edges, reported_fmax):
    """Print the violation + true-path summary for one arm. Returns a dict."""
    fmax, thr = calculate_fmax(y_val, y_pred)
    ok = reported_fmax is not None and abs(fmax - float(reported_fmax)) < 5e-3
    flag = "OK" if ok else "!! MISMATCH — numbers UNRELIABLE"
    print(f"\n── {label} ──")
    print(f"  self-check Fmax: recomputed {fmax:.4f} vs reported {reported_fmax}  [{flag}]")
    if not ok:
        return {"label": label, "trustworthy": False}

    p = y_pred[:, edges[:, 0]]      # parent scores
    c = y_pred[:, edges[:, 1]]      # child scores
    diff = c - p
    viol = diff > 0
    frac = float(viol.mean())
    mag = float(diff[viol].mean()) if viol.any() else 0.0
    bin_incon = float(((c >= thr) & (p < thr)).mean())

    y_corr = true_path_correct(y_pred, edges)
    fmax_corr, _ = calculate_fmax(y_val, y_corr)
    dfmax = fmax_corr - fmax

    print(f"  continuous violations : {frac*100:5.2f}% of pairs   mean overshoot {mag:.4f}")
    print(f"  binary inconsistencies: {bin_incon*100:5.2f}% of pairs (child>= {thr:.2f}, parent< {thr:.2f})")
    print(f"  true-path delta Fmax  : {dfmax:+.4f}  ({fmax:.4f} -> {fmax_corr:.4f})")
    return {
        "label": label, "trustworthy": True, "fmax": fmax,
        "cont_viol_pct": frac * 100, "mean_overshoot": mag,
        "bin_incon_pct": bin_incon * 100, "delta_fmax": dfmax,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--input_dir", required=True)
    ap.add_argument("-p", "--protcast_dataset", required=True)
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    import keras

    config = ConfigManager.load_config()
    val_split = config.get("VALIDATION_SPLIT", 0.2)
    input_dir = os.path.abspath(args.input_dir)
    name = os.path.basename(input_dir.rstrip("/"))

    # reported Fmax per arm (for self-checks)
    reported = {}
    rj = glob.glob(os.path.join(args.output_dir, "*_knn_vs_multilabel_results.json"))
    if rj:
        with open(rj[0]) as f:
            R = json.load(f)
        reported = {
            "KNN": (R.get("knn", {}) or {}).get("fmax"),
            "flat NN": (R.get("multilabel_flat", {}) or {}).get("fmax"),
            "order NN": (R.get("multilabel_order", {}) or {}).get("fmax"),
        }

    # ── Rebuild the exact data / split / scaler used by every arm ───────────
    protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
        input_dir, args.verbose)
    go_ids = sorted(go_ids)
    go_encoder = GOEncoder(f"{name}_hviol")
    go_encoder.fit(go_ids)

    protein_ids = sorted(set(protein_embeddings) & set(protein_go_terms))
    X = np.vstack([np.asarray(protein_embeddings[p], dtype=np.float32)
                   for p in protein_ids])
    y = np.zeros((len(protein_ids), len(go_ids)), dtype=np.float32)
    for i, pid in enumerate(protein_ids):
        for go_id in protein_go_terms[pid]:
            j = go_encoder.go_to_int.get(go_id)
            if j is not None:
                y[i, j] = 1.0

    X_tr, X_val_raw, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=args.seed)
    scaler = StandardScaler().fit(X_tr)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)

    # ── Edges (shared across arms) ─────────────────────────────────────────
    go_dag = getattr(
        ProtCastDataset.load_serialized_file(args.protcast_dataset),
        "annotated_dag", None)
    if go_dag is None:
        print("ERROR: dataset has no annotated_dag — cannot extract edges.")
        return
    edges = extract_dag_edges(go_dag, go_ids, go_encoder)

    print("=" * 68)
    print(f"HIERARCHY VIOLATIONS  ({name}, seed {args.seed})")
    print("=" * 68)
    print(f"  GO terms: {len(go_ids)}   within-set parent-child edges: {edges.shape[0]}"
          f"   val proteins: {X_val_raw.shape[0]}")
    if edges.shape[0] == 0:
        print("\n  No within-set parent/child edges: the predicted terms are not")
        print("  directly related in the DAG, so the true-path rule cannot act here.")
        return

    rows = []
    # KNN — raw embeddings (cosine); flat/order — scaled embeddings
    knn_p = glob.glob(os.path.join(args.output_dir, "*knn_comparison_knn.joblib"))
    if knn_p:
        art = KNNClassifier.load_model(knn_p[0])
        yp = KNNClassifier.predict_from_artifact(art, X_val_raw)
        rows.append(violation_report("KNN", yp, y_val, edges, reported.get("KNN")))

    # Flat: unambiguous. Order: the saved filename embeds the variant
    # ("..._order_soft_comparison_...", not "..._order_comparison_..."), so match
    # any variant and exclude the dual-encoder model (a different, 2-input arch).
    flat_files = glob.glob(os.path.join(args.output_dir,
                                        "*multilabel_flat_comparison_multilabel.keras"))
    order_files = [
        f for f in glob.glob(os.path.join(
            args.output_dir, "*multilabel_order_*comparison_multilabel.keras"))
        if "dual" not in os.path.basename(f)
    ]
    keras_arms = [("flat NN", flat_files)] + [("order NN", order_files)]
    for label, files in keras_arms:
        if not files:
            continue
        model = keras.models.load_model(files[0], compile=False)
        yp = model.predict(X_val_scaled, verbose=0)
        rows.append(violation_report(label, yp, y_val, edges, reported.get(label)))

    # ── Comparison table + verdict ─────────────────────────────────────────
    good = [r for r in rows if r.get("trustworthy")]
    if good:
        print("\n" + "=" * 68)
        print("SUMMARY")
        print("=" * 68)
        h = f"{'arm':<10} {'Fmax':>8} {'cont.viol%':>11} {'bin.incon%':>11} {'dFmax(TPR)':>11}"
        print(h); print("-" * len(h))
        for r in good:
            print(f"{r['label']:<10} {r['fmax']:>8.4f} {r['cont_viol_pct']:>11.2f} "
                  f"{r['bin_incon_pct']:>11.2f} {r['delta_fmax']:>+11.4f}")
        max_gain = max(r["delta_fmax"] for r in good)
        print()
        if max_gain < 0.002:
            print("  => Table ~empty across all arms: predictions are already")
            print("     hierarchy-consistent. Geometric/structural hierarchy methods")
            print("     are moot on this task — proceed to the FV experiment.")
        else:
            print(f"  => Up to {max_gain:+.4f} Fmax recoverable by the true-path rule —")
            print("     a free post-processing win worth adopting on the best arm.")


if __name__ == "__main__":
    main()

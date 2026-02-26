"""
mlflow_utils.py

Shared MLflow + DagsHub initialization and logging helpers.
Used by both training (MultiClassifier) and inference scripts
so the setup logic lives in one place.
"""

import json
from pathlib import Path
from typing import Optional


def init_mlflow(
    experiment_name: str = "Default Experiment",
    repo_owner: str = "aakpan",
    repo_name: str = "my-first-repo",
    verbose: bool = False,
):
    """
    Initialize DagsHub-backed MLflow tracking and set the active experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to log to.
    repo_owner : str
        DagsHub repository owner.
    repo_name : str
        DagsHub repository name.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    mlflow module or None
        The imported mlflow module if successful, None otherwise.
    """
    try:
        import dagshub

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
        )
    except Exception as e:
        if verbose:
            print(f"dagshub init failed: {e}")
        return None

    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        if verbose:
            print(f"MLflow experiment set: {experiment_name}")
        return mlflow
    except Exception as e:
        if verbose:
            print(f"mlflow not available; skipping experiment setup: {e}")
        return None


def save_run_metadata(model_name: str, run_id: str, experiment_name: str):
    """
    Save MLflow run metadata alongside a trained model so inference
    scripts can link back to the training run.

    Writes a JSON file named ``{model_name}_mlflow.json`` in the
    current directory.

    Parameters
    ----------
    model_name : str
        Base name of the model file (without .keras extension).
    run_id : str
        The MLflow run ID from the training run.
    experiment_name : str
        The experiment name used for this training run.
    """
    metadata = {
        "training_run_id": run_id,
        "experiment_name": experiment_name,
    }
    meta_path = f"{model_name}_mlflow.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_run_metadata(model_file: str) -> Optional[dict]:
    """
    Load MLflow run metadata that was saved alongside a trained model.

    Looks for a file named ``{model_stem}_mlflow.json`` next to the
    model file.

    Parameters
    ----------
    model_file : str
        Path to the ``.keras`` model file.

    Returns
    -------
    dict or None
        Metadata dict with ``training_run_id`` and ``experiment_name``,
        or None if the metadata file does not exist.
    """
    model_path = Path(model_file)
    meta_path = model_path.parent / f"{model_path.stem}_mlflow.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


class SemanticDistance:
    """
    Compute shortest-path distance between GO terms in the ontology DAG.

    Loads an OBO file once, builds a bidirectional adjacency map
    (parents + children), then uses BFS to find the shortest path
    between any two terms.

    Parameters
    ----------
    obo_file : str
        Path to a Gene Ontology ``.obo`` file.
    """

    def __init__(self, obo_file: str):
        from goatools.obo_parser import GODag

        dag = GODag(obo_file)
        # Build adjacency list: each node -> set of neighbours
        self._adj: dict[str, set[str]] = {}
        for go_id, term in dag.items():
            if term.is_obsolete:
                continue
            neighbours = self._adj.setdefault(go_id, set())
            for parent in term.parents:
                neighbours.add(parent.id)
                self._adj.setdefault(parent.id, set()).add(go_id)
            for child in term.children:
                neighbours.add(child.id)
                self._adj.setdefault(child.id, set()).add(go_id)

    def shortest_path(self, go_id_a: str, go_id_b: str) -> int:
        """
        Return the shortest-path distance (number of edges) between
        two GO terms.  Returns -1 if either term is unknown or no
        path exists.
        """
        if go_id_a == go_id_b:
            return 0
        if go_id_a not in self._adj or go_id_b not in self._adj:
            return -1

        from collections import deque

        visited = {go_id_a}
        queue = deque([(go_id_a, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbour in self._adj.get(current, set()):
                if neighbour == go_id_b:
                    return dist + 1
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, dist + 1))
        return -1

    def batch_distances(
        self, true_ids: list[str], pred_ids: list[str]
    ) -> list[int]:
        """
        Compute pairwise shortest-path distances for aligned lists
        of true and predicted GO IDs.
        """
        return [
            self.shortest_path(t, p) for t, p in zip(true_ids, pred_ids)
        ]


def log_inference_results(
    mlflow,
    params: dict,
    metrics: dict,
    tags: Optional[dict] = None,
    artifact_path: Optional[str] = None,
):
    """
    Log an inference run to MLflow.

    Opens a new MLflow run, logs parameters, metrics, optional tags,
    and an optional artifact file, then ends the run.

    Parameters
    ----------
    mlflow : module
        The mlflow module (returned by init_mlflow).
    params : dict
        Parameters to log (model file, algorithm, etc.).
    metrics : dict
        Metrics to log (F1 scores, inference time, counts, etc.).
    tags : dict, optional
        Extra tags to set on the run.
    artifact_path : str, optional
        Path to a local file to log as an artifact.
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        if tags:
            mlflow.set_tags(tags)
        if artifact_path:
            mlflow.log_artifact(artifact_path)

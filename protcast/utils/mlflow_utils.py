"""
mlflow_utils.py

Shared MLflow + DagsHub initialization and logging helpers.
Used by both training (MultiClassifier) and inference scripts
so the setup logic lives in one place.
"""

from typing import Optional


def init_mlflow(
    experiment_name: str = "Default Experiment",
    verbose: bool = False,
):
    """
    Initialize DagsHub-backed MLflow tracking and set the active experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to log to.
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
            repo_owner="aakpan",
            repo_name="my-first-repo",
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

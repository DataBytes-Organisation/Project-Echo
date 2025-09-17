import mlflow
import pandas as pd

# Load experiment
experiment_name = "echo_complex_model_experiment"
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

experiment_id = experiment.experiment_id

# Fetch runs
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Columns to select
selected_cols = [
    "params.model_type",
    "params.n_estimators",
    "params.max_depth",
    "params.max_iter",
    "params.solver",
    "metrics.accuracy",
    "metrics.f1_score"
]

# Safe selection based on available columns
existing_cols = [col for col in selected_cols if col in runs.columns]
runs_selected = runs[existing_cols]

# Save comparison result
runs_selected.to_csv("experiments/model_comparison.csv", index=False)
print("Model comparison CSV generated successfully!")

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Set experiment name
mlflow.set_experiment("model_registry_demo")

# Start tracking
with mlflow.start_run():

    # Define and train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    accuracy = clf.score(X_test, y_test)

    # Log param, metric, model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name="IrisRandomForest"
    )

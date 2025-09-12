import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set Experiment
mlflow.set_experiment("model_registry_demo")

with mlflow.start_run():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name="RandomForestIris"
    )

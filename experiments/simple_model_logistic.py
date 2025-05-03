import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow tracking
with mlflow.start_run(run_name="simple_logistic_regression"):

    # Define model
    model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto')

    # Log model parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("solver", "lbfgs")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="logistic_model")

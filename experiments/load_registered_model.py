import mlflow.sklearn

# Load model from MLflow Model Registry
model = mlflow.sklearn.load_model("models:/RandomForestIris/1")

# Predict a sample
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print(f"Predicted Class: {prediction}")

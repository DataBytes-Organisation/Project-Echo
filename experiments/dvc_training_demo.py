import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dummy Iris data
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
with open('experiments/random_forest_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved successfully!")

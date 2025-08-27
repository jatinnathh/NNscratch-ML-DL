# Example dataset
from ML.main import DecisionTreeClassifier
X = [[2.7], [1.5], [3.6], [4.5], [3.1], [2.0]]
y = [0, 0, 1, 1, 1, 0]

# Train
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
tree.fit(X, y)

# Predict
print("Predictions:", tree.predict([[2.0], [3.5]]))
print("Accuracy:", tree.score(X, y))


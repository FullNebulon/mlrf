from data.extract import unpickle

data_batches = []
labels_batches = []

for i in range(1, 6):
    batch = unpickle(f"../data/cifar-10-batches-py/data_batch_{i}")
    data_batches.append(batch[b'data'])
    labels_batches.append(batch[b'labels'])


import numpy as np

X_train = np.concatenate(data_batches) 
y_train = np.concatenate(labels_batches)

print("Data loaded")

### Executing grid search for SGD

print("Grid search for SGD")

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

param_grid = {
    'loss': ['hinge', 'log'],
    'alpha': [0.001, 0.01, 0.1, 1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training SGD model

print("Training SGD model")

model = SGDClassifier(loss='log', alpha=1,
                      max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

### Save the model

from joblib import dump
dump(model, 'models/sgd_flatten.joblib')

print("SGD model saved")

### Grid search for logistic regression

print("Grid search for Logistic regression model")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=20000)

param_grid = {
    'C': [0.01, 0.1, 1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training the model

print("Training Logistic regression model")

model = LogisticRegression(max_iter=1000, C=0.1)

model.fit(X_train, y_train)

### Save the model

dump(model, 'models/logistic_regression_flatten.joblib')

print("Logistic regression model saved")

### Grid search for random forest

print("Grid search for Random forest")

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training the model

print("Training random forest model")

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split= 5,min_samples_leaf=2, random_state=42)

randomForestModel.fit(X_train, y_train)

### Save the model

dump(model, 'models/random_forest_flatten.joblib')

print("random forest model saved")
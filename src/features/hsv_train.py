import numpy as np

from data.extract import load_cifar10_batch
from hsv_extract import extract_color_histogram


### Loading of training set

file_list = [f"../data/cifar-10-batches-py/data_batch_{i}" for i in range(1,6)]

images_train, labels_train = load_cifar10_batch(file_list)

X_train = []
for image in images_train:
    histogram = extract_color_histogram(image)
    X_train.append(histogram)

X_train = np.array(X_train)
y_train = np.array(labels_train)

print("Data loaded for hsv training")

### Grid search for Random Forest Classifier

print("Executing grid seach for Random Classifier")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

### Training of Random Forest Classifier

print("Training random forest classifier")

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=5, random_state=42)

randomForestModel.fit(X_train, y_train)

### Save the fitted model

from joblib import dump
dump(randomForestModel, 'models/random_forest_hsv.joblib')

print("hsv random forest saved")

### Grid search for Logistic Regression

print("Executing grid search for logistic regression")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=20000)

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training of Logistic Regression model

print("training logistic regression")

model = LogisticRegression(max_iter=20000, C=1, penalty='l2', random_state=42)

model.fit(X_train, y_train)

### Save the fitted model

dump(model, 'models/logistic_regression_hsv.joblib')

print("hsv logistic regression saved")

### Grid search for SGD

print("Executing grid search for sgd")

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

param_grid = {
    'loss': ['hinge', 'log'],
    'alpha': [0.001, 0.01, 0.1, 1],
    'penalty': ['l2', 'l1'],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training SGD model

print("training sgd model")

model = SGDClassifier(loss='log', alpha=0.001, penalty='l2',
                      max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

### Save the fitted model

dump(model, 'models/sgd_hsv.joblib')

print("hsv sgd saved")
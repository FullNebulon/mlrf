import numpy as np

from ..data.extract import load_cifar10_batch
from hog_extract import extract_hog_features


### Load of training set

file_list = [f"../../data/cifar-10-batches-py/data_batch_{i}" for i in range(1,6)]

images_train, labels_train = load_cifar10_batch(file_list)

X_train = []
for image in images_train:
    hog_features = extract_hog_features(image)
    X_train.append(hog_features)

X_train = np.array(X_train)
y_train = np.array(labels_train)

### Grid search for Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=20000)

param_grid = {
    'C': [0.01, 0.1, 1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training of Logistic Regression model

model = LogisticRegression()

model.fit(X_train, y_train)

### Save the model

from joblib import dump
dump(model, '../models/logistic_regression_hog.joblib')


### Grid seach for SGD

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

param_grid = {
    'loss': ['hinge', 'log'],
    'alpha': [0.001, 0.01, 0.1, 1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best parameters: ", best_params)

### Training of SGD model

model = SGDClassifier(loss='log', alpha=0.001, # 'hinge' pour un SVM linéaire, 'log' pour la régression logistique
                      max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

### Save the model

dump(model, '../models/sgd_hog.joblib')


### Grid search for Random Forest

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

### Training of Random Forest model

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split= 10,min_samples_leaf=2, random_state=42)

randomForestModel.fit(X_train, y_train)

### Save the model

dump(randomForestModel, '../models/random_forest_hog.joblib')
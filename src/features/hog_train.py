import numpy as np

from data.extract import load_cifar10_batch
from features.hog_extract import extract_hog_features


### Load of training set

file_list = [f"../data/cifar-10-batches-py/data_batch_{i}" for i in range(1,6)]

images_train, labels_train = load_cifar10_batch(file_list)

y_train = np.array(labels_train)
X_train = extract_hog_features(images_train)

print("Data for hog training loaded")

### Grid search for Logistic Regression

print("Executing grid search for logistic regression")

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

print("training logistic regression model")

model = LogisticRegression()

model.fit(X_train, y_train)

### Save the model

from joblib import dump
dump(model, 'models/logistic_regression_hog.joblib')

print("hog logosctic regression saved")

### Grid seach for SGD

print("Executing grid search for sgd")

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

print("training sgd model")

model = SGDClassifier(loss='log', alpha=0.001, # 'hinge' pour un SVM linéaire, 'log' pour la régression logistique
                      max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

### Save the model

dump(model, 'models/sgd_hog.joblib')

print("hog sgd saved")

## Grid search for Random Forest

print("Executing grid search for random forest")

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

print("training random forest model")

randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split= 10,min_samples_leaf=2, random_state=42)

randomForestModel.fit(X_train, y_train)

### Save the model

dump(randomForestModel, 'models/random_forest_hog.joblib')

print("hog random forest saved")
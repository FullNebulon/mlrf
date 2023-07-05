from data.extract import unpickle
from features.hog_extract import extract_hog_features
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

test_batch = unpickle("../data/cifar-10-batches-py/test_batch")
X_test = test_batch[b'data']
y_test = test_batch[b'labels']

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def roc_flatten(model, model_name):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred_bin = model.predict_proba(X_test)

    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown'])

    plt.figure(figsize=(7, 7))

    for i, color in zip(range(len(class_names)), colors):
        # Calculer la courbe ROC pour la i-ème classe
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])

        # Calculer l'aire sous la courbe ROC (AUC)
        roc_auc = auc(fpr, tpr)

        # Tracer la courbe ROC
        plt.plot(fpr, tpr, color=color, label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc:.2f})')

    # Tracer la ligne de hasard
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for ' + model_name + ' on flatten')
    plt.legend(loc="lower right")
    plt.savefig('visualization/'+model_name+'_flatten_roc.png')

### Loading the models

from joblib import load

random_forest = load('models/random_forest_flatten.joblib')

logistic_regression = load('models/logistic_regression_flatten.joblib')

sgd = load('models/sgd_flatten.joblib')

### Evaluation of random forest

y_pred = random_forest.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Random Forest Accuracy: {accuracy}")

### Computing confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Random Forest Confusion Matrix on flatten')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig('visualization/rand_forest_flatten_cm.png')

roc_flatten(random_forest, 'rand_forest')

### Evaluation of logistic regression

print("Logistic Regression Accuracy:", logistic_regression.score(X_test, y_test))

### Computing confusion matrix

y_pred = logistic_regression.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Logistic Regression Confusion Matrix on flatten')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig('visualization/log_reg_flatten_cm.png')

roc_flatten(logistic_regression, 'log_reg')

### Evaluation of SGD

from sklearn.metrics import accuracy_score

y_pred = sgd.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'SGD Accuracy: {accuracy}')

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('SGD Confusion Matrix on flatten')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig('visualization/sgd_flatten_cm.png')

roc_flatten(sgd, 'sgd')
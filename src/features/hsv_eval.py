import numpy as np

from ..data.extract import load_cifar10_batch
from hsv_extract import extract_color_histogram

### Loading of test set

images_test, labels_test = load_cifar10_batch(["../../data/cifar-10-batches-py/test_batch"])

X_test = []
for image in images_test:
    histogram = extract_color_histogram(image)
    X_test.append(histogram)

X_test = np.array(X_test)
y_test = np.array(labels_test)


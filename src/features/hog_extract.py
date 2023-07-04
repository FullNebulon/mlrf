import numpy as np
from skimage.feature import hog

def extract_hog_features(data):
    hog_features = []

    for img in data:
        # Reshape the image
        img_reshaped = np.reshape(img, (32, 32, 3))

        # Initialize a list to store the hog features of each channel
        hog_features_img = []

        for channel in range(img_reshaped.shape[2]):
            # Extract HOG features from the current channel
            feature = hog(img_reshaped[:, :, channel], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

            # Append the HOG features of the current channel to hog_features_img
            hog_features_img.append(feature)

        # Concatenate the HOG features of each channel and append them to hog_features
        hog_features.append(np.concatenate(hog_features_img))

    return np.array(hog_features)
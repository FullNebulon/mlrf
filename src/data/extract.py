import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_batch(files_list):
    images = None
    labels = None
    for file in files_list:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        if images is None:
            images = dict[b'data']
        else:
            images = np.concatenate((images, dict[b'data']))
        if labels is None:
            labels = dict[b'labels']
        else:    
            labels = np.concatenate((labels, dict[b'labels']))
            
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels
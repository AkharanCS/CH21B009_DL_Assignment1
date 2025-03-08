import keras
import numpy as np
from keras.datasets import fashion_mnist

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels_enc = one_hot_encode(train_labels,10)
test_labels_enc = one_hot_encode(test_labels,10)

print(train_labels[:10])
import keras
import numpy as np
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from NN import hidden_layer,output_layer,NeuralNetwork,activations
from GD import optimizer

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels = one_hot_encode(train_labels,10)
test_labels = one_hot_encode(test_labels,10)

nn2 = NeuralNetwork(2,[20,10],28*28,10,"sigmoid","cross_entropy")
nn2.build_network()
opt = optimizer("momentum_based_GD",2,1000,0.001,train_images[:1000],train_labels[:1000],test_images,test_labels,"cross_entropy")
opt.momentum_based_GD(nn2)
print(nn2.forward_pass(train_images[0].reshape(28*28,1)/255))
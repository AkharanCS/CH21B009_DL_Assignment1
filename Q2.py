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

nn1 = NeuralNetwork(2,[128,64],28*28,10,"sigmoid","cross_entropy")
nn1.build_network()
opt = optimizer("SGD",10,1,0.001,train_images[:1000],train_labels[:1000],test_images,test_labels,"cross_entropy")
opt.SGD(nn1)
print(nn1.forward_pass(train_images[5].reshape(28*28,1)/255))

    
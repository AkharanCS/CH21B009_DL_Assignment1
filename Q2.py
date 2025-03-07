import keras
import numpy as np
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from NN import hidden_layer,output_layer,NeuralNetwork,activations
from GD import optimizer
from utils import utils

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels = utils.one_hot_encode(train_labels,10)

nn1 = NeuralNetwork(2,[512,256],28*28,10,"sigmoid","cross_entropy")
nn1.build_network()


    
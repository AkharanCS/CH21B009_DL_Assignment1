import wandb
import keras
import numpy as np
import yaml
from keras.datasets import mnist
from matplotlib import pyplot as plt
from NN import hidden_layer,output_layer,NeuralNetwork,activations
from GD import optimizer
from utils import one_hot_encode


def train(config):
    nn = NeuralNetwork(config["n_layers"],[config["layer_size"] for _ in range(config["n_layers"])],28*28,10,config["activation_fun"],"cross_entropy",config["weight_init"])
    nn.build_network()
    optim = optimizer(config["opt"],config["epochs"],config["batch_size"],config["lr"],x_train[:10000],y_train_enc[:10000],x_val,y_val_enc,y_val,"cross_entropy",config["weight_decay"])
    epo,validation_loss,validation_acc = optim.optim_fun(nn)
    y_pred_test = nn.predict(x_test)
    test_error = optim.cross_entropy_loss(nn,x_test,y_test_enc,config["weight_decay"])
    test_acc = nn.accuracy_score(y_pred_test,y_test) 
    print("Validation Accuracy on the first configuration:",validation_acc[-1])
    print("Test Accuracy on the first configuration:",test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
x_val = train_images[:int(0.1*len(train_images))]/255
y_val = train_labels[:int(0.1*len(train_images))]
x_train = train_images[int(0.1*len(train_images)):]/255
y_train = train_labels[int(0.1*len(train_images)):]
x_test = test_images/255
y_test = test_labels

y_train_enc = one_hot_encode(y_train,10)
y_val_enc = one_hot_encode(y_val,10)
y_test_enc = one_hot_encode(y_test,10)


# Best three configurations that performed well on fashion_mnist
config1 = {"epochs":10, "n_layers":3, "layer_size":128,"weight_decay":0,
           "lr":0.001, "opt":"sgd", "batch_size":32, "weight_init":"Xavier", "activation_fun":"relu"}
config2 = {"epochs":10, "n_layers":5, "layer_size":128,"weight_decay":0,
           "lr":0.001, "opt":"sgd", "batch_size":16, "weight_init":"Xavier", "activation_fun":"relu"}
config3 = {"epochs":10, "n_layers":4, "layer_size":64,"weight_decay":0,
           "lr":0.0001, "opt":"adam", "batch_size":64, "weight_init":"Xavier", "activation_fun":"relu"}


# Configuration 1 running on mnist
print("Running Configuration 1 on mnist")
train(config1)

# Configuration 2 running on mnist
print("Running Configuration 2 on mnist")
train(config2)

# Configuration 3 running on mnist
print("Running Configuration 3 on mnist")
train(config3)


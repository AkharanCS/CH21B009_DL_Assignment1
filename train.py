import argparse
import wandb
import keras
import numpy as np
from keras.datasets import mnist,fashion_mnist
from NN import NeuralNetwork
from GD import optimizer
from utils import one_hot_encode

def train(config):
    if config["loss"] == "mean_squared_error":
        config["loss"] = "squared_error"
    
    if config["optimizer"] == "nag":
        config["activation"] = "nesterov"
    
    if config["activation"] == "ReLU":
        config["activation"] = "relu"
    
    nn = NeuralNetwork(config["num_layers"],[config["hidden_size"] for _ in range(config["num_layers"])],28*28,10,config["activation"],config["loss"],config["weight_init"])
    nn.build_network()
    optim = optimizer(config["optimizer"],config["epochs"],config["batch_size"],config["learning_rate"],x_train[:10000],y_train_enc[:10000],x_val,y_val_enc,y_val,config["loss"],config["weight_decay"])
    epo,validation_loss,validation_acc = optim.optim_fun(nn)
    for i in range(len(epo)):
        wandb.log({"epochs": epo[i], "val_loss": validation_loss[i], "val_accuracy": validation_acc[i]})
    y_pred_test = nn.predict(x_test)
    test_error = optim.cross_entropy_loss(nn,x_test,y_test_enc,config["weight_decay"])
    test_acc = nn.accuracy_score(y_pred_test,y_test) 
    wandb.log({"test_loss": test_error, "test_acc": test_acc})

# Setting up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname")
parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-m", "--momentum", type=float, default=0.5)
parser.add_argument("-beta", "--beta", type=float, default=0.5)
parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier")
parser.add_argument("-nhl", "--num_layers", type=int, default=3)
parser.add_argument("-sz", "--hidden_size", type=int, default=128)
parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "ReLU"], default="ReLU")

config = vars(parser.parse_args())

if config["dataset"] == "mnist":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_val = train_images[:int(0.1*len(train_images))]/255
    y_val = train_labels[:int(0.1*len(train_images))]
    x_train = train_images[int(0.1*len(train_images)):]/255
    y_train = train_labels[int(0.1*len(train_images)):]
    x_test = test_images/255
    y_test = test_labels

if config["dataset"] == "fashion_mnist":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    x_val = train_images[:int(0.1*len(train_images))]/255
    y_val = train_labels[:int(0.1*len(train_images))]
    x_train = train_images[int(0.1*len(train_images)):]/255
    y_train = train_labels[int(0.1*len(train_images)):]
    x_test = test_images/255
    y_test = test_labels

y_train_enc = one_hot_encode(y_train,10)
y_val_enc = one_hot_encode(y_val,10)
y_test_enc = one_hot_encode(y_test,10)

wandb.init(project=config["wandb_project"],entity=config["wandb_entity"])
train(config)
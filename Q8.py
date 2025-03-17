import wandb
import keras
import numpy as np
import yaml
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from NN import hidden_layer,output_layer,NeuralNetwork,activations
from GD import optimizer

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def train():
    wandb.init()

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

    config = wandb.config
    wandb.run.name = f"hl_{config.no_of_hidden_layers}_bs_{config.batch_size}_ac_"+config.activation_functions+"_opt_"+config.optimizer
    wandb.run.save()

    epochs = config.epochs
    n_layers = config.no_of_hidden_layers
    layer_size = config.size_of_every_hidden_layer
    weight_decay = config.weight_decay
    lr = config.learning_rate
    opt = config.optimizer
    batch_size = config.batch_size 
    weight_init = config.weight_initialisation
    activation_fun = config.activation_functions

    nn = NeuralNetwork(n_layers,[layer_size for _ in range(n_layers)],28*28,10,activation_fun,"squared_error",weight_init)
    nn.build_network()
    optim = optimizer(opt,epochs,batch_size,lr,x_train[:10000],y_train_enc[:10000],x_val,y_val_enc,y_val,"squared_error",weight_decay)
    epo,validation_loss,validation_acc = optim.optim_fun(nn)
    for i in range(len(epo)):
        wandb.log({"epochs": epo[i], "val_loss": validation_loss[i], "val_accuracy": validation_acc[i]})
    
    y_pred_test = nn.predict(x_test)
    test_error = optim.cross_entropy_loss(nn,x_test,y_test_enc,weight_decay)
    test_acc = nn.accuracy_score(y_pred_test,y_test) 
    wandb.log({"test_loss": test_error, "test_acc": test_acc})


with open("config.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep_config, project="Assignment1_Q8")
wandb.agent(sweep_id, function=train,count=20)

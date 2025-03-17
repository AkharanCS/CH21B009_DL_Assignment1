import wandb
import keras
import numpy as np
import yaml
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
from NN import hidden_layer,output_layer,NeuralNetwork,activations
from GD import optimizer
from utils import one_hot_encode

# hl_3_bs_32_ac_relu_opt_sgd - best run

wandb.init(project="Assignment1_Q7")

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

wandb.run.name = "hl_3_bs_32_ac_relu_opt_sgd - best run"
wandb.run.save()

# best parameters found using the previous experiments
epochs = 10
n_layers = 3
layer_size = 128
weight_decay = 0
lr = 0.001
opt = "sgd"
batch_size = 32
weight_init = "Xavier"
activation_fun = "relu"

# Training the network using best parameters
nn = NeuralNetwork(n_layers,[layer_size for _ in range(n_layers)],28*28,10,activation_fun,"cross_entropy",weight_init)
nn.build_network()
optim = optimizer(opt,epochs,batch_size,lr,x_train[:30000],y_train_enc[:30000],x_val,y_val_enc,y_val,"cross_entropy",weight_decay)
epo,validation_loss,validation_acc = optim.optim_fun(nn)
y_pred_test = nn.predict(x_test)
test_error = optim.cross_entropy_loss(nn,x_test,y_test_enc,weight_decay)
test_acc = nn.accuracy_score(y_pred_test,y_test) 
wandb.log({"test_loss": test_error, "test_accuracy": test_acc})
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=y_test, preds=y_pred_test,
                    class_names=[str(i) for i in range(10)])})


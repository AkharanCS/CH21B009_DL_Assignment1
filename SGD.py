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
x_val = train_images[:int(0.1*len(train_images))]/255
y_val = train_labels[:int(0.1*len(train_images))]
x_train = train_images[int(0.1*len(train_images)):]/255
y_train = train_labels[int(0.1*len(train_images)):]
x_test = test_images/255
y_test = test_labels

y_train_enc = one_hot_encode(y_train,10)
y_val_enc = one_hot_encode(y_val,10)
y_test_enc = one_hot_encode(y_test,10)


nn1 = NeuralNetwork(2,[512,256],28*28,10,"relu","cross_entropy")
nn1.build_network()
opt = optimizer("SGD",25,1,0.0001,x_train[:1000],y_train_enc[:1000],x_val,y_val_enc,"cross_entropy")
opt.SGD(nn1)
y_pred = nn1.predict(x_val)
print(y_pred[:10])
print(nn1.accuracy_score(y_pred,y_val))
    
import numpy as np
from NN import hidden_layer,output_layer,NeuralNetwork,activations

class optimizer():
    def __init__(self,optimizer_name,epochs,batch_size,learning_rate,train_x,train_y,val_x,val_y):
        self.opt_type = optimizer_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = learning_rate
        self.x_train = train_x
        self.y_train = train_y
        self.x_val = val_x
        self.y_val = val_y
    
    def SGD(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        batch_size = self.batch_size
        for i in range(epochs):
            for j in range(len(self.x_train)):
                x = self.x_train[j].reshape(28*28,1)/255
                y = self.y_train[j].reshape(10,1)
                yhat = nn.forward_pass(x)
                Loss = loss(y,yhat)
                w_grad_now,b_grad_now = nn.backpropagation(x,y)
                for k in range(len(nn.weights)):
                    nn.weights[k] -= lr*w_grad_now[k]
                    nn.bias[k] -= lr*b_grad_now[k]
            print(f"epoch {i}, Cost: {Loss}")

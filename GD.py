import numpy as np
from NN import hidden_layer,output_layer,NeuralNetwork,activations

class loss_after_epoch():
    def cross_entropy_loss(self,nn:NeuralNetwork,X,Y):
        error = 0.0
        for x,y in zip(X,Y):
            yhat = nn.forward_pass(x.reshape(28*28,1)/255)
            error += (-1*np.sum(y.T@np.log(yhat+0.0001)))
        return error
    
    def squared_error_loss(self,nn:NeuralNetwork,X,Y):
        error = 0.0
        for x,y in zip(X,Y):
            yhat = nn.forward_pass(x.reshape(28*28,1)/255)
            error += np.sum((y.T - yhat)**2)
        return error
    
class optimizer(loss_after_epoch):
    def __init__(self,optimizer_name,epochs,batch_size,learning_rate,train_x,train_y,val_x,val_y,loss):
        self.opt_type = optimizer_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = learning_rate
        self.x_train = train_x
        self.y_train = train_y
        self.x_val = val_x
        self.y_val = val_y
        if loss == "cross_entropy":
            self.loss = self.cross_entropy_loss
        if loss == "squared_error":
            self.loss = self.squared_error_loss
    
    def SGD(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        for i in range(epochs):
            for j in range(len(self.x_train)):
                x = self.x_train[j].reshape(28*28,1)/255
                y = self.y_train[j].reshape(10,1)
                yhat = nn.forward_pass(x)
                w_grad_now,b_grad_now = nn.backpropagation(x,y)
                for k in range(len(nn.weights)):
                    nn.weights[k] -= lr*w_grad_now[k]
                    nn.bias[k] -= lr*b_grad_now[k]
            loss = self.loss(nn,self.x_train,self.y_train)
            print(f"epoch {i}, Cost: {loss}")

import numpy as np
from NN import hidden_layer,output_layer,NeuralNetwork,activations

class loss_after_epoch():
    def cross_entropy_loss(self,nn:NeuralNetwork,X,Y):
        error = 0.0
        for x,y in zip(X,Y):
            yhat = nn.forward_pass(x.reshape(28*28,1)/255)
            error += (-1*np.sum(y.T@np.log(yhat+0.0001)))
        return error/len(X)
    
    def squared_error_loss(self,nn:NeuralNetwork,X,Y):
        error = 0.0
        for x,y in zip(X,Y):
            yhat = nn.forward_pass(x.reshape(28*28,1)/255)
            error += np.sum((y.T - yhat)**2)
        return error/len(X)
    
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
        print("Iniial Loss:",self.loss(nn,self.x_train,self.y_train))
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

    def momentum_based_GD(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        batch_size = self.batch_size
        beta = 0.9
        prev_uw = list(nn.weight_grad_cumm)
        prev_ub = list(nn.bias_grad_cumm)
        uw = list(nn.weight_grad_cumm)
        ub = list(nn.bias_grad_cumm)
        print("Iniial Loss:",self.loss(nn,self.x_train,self.y_train))
        for i in range(epochs):
            dw = list(nn.weight_grad_cumm)
            db = list(nn.bias_grad_cumm)
            for j in range(0,len(self.x_train),batch_size):
                x_batch = self.x_train[j:min(len(self.x_train),j + batch_size)]
                y_batch = self.y_train[j:min(len(self.x_train),j + batch_size)]
                for k in range(len(x_batch)):
                    x = x_batch[k].reshape(28*28,1)/255
                    y = y_batch[k].reshape(10,1)
                    yhat = nn.forward_pass(x)
                    w_grad_now,b_grad_now = nn.backpropagation(x,y)
                    for l in range(len(w_grad_now)):
                        dw[l] += w_grad_now[l]
                        db[l] += b_grad_now[l]
                for m in range(len(nn.weights)):
                    uw[m] = beta*prev_uw[m] + lr*dw[m]
                    ub[m] = beta*prev_ub[m] + lr*db[m]
                    nn.weights[m] -= lr*uw[m]
                    nn.bias[m] -= lr*ub[m]
                    prev_uw[m] = uw[m]
                    prev_ub[m] = ub[m]
            loss = self.loss(nn,self.x_train,self.y_train)
            print(f"epoch {i}, Cost: {loss}")
        

    def NAGD(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        batch_size = self.batch_size
        beta = 0.9
        prev_uw = list(nn.weight_grad_cumm)
        prev_ub = list(nn.bias_grad_cumm)
        uw = list(nn.weight_grad_cumm)
        ub = list(nn.bias_grad_cumm)
        print("Iniial Loss:",self.loss(nn,self.x_train,self.y_train))
        for i in range(epochs):
            dw = list(nn.weight_grad_cumm)
            db = list(nn.bias_grad_cumm)
            for n in range(len(uw)):
                uw[n] = beta*prev_uw[n]
                ub[n] = beta*prev_ub[n]
                nn.weights[n] -= uw[n]
                nn.bias[n] -= ub[n]
            for j in range(0,len(self.x_train),batch_size):
                x_batch = self.x_train[j:min(len(self.x_train),j + batch_size)]
                y_batch = self.y_train[j:min(len(self.x_train),j + batch_size)]
                for k in range(len(x_batch)):
                    x = x_batch[k].reshape(28*28,1)/255
                    y = y_batch[k].reshape(10,1)
                    yhat = nn.forward_pass(x)
                    w_grad_now,b_grad_now = nn.backpropagation(x,y)
                    for l in range(len(w_grad_now)):
                        dw[l] += w_grad_now[l]
                        db[l] += b_grad_now[l]
                for m in range(len(nn.weights)):
                    uw[m] = beta*prev_uw[m] + lr*dw[m]
                    ub[m] = beta*prev_ub[m] + lr*db[m]
                    nn.weights[m] -= uw[m]
                    nn.bias[m] -= ub[m]
                    prev_uw[m] = uw[m]
                    prev_ub[m] = ub[m]
            loss = self.loss(nn,self.x_train,self.y_train)
            print(f"epoch {i}, Cost: {loss}")


    def rms_prop(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        batch_size = self.batch_size
        beta = 0.9
        eps = 1e-4
        uw = list(nn.weight_grad_cumm)
        ub = list(nn.bias_grad_cumm)
        print("Iniial Loss:",self.loss(nn,self.x_train,self.y_train))
        for i in range(epochs):
            dw = list(nn.weight_grad_cumm)
            db = list(nn.bias_grad_cumm)
            for j in range(0,len(self.x_train),batch_size):
                x_batch = self.x_train[j:min(len(self.x_train),j + batch_size)]
                y_batch = self.y_train[j:min(len(self.x_train),j + batch_size)]
                for k in range(len(x_batch)):
                    x = x_batch[k].reshape(28*28,1)/255
                    y = y_batch[k].reshape(10,1)
                    yhat = nn.forward_pass(x)
                    w_grad_now,b_grad_now = nn.backpropagation(x,y)
                    for l in range(len(w_grad_now)):
                        dw[l] += w_grad_now[l]
                        db[l] += b_grad_now[l]
                
                for m in range(len(nn.weights)):
                    uw[m] = beta*uw[m] + (1-beta)*(dw[m]**2)
                    ub[m] = beta*ub[m] + (1-beta)*(db[m]**2)
                    nn.weights[m] -= lr*uw[m]/(np.sqrt(uw[m])+eps)
                    nn.bias[m] -= lr*ub[m]/(np.sqrt(ub[m])+eps)
            
            loss = self.loss(nn,self.x_train,self.y_train)
            print(f"epoch {i}, Cost: {loss}")


    def adam(self,nn:NeuralNetwork):
        epochs = self.epochs
        lr = self.eta
        batch_size = self.batch_size
        beta1, beta2 = 0.9,0.999
        eps = 1e-10
        uw = list(nn.weight_grad_cumm)
        ub = list(nn.bias_grad_cumm)
        mw = list(nn.weight_grad_cumm)
        mb = list(nn.bias_grad_cumm)
        uw_hat = list(nn.weight_grad_cumm)
        ub_hat = list(nn.bias_grad_cumm)
        mw_hat = list(nn.weight_grad_cumm)
        mb_hat = list(nn.bias_grad_cumm)
        print("Iniial Loss:",self.loss(nn,self.x_train,self.y_train))
        for i in range(epochs):
            dw = list(nn.weight_grad_cumm)
            db = list(nn.bias_grad_cumm)
            for j in range(0,len(self.x_train),batch_size):
                x_batch = self.x_train[j:min(len(self.x_train),j + batch_size)]
                y_batch = self.y_train[j:min(len(self.x_train),j + batch_size)]
                for k in range(len(x_batch)):
                    x = x_batch[k].reshape(28*28,1)/255
                    y = y_batch[k].reshape(10,1)
                    yhat = nn.forward_pass(x)
                    w_grad_now,b_grad_now = nn.backpropagation(x,y)
                    for l in range(len(w_grad_now)):
                        dw[l] += w_grad_now[l]
                        db[l] += b_grad_now[l]
                
            for m in range(len(nn.weights)):
                mw[m] = beta1*mw[m] + (1-beta1)*dw[m]
                mb[m] = beta1*mb[m] + (1-beta1)*db[m]
                uw[m] = beta2*uw[m] + (1-beta2)*(dw[m]**2)
                ub[m] = beta2*ub[m] + (1-beta2)*(db[m]**2)

                # Bias correction
                mw_hat[m] = mw[m]/(1-np.power(beta1,i+1))
                mb_hat[m] = mb[m]/(1-np.power(beta1,i+1))
                uw_hat[m] = uw[m]/(1-np.power(beta2,i+1))
                ub_hat[m] = ub[m]/(1-np.power(beta2,i+1))

                # Updating parameters
                nn.weights[m] -= lr*mw_hat[m]/(np.sqrt(uw_hat[m])+eps)
                nn.bias[m] -= lr*mb_hat[m]/(np.sqrt(ub_hat[m])+eps)
            
            loss = self.loss(nn,self.x_train,self.y_train)
            print(f"epoch {i}, Cost: {loss}")

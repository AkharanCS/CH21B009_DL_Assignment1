import numpy as np

class hidden_layer():
    def __init__(self,layer_index,num_neurons):
        self.layer_index = layer_index
        self.layer_size = num_neurons
        self.pre_activation = np.zeros((num_neurons,1),dtype=float)
        self.activation = np.zeros((num_neurons,1),dtype=float)
        self.gradient = np.zeros((num_neurons,1),dtype=float)

class output_layer():
    def __init__(self,no_of_classes):
        self.layer_size = no_of_classes
        self.pre_activation = np.zeros((no_of_classes,1),dtype=float)
        self.yhat = np.zeros((no_of_classes,1),dtype=float)
        self.gradient = np.zeros((no_of_classes,1),dtype=float)

class activations():
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        sig = 1 / (1 + np.exp(-x))
        return sig*(1-sig)
    
    def relu(self,x):
        return np.maximum(0, x)
    
    def relu_derivative(self,x):
        return (x > 0).astype(float)
    
    def tanh(self,x):
        return np.tanh(x)
    
    def tanh_derivative(self,x):
        return 1 - np.tanh(x)**2

    def softmax(self,x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class loss():
    def cross_entropy(self,y,yhat):
        return (-1*np.sum(y.T@np.log(yhat+0.0001)))
    
    def cross_entropy_der(self,y,yhat):
        return -1*(y-yhat)
    
    def squared_error(self,y,yhat):
        return np.sum((y - yhat)**2)
    
    def squared_error_der(self,y,yhat):
        return -2*(y - yhat)
    
class NeuralNetwork(hidden_layer,output_layer,activations,loss):
    def __init__(self, num_hidden_layers:int, num_neurons_in_each_layer: list, input_size: int, num_of_classes:int,hidden_activation,loss):
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_in_each_layer = num_neurons_in_each_layer
        self.output_size = num_of_classes
        self.layers = []

        self.weights = [np.random.randn(num_neurons_in_each_layer[0],input_size)]
        for i in range(1,num_hidden_layers):
            self.weights.append(np.random.randn(num_neurons_in_each_layer[i],num_neurons_in_each_layer[i-1]))
        self.weights.append(np.random.randn(self.output_size,num_neurons_in_each_layer[-1]))
        
        self.bias = []
        for i in range(len(self.weights)-1):
            self.bias.append(np.zeros((num_neurons_in_each_layer[i],1),dtype = float))
        self.bias.append(np.zeros((self.output_size,1),dtype=float))
        
        self.weight_grad = list(self.weights)
        self.bias_grad = list(self.bias)

        self.weight_grad_cumm = list(self.weights)
        for i in range(len(self.weight_grad_cumm)):
            for j in range(len(self.weight_grad_cumm[i])):
                self.weight_grad_cumm[i][j] = 0
        self.bias_grad_cumm = list(self.bias)

        if hidden_activation == "sigmoid":
            self.activation_fn = self.sigmoid
            self.activation_der = self.sigmoid_derivative
        if hidden_activation == "relu":
            self.activation_fn = self.relu
            self.activation_der = self.relu_derivative
        if hidden_activation == "tanh":
            self.activation_fn = self.tanh
            self.activation_der = self.tanh_derivative

        if loss == "cross_entropy":
            self.loss = self.cross_entropy
            self.loss_der = self.cross_entropy_der
        if loss == "squared_error":
            self.loss = self.squared_error
            self.loss_der = self.squared_error_der

    def build_network(self):
        for i in range(1,self.num_hidden_layers+1):
            self.layers.append(hidden_layer(layer_index=i,num_neurons=self.num_neurons_in_each_layer[i-1]))
        self.layers.append(output_layer(self.output_size))
    
    def forward_pass(self,input):
        hk = np.array(input)
        for i in range(len(self.layers)-1):
            self.layers[i].pre_activation = self.bias[i] + self.weights[i]@hk
            self.layers[i].activation = self.activation_fn(self.layers[i].pre_activation)
            hk = self.layers[i].activation
        self.layers[-1].pre_activation = self.bias[-1] + self.weights[-1]@hk
        self.layers[-1].yhat = self.softmax(self.layers[-1].pre_activation)
        return self.layers[-1].yhat
    
    def backpropagation(self,input,output):
        self.layers[-1].gradient = self.loss_der(output,self.layers[-1].yhat)
        for i in range(len(self.weights)-1,0,-1):
            # Computing gradient wrt parameters
            self.weight_grad[i] = self.layers[i].gradient@(self.layers[i-1].activation.T)
            self.bias_grad[i] = self.layers[i].gradient
            # Computing gradients wrt layer below
            grad_temp = self.weights[i].T@self.layers[i].gradient
            # Computing gradients wrt layer below (pre act)
            self.layers[i-1].gradient = grad_temp*(self.activation_der(self.layers[i-1].activation))
        self.weight_grad[0] = self.layers[0].gradient@(input.T)
        self.bias_grad[0] = self.layers[0].gradient
        return self.weight_grad,self.bias_grad





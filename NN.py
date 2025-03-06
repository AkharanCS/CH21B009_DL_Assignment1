import numpy as np

class hidden_layer():
    def __init__(self,layer_index,num_neurons):
        self.layer_index = layer_index
        self.layer_size = num_neurons
        self.pre_activation = np.zeros((num_neurons,1),dtype=float)
        self.activation = np.zeros((num_neurons,1),dtype=float)

class output_layer():
    def __init__(self,no_of_classes):
        self.layer_size = no_of_classes
        self.pre_activation = np.zeros((no_of_classes,1),dtype=float)
        self.yhat = np.zeros((no_of_classes,1),dtype=float)
    # gradient = -1*(el-yhat)

class NeuralNetwork(hidden_layer,output_layer):
    def __init__(self, num_hidden_layers:int, num_neurons_in_each_layer: list, input_size: int, num_of_classes:int):
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_in_each_layer = num_neurons_in_each_layer
        self.output_size = num_of_classes
        self.layers = []

        self.weights = [np.zeros((num_neurons_in_each_layer[0],input_size),dtype=float)]
        for i in range(1,num_hidden_layers):
            self.weights.append(np.zeros((num_neurons_in_each_layer[i],num_neurons_in_each_layer[i-1]),dtype=float))
        self.weights.append(np.zeros((self.output_size,num_neurons_in_each_layer[-1]),dtype=float))

        self.bias = []
        for i in range(len(self.weights)-1):
            self.bias.append(np.zeros((num_neurons_in_each_layer[i],1),dtype = float))
        self.bias.append(np.zeros((self.output_size,1),dtype=float))

    def build_network(self,n):
        for i in range(1,self.num_hidden_layers+1):
            self.layers.append(hidden_layer(layer_index=i,num_neurons=self.num_neurons_in_each_layer[i-1]))
        self.layers.append(output_layer(self.output_size))
    
nn = NeuralNetwork(2,[3,3],5,2)
print(nn.weights)
print(nn.bias)

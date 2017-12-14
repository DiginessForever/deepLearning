import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x: 1/(1 + np.exp(-x))
        #def sigmoid(x):
        #    return 1/(1+np.exp(-x))
        #self.activation_function = sigmoid

    #def activation_function(self, x):
    #    return 1/(1+np.exp(-x))

                   
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        #inputs = np.array(features, ndmin=2).T
        #targets = np.array(targets, ndmin=2).T
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch
        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(self.weights_input_to_hidden.T, X) # signals into hidden layer
        #print('hidden_inputs: ' +str(hidden_inputs))
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        #print('hidden_outputs: ' + str(hidden_outputs))

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(self.weights_hidden_to_output.T, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer  
        #print('final_inputs = final_outputs: ' + str(final_outputs))
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        output_errors = y - final_outputs
        #print('output_errors: ' + str(output_errors))   #A single number in an array.
        #hidden_errors = np.dot(self.weights_hidden_to_output, output_errors)
        #print('who - shape: ' + str(self.weights_hidden_to_output.shape) + ", " + str(self.weights_hidden_to_output))  #A (20,1) array
        who = np.reshape(self.weights_hidden_to_output, (len(self.weights_hidden_to_output)), order='C')
        #print('who shape: ' + str(who.shape) + ", " + str(who))
        hidden_errors = np.dot(output_errors, who[:,None].T)
        #print('hidden_errors: ' + str(hidden_errors))
        hidden_output_deriv = hidden_outputs * (1 - hidden_outputs)
        #print('hidden_grad: ' + str(hidden_output_deriv))
        hidden_error_term = hidden_output_deriv * hidden_errors
        #print('hidden error term: ' + str(hidden_error_term))

        #print('hidden outputs, shape: ' + str(hidden_outputs.shape) + ", " + str(hidden_outputs))
        #print('output_errors, shape: ' + str(output_errors.shape) + ", " + str(output_errors))
        #print('delta_weights_h_o, shape: ' + str(delta_weights_h_o.shape))

        for i in range(0, len(output_errors)):
            delta_weights_h_o +=  np.dot(hidden_outputs[:,None], output_errors[i])
            #print("for loop:  " + str(np.dot(hidden_outputs[:,None], output_errors[i])))

        #print('X, shape: ' + str(X.shape) + ", " + str(X))
        #print('hidden_error_term, shape: ' + str(hidden_error_term.shape) + ", " + str(hidden_error_term))
        #print('delta_weights_i_h, shape: ' + str(delta_weights_i_h.shape) + ", " + str(delta_weights_i_h))
        delta_weights_i_h += np.dot(X[:,None], hidden_error_term[:,None].T)
        

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 500
learning_rate = 0.1
hidden_nodes = 25
output_nodes = 1

'''
import pandas as pd
import unittest
import sys

###  READ STUFF IN
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
test_data = data[-21*24:]
data = data[:-21*24]
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

def MSE(y, Y):
    return np.mean((y-Y)**2)

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    print("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) + "% ... Training loss: " + str(train_loss)[:5] + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)'''
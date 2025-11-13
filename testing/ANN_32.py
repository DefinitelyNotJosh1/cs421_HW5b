# ANN.py - Artificial Neural Network
# Homework 5a for CS-421
# A very simple artificial neural network implementation in Python. 
# Authors:
# - Joshua Krasnogorov
# - Trenton Pham

import numpy as np
import os
import pandas as pd

# Define it as a class - will make the implementation in ReANTICS way easier if we do it this way
# Josh:Took inspiration from my own implementation of an ANN in ML class, but simplified it for this assignment
class ANN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, alpha, batch_size, stop_threshold, weights_and_biases_file):
        # Initialize weights and biases
        self.weights_and_biases_file = weights_and_biases_file
        if self.weights_and_biases_file:
            # if the file exists, great, load it up
            if os.path.exists(self.weights_and_biases_file):
                with open(weights_and_biases_file, "rb") as f:
                    data = np.load(f)
                    self.w1 = data["w1"]
                    self.b1 = data["b1"]
                    self.w2 = data["w2"]
                    self.b2 = data["b2"]
                    self.w3 = data["w3"]
                    self.b3 = data["b3"]
            else:
                # if it doesn't exist, create new weights and biases and a new file
                print(f"Weights and biases file {self.weights_and_biases_file} does not exist, creating new weights and biases and a new file")
                self.w1 = np.random.rand(input_size, hidden_size1) * 2 - 1 # -1 to 1
                self.b1 = np.random.rand(1, hidden_size1) * 2 - 1
                self.w2 = np.random.rand(hidden_size1, hidden_size2) * 2 - 1
                self.b2 = np.random.rand(1, hidden_size2) * 2 - 1
                self.w3 = np.random.rand(hidden_size2, output_size) * 2 - 1
                self.b3 = np.random.rand(1, output_size) * 2 - 1
                # save weights to new file
                with open(self.weights_and_biases_file, "wb") as f:
                    np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)
        else:
            # if no file is specified, create new weights and biases
            self.w1 = np.random.rand(input_size, hidden_size1) * 2 - 1 # -1 to 1
            self.b1 = np.random.rand(1, hidden_size1) * 2 - 1
            self.w2 = np.random.rand(hidden_size1, hidden_size2) * 2 - 1
            self.b2 = np.random.rand(1, hidden_size2) * 2 - 1
            self.w3 = np.random.rand(hidden_size2, output_size) * 2 - 1
            self.b3 = np.random.rand(1, output_size) * 2 - 1

        # other parameters
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size 
        self.alpha = alpha
        self.batch_size = batch_size
        self.accuracy_per_epoch = []
        self.error_per_epoch = [1]
        self.stop_threshold = stop_threshold

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    ##
    # forward
    #
    # Description: Propogates the input forward through the network.
    #
    # Parameters:
    #   input - the input to the network
    # 
    # Return: The output of the network after passing through the network.
    ##
    def forward(self, input):
        # Hidden layer
        self.z1 = np.dot(input, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Hidden layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # Output layer
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    ##
    # backward
    #
    # Description: Propogates the error backward through the network.
    #
    # Parameters:
    #   x_input - the input to the network
    #   y_output - the desired output of the network
    #
    # Return: Nothing, updates weights and biases in itself.
    ##
    def backward(self, x_input, y_output):
        # I divide deltas by n to get the average gradient across the batch to make learning rate independent of batch size
        n = x_input.shape[0]
        
        # Output layer error
        # Note: 
        #   for this assignment, omitting the sigmoid derivative makes convergence ~10x faster.
        #   I'll keep it so that it's consistent with what we did in class. May remove for 5b.
        error_output = (y_output - self.a3) * self.sigmoid_derivative(self.a3)

        # Calculate weights and biases for output layer
        dw3 = np.dot(self.a2.T, error_output) / n
        db3 = np.sum(error_output, axis=0, keepdims=True) / n

        # Hidden layer 2 error
        error_hidden2 = np.dot(error_output, self.w3.T) * self.sigmoid_derivative(self.a2)

        # Calculate weights and biases for hidden layer 2
        dw2 = np.dot(self.a1.T, error_hidden2) / n
        db2 = np.sum(error_hidden2, axis=0, keepdims=True) / n

        # Hidden layer 1 error
        error_hidden1 = np.dot(error_hidden2, self.w2.T) * self.sigmoid_derivative(self.a1)

        # Calculate weights and biases for hidden layer 1
        dw1 = np.dot(x_input.T, error_hidden1) / n
        db1 = np.sum(error_hidden1, axis=0, keepdims=True) / n

        # Update weights and biases
        self.w1 += self.alpha * dw1
        self.b1 += self.alpha * db1
        self.w2 += self.alpha * dw2
        self.b2 += self.alpha * db2
        self.w3 += self.alpha * dw3
        self.b3 += self.alpha * db3

    # Train the model
    def train(self, x_input, y_output):
        epoch = 1
        # Ensure targets are a (N, 1) column vector to avoid unintended broadcasting
        if y_output.ndim == 1:
            y_output = y_output.reshape(-1, 1)
        while (self.error_per_epoch[-1] > self.stop_threshold):
            epoch += 1
            # Shuffle data
            perm = np.random.permutation(len(x_input))
            x_input_shuffled = x_input[perm]
            y_output_shuffled = y_output[perm]

            # Batch training
            batch_count = len(x_input_shuffled) // self.batch_size
            for i in range(batch_count):
                batch_input = x_input_shuffled[i*self.batch_size:(i+1)*self.batch_size]
                batch_output = y_output_shuffled[i*self.batch_size:(i+1)*self.batch_size]
                if batch_output.ndim == 1:
                    batch_output = batch_output.reshape(-1, 1)

                self.forward(batch_input)
                self.backward(batch_input, batch_output)
            
            output_pred = self.forward(x_input)

            # Calculate accuracy over the full dataset
            accuracy = np.mean(output_pred == y_output)
            self.accuracy_per_epoch.append(accuracy)
            # print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")

            # Calculate average error over dataset for this epoch
            error = np.mean(np.abs(output_pred - y_output))
            self.error_per_epoch.append(error)

            # print every 100 epochs; wayyyy too many prints if every epoch
            if epoch % 100 == 0: 
                print(f"Epoch {epoch}, Error: {error:.8f}, Accuracy: {accuracy:.8f}")
                # save weights and biases every 100 epochs
                if self.weights_and_biases_file:
                    with open(self.weights_and_biases_file, "wb") as f:
                        np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)
                
                # trim accuracy and error lists to the last 100 epochs
                self.accuracy_per_epoch = self.accuracy_per_epoch[-100:]
                self.error_per_epoch = self.error_per_epoch[-100:]

                # lower learning rate as we get closer to the stop threshold
                if epoch % 1000 == 0:
                    self.alpha *= 0.95

            # if average errror is less than stop threshold, stop training
            if error < self.stop_threshold:
                print(f"Training stopped at epoch {epoch+1} at an error of {error:.4f} because average error is less than stop threshold of {self.stop_threshold:.4f}")


# Example training data, this is what we'll use


# Create an ANN
input_size = 32

output_size = 1
hidden_size1 = 96
hidden_size2 = 32
alpha = 3.0
batch_size = 5000
stop_threshold = 0.000000001

ann = ANN(input_size, hidden_size1, hidden_size2, output_size, alpha, batch_size, stop_threshold, "testing/weights32_96_32_Trenton.npz")

data = pd.read_csv("src/Trenton_mapping.csv")

# shuffle the rows b
data = data.sample(frac=1).reset_index(drop=True)

x_input = data.iloc[:, :-1].values
y_output = data.iloc[:, -1].values.reshape(-1, 1)

# get max output and min output
max_output = np.max(y_output)
min_output = np.min(y_output)
print("Max utility: ", max_output)
print("Min utility: ", min_output)
# normalize Y output to be 0 and 1
y_output = (y_output - min_output) / (max_output - min_output)

ann.train(x_input, y_output)

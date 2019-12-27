#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:21:11 2019

@author: Sandeep Reddy Gopu
"""

#importing the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import math

#loading the dataset
data = pd.read_csv("stock_data.csv")
data.head()
#print(data.head())
warnings.filterwarnings("ignore")
#selection of close column
data_for_prediction = data['Close'].values

#data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_prediction.reshape(-1,1))

scaled_data1=[]
for i in range(800):
    scaled_data1.append(scaled_data[i])

def window_data(data, window_size):
    x = []
    y = []
    i = 0
    while(i+window_size) <= len(data) - 1:
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i = i + 1
    assert len(x) == len(y)
    return x,y

X,Y = window_data(scaled_data, 7)
#print(X,Y)

#dividing into test and train data
X_train = np.array(X[0:556])
Y_train = np.array(Y[0:556])

X_test = np.array(X[556:662])
Y_test = np.array(Y[556:662])

print(" ")
print("LSTM algorithm for stock price predictions:")
print("As the dataset we are considering has more data, the execution of this code will take atleast 15 - 20 minutes of time. Please wait for the output.")

#defining the network
batch_size = 7
window_size = 7
hidden_layer = 256
clip_margin = 4
learning_rate = 0.001
epochs = 75

#defining the placeholders
inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, window_size,1])
target = tf.compat.v1.placeholder(tf.float32, [batch_size, 1])

#LSTM Weights
#weights for the input gate
def LSTM_input_gate_weights(hidden_layer, stddev):
    input_gate_weights = tf.Variable(tf.random.truncated_normal([1, hidden_layer], stddev = 0.05))
    hidden_input = tf.Variable(tf.random.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
    bias_input = tf.Variable(tf.zeros([hidden_layer]))
    return input_gate_weights, hidden_input, bias_input

    #weights for the forgot gate
def LSTM_forgot_gate_weights(hidden_layer, stddev):
    forgot_gate_weights = tf.Variable(tf.random.truncated_normal([1, hidden_layer], stddev = 0.05))
    hidden_forgot = tf.Variable(tf.random.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
    bias_forgot = tf.Variable(tf.zeros([hidden_layer]))
    return forgot_gate_weights, hidden_forgot, bias_forgot

def LSTM_output_gate_weights(hidden_layer, stddev):
    #weights for the output gate
    output_gate_weights = tf.Variable(tf.random.truncated_normal([1, hidden_layer], stddev = 0.05))
    hidden_output = tf.Variable(tf.random.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
    bias_output = tf.Variable(tf.zeros([hidden_layer]))
    return output_gate_weights, hidden_output, bias_output

def LSTM_memory_cell(hidden_layer, stddev):
   #weights for the memory cell
    memory_cell_weights = tf.Variable(tf.random.truncated_normal([1, hidden_layer], stddev = 0.05))
    hidden_memory_cell = tf.Variable(tf.random.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
    bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))
    return memory_cell_weights, hidden_memory_cell, bias_memory_cell

#output layer weight
def LSTM_output_layer(hidden_layer, stddev):
    weights_output = tf.Variable(tf.random.truncated_normal([hidden_layer, 1], stddev = 0.05))
    bias_output_layer = tf.Variable(tf.zeros([1]))
    return weights_output, bias_output_layer

#sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# derivative of sigmoid function

def sigmoid_derivative(x):
    return x * (1 - x)

#tanh function
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

#function to compute gate states
def LSTM_Cell(input, output, state):
    stddev = 0.05
    weights_input_gate, weights_input_hidden, bias_input = LSTM_input_gate_weights(hidden_layer, stddev)
    weights_forgot_gate, weights_forgot_hidden, bias_forgot = LSTM_forgot_gate_weights(hidden_layer, stddev)
    weights_output_gate, weights_output_hidden, bias_output = LSTM_output_gate_weights(hidden_layer, stddev)
    weights_memory_cell, weights_memory_cell_hidden, bias_memory_cell = LSTM_memory_cell(hidden_layer, stddev)
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate)+tf.matmul(output, weights_input_hidden)+bias_input)
    forgot_gate = tf.sigmoid(tf.matmul(input, weights_forgot_gate)+tf.matmul(output, weights_forgot_hidden)+bias_forgot)
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate)+tf.matmul(output, weights_output_hidden)+bias_output)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell)+tf.matmul(output, weights_memory_cell_hidden)+bias_memory_cell)
    state = state * forgot_gate + input_gate * memory_cell
    output = output_gate * tf.tanh(state)
    return state, output

#network loop
def network_output(hidden_layer):
    network_outputs = []
    weights_output, bias_output_layer = LSTM_output_layer(hidden_layer, 0.05)
    for i in range(batch_size):
        batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
        batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
        for j in range(window_size):
            batch_state, batch_output = LSTM_Cell(tf.reshape(inputs[i][j],(-1,1)), batch_state, batch_output)
        network_outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
    return network_outputs

#defining the loss
def loss(network_outputs):
    losses = []
    for i in range(len(network_outputs)):
        losses.append(tf.compat.v1.losses.mean_squared_error(tf.reshape(target[i], (-1,1)), network_outputs[i]))
    loss = tf.reduce_mean(losses)
    return loss

network_output = network_output(hidden_layer)
network_loss = loss(network_output)
gradients = tf.gradients(network_loss, tf.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
optimizer = tf.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

#training the network
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

for i in range(epochs):
    trained_output = []
    j = 0
    epoch_loss = []
    while(j+batch_size) <= len(X_train):
        X_batch = X_train[j:j+batch_size]
        Y_batch = Y_train[j:j+batch_size]
        o,c,_ = session.run([network_output,network_loss, trained_optimizer], feed_dict={inputs:X_batch, target:Y_batch})
        epoch_loss.append(c)
        trained_output.append(o)
        j = j + batch_size
    if (i % 5) == 0:
        print('Epoch {} of {} '.format(i, epochs), 'Current loss in the network = {}'.format(np.mean(epoch_loss)))

def trained_prediction(trained_output):
    trained_prediction =[]
    for i in range(len(trained_output)):
        for j in range(len(trained_output[i])):
            trained_prediction.append(trained_output[i][j][0])
    return trained_prediction

def test_prediction(network_output):
    test_data = []
    i = 0
    while i+batch_size <= len(X_test):
        o = session.run([network_output], feed_dict={inputs:X_test[i:i+batch_size]})
        i = i + batch_size
        test_data.append(o)

    test_data1= []
    for i in range(len(test_data)):
        for j in range(len(test_data[i][0])):
            test_data1.append(test_data[i][0][j])

    test_prediction = []
    for i in range(661):
        if i >= 556:
            test_prediction.append(test_data1[i-556])
        else:
            test_prediction.append(None)
    return test_prediction
#plot the predicted data
predictions_on_trained_data = trained_prediction(trained_output)
predictions_on_test_data = test_prediction(network_output)
plt.figure(figsize=(14, 7))
plt.plot(scaled_data1, label='Original data')
plt.plot(predictions_on_trained_data, label='Training data')
plt.plot(predictions_on_test_data, label='Testing data')
plt.legend()
plt.show()

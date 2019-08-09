import math
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05
for epoch in range(20000):
    inputs = feature_set
    XW = np.dot(feature_set, weights) + bias
    z = sigmoid(XW)
    error = z-labels
    #print (error.sum())

    derivative = error*der_sigmoid(z) #Multiply by inputs
    inputs = feature_set.T
    weights -= lr * np.dot(inputs, derivative)

    for num in derivative:
        bias -= lr * num

single_point = np.array([0,1,0])
result = sigmoid(np.dot(single_point, weights)+bias)
print ("ANSWER: ",result)

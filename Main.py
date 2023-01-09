from ANN import *
from Utils import *
import pandas as pd
import numpy

Structure = [784, 300, 10]
data = pd.read_csv('train.csv')
data = numpy.array(data)
m, n = data.shape
numpy.random.shuffle(data)
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

Weight, Biases = Train(500, Structure, X_train, Y_train, 41000//5, X_dev, Y_dev)
WriteWeights(Weight, Biases)
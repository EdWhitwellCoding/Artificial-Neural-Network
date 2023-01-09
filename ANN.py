import numpy
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from Utils import *

def Init(Structure):
    Weights = []
    Biases = []
    for Index in range(0, len(Structure)-1):
        Weights.append(numpy.random.rand(Structure[Index+1], Structure[Index])-0.5)
        Biases.append(numpy.random.rand(Structure[Index+1], 1)-0.5)

    return Weights, Biases

def Relu(Z):
    return numpy.maximum(Z, 0)

def ReluDeriv(Z):
    return Z > 0

def SoftMax(Z):
    print(Z.shape)
    Sum = sum(numpy.exp(Z))
    A = numpy.exp(Z) /  (Sum + 0.001)
    return A

def EncodeLable(Y, NumOptions):
    Desired = numpy.zeros((Y.size, NumOptions))
    Desired[numpy.arange(Y.size), Y] = 1 # Dersired is a 2d array the first dimension is covered by the numpy.arange(Y.size) which does the equivilent to the Indexer in a for loop, 
    return Desired


def ForwardPropagation(Inputs, Weights, Biases):
    Zs = [Inputs] # the net of the inputs * weights + bias
    As = [Inputs] # the out put of the nodes / the after activation
     
    Image.fromarray(Inputs[:, 0].reshape((28, 28)) * 255).convert("L").save("Test.png")

    for Index in range(0, len(Weights)):
        Z = (Weights[Index].dot(As[-1]) + Biases[Index])
        A = Relu(Z) if Index < len(Weights)-1 else SoftMax(Z)
        Zs.append(Z)
        As.append(A)

    return As, Zs

def Backpropagation(Activations, Desired, DataSetSize, Weights, NumLayers, Zs):
    print(Activations[-1].shape, Desired.shape)
    DerivZ = Activations[-1]-Desired.T   #Correct
    DerivativesW = [DerivZ.dot(Activations[-2].T)/DataSetSize]   #Correct
    DerivativesB = [numpy.sum(DerivZ, axis=1)/DataSetSize]   #Correct
    for Index in range(NumLayers-2, 0, -1):

        DerivZ = Weights[Index].T.dot(DerivZ) * ReluDeriv(Zs[Index])
        DerivativesW.append(DerivZ.dot(Activations[Index-1].T)/DataSetSize)
        DerivativesB.append(numpy.sum(DerivZ, axis=1)/DataSetSize)



    return DerivativesW, DerivativesB

def UpdateWeightsBiases(LearnRate, Weights, Biases, DerivW, DerivB):
    for Index in range(0, len(Weights)):
        Weights[Index] -= LearnRate * DerivW[-Index-1]
        Biases[Index]-= LearnRate * DerivB[-Index-1][numpy.newaxis].T

    return Weights, Biases

def get_predictions(Outputs):
    return numpy.argmax(Outputs, 0)

def get_accuracy(Outputs, Y):
    print(Outputs, Y)
    return numpy.sum(Outputs == Y) / Y.size

def Train(Iterations, Structure, Inputs, ExpectedOutput, EpochSize, Test, TestExpected):
    Weights, Biases = Init(Structure)
    NumEpochs = int(Inputs.shape[1] // EpochSize)
    Accuracy = 0
    InputsCopy = Inputs.copy()
    learnrate = 0.05
    Inputs = AddNoise(InputsCopy.copy(), 0.00005)
    for Index in range(0, Iterations):
        for Indexer in range(0, NumEpochs):
            Start = Indexer * EpochSize
            End = Start + EpochSize
            Activations, Nets = ForwardPropagation(Inputs[:, Start:End], Weights, Biases)
            DerivsW, DerivsB = Backpropagation(Activations, EncodeLable(ExpectedOutput[Start:End], 10), EpochSize, Weights, len(Structure), Nets)
            Weights, Biases = UpdateWeightsBiases(learnrate, Weights, Biases, DerivsW, DerivsB)
        if Index % 10 == 0:
            Inputs = AddNoise(InputsCopy.copy(), 0.00001)

            Activations, Nets = ForwardPropagation(Test.copy() , Weights, Biases)
            #Inputs = AddNoise(InputsCopy, random.uniform(0.001, 0.002))
            print("Iteration: ", Index)
            predictions = get_predictions(Activations[-1])
            Accuracy = get_accuracy(predictions, TestExpected)
            print(Accuracy)

                
        #if Index % 50 == 0:
        #    Inputs = RotateImage(InputsCopy)


    return Weights, Biases
import numpy
import random
import cv2
import scipy

def WriteWeights(Weights, Biases):
    for Index in range(0, len(Weights)):
        numpy.savetxt('Weights' + str(Index) + '.csv', Weights[Index], delimiter=',')
        numpy.savetxt('Biases' + str(Index) + '.csv', Biases[Index], delimiter=',')

def AddNoise(image, var):
      row,col= image.shape
      mean = 0
      sigma = var**0.5
      gauss = numpy.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy


def RotateImage(images):
    shape = images.shape[-1]
    images = images.reshape((28, 28, shape))
    for Index in range(0, images.shape[0]):
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((14,14), angle, 1.0)
        rotated_image = cv2.warpAffine(images[:, :, Index], M, (28, 28))
        images[:, :, Index] = rotated_image
    images = images.reshape((784, shape))
    return images

def ReadWeightsAndBiasesFromcsv():
    Weights, Biases = [], []
    for Index in range(0, 2):
        Weights.append(numpy.loadtxt("Weights" + str(Index) + ".csv", delimiter=","))
        Current = numpy.loadtxt("Biases" + str(Index) + ".csv", delimiter=",")
        Current = Current.reshape((Current.shape[0], 1))
        Biases.append(Current)

    return Weights, Biases

def recenter_images(images):
    shape = images.shape[-1]
    images = images.reshape((28, 28, shape))
    for Index in range(0, images.shape[0]):
        offsetted = scipy.ndimage.interpolation.shift(images[:, :, Index], (random.randint(-3, 3), random.randint(-3, 3)))
        images[:, :, Index] = offsetted

    return images.reshape((784, shape))
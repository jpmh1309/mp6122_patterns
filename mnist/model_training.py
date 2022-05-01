# Costa Rica Institute of Technology
# Course: MP-6122 Pattern Recognition
# Student: Jose Martinez Hdez
# Year: 2022
# Laboratory 3: CNN and ANN - MNIST Classification in Keras using a Jetson Nano

import tflearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn.datasets.mnist as mnist
from tensorflow.python.framework import ops

# Function for displaying a training image by it's index in the MNIST set
def show_digit(index, trainX=None, trainY=None):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    ops.reset_default_graph()
    
    # Include the input layer, hidden layer(s), and set how you want to train the model
    #Inputs
    net = tflearn.input_data([None, 784])
    
    #Hidden layers
    net = tflearn.fully_connected(net, 100, activation = 'ReLU')
    
    #Output
    net = tflearn.fully_connected(net, 10, activation = 'softmax')
    
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    
    # This model assumes that your network is named "net"    
    model = tflearn.DNN(net)
    return model


def main():

    # Retrieve the training and test data
    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

    # Display the first (index 0) training image
    print(trainY[0])
    show_digit(0, trainX, trainY)

    # Build the model
    model = build_model()

    # Training
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=20)

    # Compare the labels that our model predicts with the actual labels

    # Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
    predictions = np.array(model.predict(testX)).argmax(axis=1)

    # Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
    actual = testY.argmax(axis=1)
    test_accuracy = np.mean(predictions == actual, axis=0)

    # Print out the result
    print("Test accuracy: ", test_accuracy)

    # Save the entire model as a SavedModel.
    # !mkdir -p mnist/saved_model
    model.save('mnist/saved_model/my_model')

    # Reload a fresh Keras model from the saved model:
    new_model = build_model()
    new_model.load('mnist/saved_model/my_model')

    # Compare the labels that our model predicts with the actual labels

    # Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
    predictions = np.array(new_model.predict(testX)).argmax(axis=1)

    # Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
    actual = testY.argmax(axis=1)
    test_accuracy = np.mean(predictions == actual, axis=0)

    # Print out the result
    print("Test accuracy: ", test_accuracy)

if __name__ == "__main__":
    main()
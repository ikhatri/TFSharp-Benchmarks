# Copyright 2018 Ishan Khatri
# A fashionable neural network to classify the Fashion-MNIST dataset
# Based on: https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

print("TensorFlow Version: "+tf.__version__)

# Load training data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

python_model_name = "python_model.hd5"

# # Create model
# model = create_model()
# print("Python model created: ")
# model.summary()

# # Train model
# print("Begin training")
# model.fit(train_images, train_labels, epochs=10)
# print("Begin evaluation")
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

# # Save model
# model.save(python_model_name)
# print(colored("Python model saved to " + python_model_name, "blue"))

# Load Python model
loaded_model = keras.models.load_model(python_model_name)
loaded_model.summary()
loaded_model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(test_images, test_labels)
print("Test accuracy:", loaded_test_acc)

# Load C# model

# CS165B HW4 - Katyayani G. Raman (4803987)

import os
from PIL import Image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Function to load images from a given folder path and return them as numpy arrays
def get_data(folder_path, isTrainingData):
    filenames = os.listdir(folder_path)
    filenames = [f for f in filenames if not f.startswith('.DS_Store')]

    # initialize empty lists to hold training images and labels
    images = []
    labels = []

    if (isTrainingData):
        # loop over each subfolder in the folder path
        for subfolder in os.listdir(folder_path):
            # construct full path for the subfolder
            subfolder_path = os.path.join(folder_path, subfolder)
            # check if the subfolder is a directory
            if os.path.isdir(subfolder_path):
                # convert the subfolder name to an integer label
                label_num = int(subfolder)
                # loop over each image file in the subfolder
                for sample_image in os.listdir(subfolder_path):
                    # check if the file is a PNG image
                    if sample_image.endswith('.png'):
                        # construct full file path for the image
                        image_path = os.path.join(subfolder_path, sample_image)
                        # open the image file using PIL and extract its pixel data
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        vector_image = np.array(image.getdata())      
                        # append the pixel data and corresponding label to the training or testing lists
                        images.append(image_array)
                        labels.append(label_num)
    else:
        for filename in filenames:
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                image_array = np.array(image)
                images.append(image_array)

    return np.array(images), np.array(labels)

# Getting training + testing data
folder_path_training = './hw4_train'
folder_path_testing = './hw4_test'

train_images, train_labels = get_data(folder_path_training, True)
test_images, test_labels = get_data(folder_path_testing, False)

# convert integer labels to one-hot encoded vectors
num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

# Scaling values to be within the range (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Defining class names for the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Building the model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), input_shape=(28,28,1), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), input_shape=(28,28,1), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
  tf.keras.layers.Flatten(input_shape=train_images.shape),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])



# Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'], run_eagerly=True)

# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'], run_eagerly=True)

# Training the model
model.fit(train_images, train_labels, epochs=5)

# Evaluating the model on the test data
predictions = model.predict(test_images)
predictions = [np.argmax(predictions[index]) for index in range(len(predictions))]

# WRITE PREDICTIONS TO OUTPUT FILE
with open('prediction.txt', 'a') as file:
    for prediction in predictions:
        file.write(str(prediction) + '\n')
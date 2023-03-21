# CS165B HW4 - Katyayani G. Raman (4803987)

import os
from PIL import Image
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy as scp
import random

# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Set the random seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Configure TensorFlow ops to run deterministically.
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism() 

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
        # Sort the filenames in ascending order based on filename prefix, since it is an integer
        filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
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
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.25, random_state=1234)

# convert integer labels to one-hot encoded vectors
num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Scaling values to be within the range (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0
X_val = X_val / 255.0

# Reshape the input data to have a rank of 4
train_images = np.reshape(train_images, (-1, 28, 28, 1))

models = [tf.keras.models.load_model('model_1C.h5'), tf.keras.models.load_model('model_2C.h5'),
          tf.keras.models.load_model('model_3C.h5'), tf.keras.models.load_model('model_4C.h5'), 
          tf.keras.models.load_model('model_5C.h5')
          ]
predictions = []


for i, model in enumerate(models):
    prediction = model.predict(test_images)
    predictions.append(prediction)

predictions = np.asarray(predictions)

ensemble_pred = np.argmax(np.sum(predictions, axis=0), axis=1)

# Save the ensemble predictions to an output file
with open('prediction.txt', 'a') as file:
    for prediction in ensemble_pred:
        file.write(str(prediction) + '\n')

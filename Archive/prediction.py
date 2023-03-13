# CS165B HW4 - Katyayani G. Raman (4803987)
#hubs212
#gitignotr ignotus


# IMPORTS
import os
from PIL import Image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



def get_data(folder_path):
    filenames = os.listdir(folder_path)
    filenames = [f for f in filenames if not f.startswith('.DS_Store')]

    # initialize empty lists to hold training images and labels
    images = []
    labels = []

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
    return np.array(images), np.array(labels)

#GETTING TRAINING + TESTING DATA: 
folder_path_training = './hw4_train'
folder_path_testing = './hw4_test'
train_images, train_labels = get_data(folder_path_training)
test_images, test_labels = get_data(folder_path_testing)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#PRE-PROCESSING
#scaling values to be within the range (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0

#BUILDING MODEL: 

# model 1
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, kernel_size=(7, 7), input_shape=(28,28,1), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
  tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
  tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])

# #model 2: 
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=image_shape),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])

#COMPILING + TRAINING THE MODEL: 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

#EVALUATION:
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))
results = ["empty"]
if (len(test_images) != 0):
    results = model.evaluate(test_images,  test_labels, verbose=2)
print(results)
predictions = model.predict(test_images)
predictions = [np.argmax(predictions[index]) for index in range(len(predictions))]

#WRITE PREDICTIONS TO OUTPUT FILE
with open('prediction.txt', 'a') as file:
    for prediction in predictions:
        file.write(str(prediction) + '\n')





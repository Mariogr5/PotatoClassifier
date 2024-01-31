import math
import os
import random
import shutil
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
from platform import python_version

print('Python version:', python_version())
print('Numpy version:', np.__version__)
print('Seaborn version:', sns.__version__)
import tensorflow as tf

print('tensorflow version: ', tf.__version__)
print('keras version:', keras.__version__)

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


def prepare_datas(path):
    source = path
    target = './train_data/'
    shutil.copytree(source, target)
    os.mkdir('test_data')
    path = "./train_data/"
    for file in os.listdir(path):
        os.mkdir('./test_data/' + file)

    total_train_images, total_test_images, total_train_classes, total_test_classes = 0, 0, 0, 0
    path = "./train_data/"
    for file in os.listdir(path):
        total_train_classes += 1
        total_images = len(os.listdir(path + file + "/"))
        test_image_count = (25 / 100) * total_images  # 25% for test and 75% for train
        for i in range(math.ceil(test_image_count)):
            img = random.choice(os.listdir(path + file + '/'))
            shutil.move(path + file + '/' + img, './test_data/' + file + '/')
        print(file, total_images, math.ceil(test_image_count))
        total_train_images += (total_images - math.ceil(test_image_count))
    print("total train images are : ", total_train_images, " and total train classes are : ", total_train_classes)


class PotatoesNN():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Convolution2D(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='relu',
                                     input_shape=(32, 32, 3)))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='relu'
                                     ))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(units=80, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=7, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.training_set = train_datagen.flow_from_directory(
            './train_data/',
            target_size=(32, 32),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical')
        self.test_set = test_datagen.flow_from_directory(
            './test_data/',
            target_size=(32, 32),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical')

    def train(self, number_of_epochs):
        history = self.model.fit(
            self.training_set,
            steps_per_epoch=(336 / 32),
            epochs=number_of_epochs,
            validation_data=self.test_set,
            validation_steps=(115 / 32))

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'r', label='Training acc')
        plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def save_model(self, name_of_nn):
        self.model.save(name_of_nn)

    def get_training_set(self):
        return self.training_set

if __name__ == '__main__':
    prepare_datas('C:/Users/mario/Desktop/kerassieci/Datasheet_base')
    potatoes_model = PotatoesNN()
    potatoes_model.train(100)
    potatoes_model.save_model("potatoes_disease_detector_sample.h5")

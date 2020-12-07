import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import cv2
import warnings
import numpy as np
import argparse
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SIZE = 64
TARGET_SIZE = (SIZE,SIZE)
INPUT_SHAPE=(SIZE,SIZE,3)

def build_simple_model():
    # Build model
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=INPUT_SHAPE,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
    return model

def build_model_2():
    model=Sequential()
    
    model.add(Conv2D(100,(3,3),input_shape=INPUT_SHAPE,activation='relu',strides=2))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data/raw_dataset', help="Directory with the SIGNS dataset")
    parser.add_argument('-e', '--epochs', type=int, default=20, help="Where to write the new data")

    args = parser.parse_args()
    
    # Load and preprocess data
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    valid_dir = os.path.join(args.data_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, rotation_range=25, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')

    print(train_generator.class_indices)
    print(train_generator.image_shape)

    # # Build model
    model = build_simple_model()
    # model = build_model_2()
    model.summary()

    # Train model
    model_train = model.fit(train_generator, epochs=args.epochs, validation_data=valid_generator, batch_size=32)
    print(model_train)

    test_loss,test_accuracy = model.evaluate_generator(test_generator)
    print('test_loss: ',test_loss)
    print('test_accuracy: ',test_accuracy)

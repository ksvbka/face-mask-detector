from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, Xception, VGG16
from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dropout, \
                                    Flatten, Dense, BatchNormalization, \
                                    SpatialDropout2D, AveragePooling2D, Input
import os
import cv2
import warnings
import argparse
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('WARNING')

SIZE = 64
LEARNING_RATE = 0.0001
TARGET_SIZE = (SIZE,SIZE)
INPUT_SHAPE=(SIZE,SIZE,3)


def CNN_model():
    # Build model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=INPUT_SHAPE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

def MobileNetV2_model():
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])
    return model

def VGG16_model():
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARNING_RATE),metrics=["accuracy"])
    return model

def Xception_model():
    baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARNING_RATE),metrics=["accuracy"])
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='data/raw_dataset', help="Directory with the SIGNS dataset")
    parser.add_argument('-e', '--epochs', type=int, default=20, help="Where to write the new data")
    parser.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to output face mask detector model")
    args = parser.parse_args()

    # Load and preprocess data
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    valid_dir = os.path.join(args.data_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, zoom_range=0.2, shear_range=0.2, brightness_range=[0.9, 1.1], horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2)
    test_datagen  = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
    test_generator  = test_datagen.flow_from_directory(test_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')

    print(train_generator.class_indices)
    print(train_generator.image_shape)

    # Build model
    # model = CNN_model()
    model = MobileNetV2_model()
    # model = VGG16_model()
    # model = Xception_model()
    model.summary()

    # Train model
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='auto')
    model_train = model.fit(train_generator, epochs=args.epochs, validation_data=valid_generator, batch_size=32, verbose=1, callbacks=[earlystop], shuffle=True)
    print(model_train)

    test_loss,test_accuracy = model.evaluate_generator(test_generator)
    print('test_loss: ', test_loss)
    print('test_accuracy: ', test_accuracy)

    # serialize the model to disk
    print("saving mask detector model...")
    model.save(args.model, save_format="h5")

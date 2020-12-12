from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, Xception, VGG16, InceptionV3
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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', type=str, default='data/raw_dataset',
                    help="Directory of dataset")
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help="Where to write the new data")
parser.add_argument("-m", "--model", type=str, default="mask_detector.model",
                    help="Path to output face mask detector model")
parser.add_argument('-s', '--size', type=int, default=64,
                    help="Size of input data")
parser.add_argument('-b', '--batch-size', type=int, default=32,
                    help="Bactch size of data generator")
parser.add_argument('-l', '--learning-rate', type=float, default=0.0001,
                    help="Learning rate value")
parser.add_argument('-n', '--net-type', type=str, default='MobileNetV2',
                    choices=['CNN', 'MobileNetV2', 'VGG16', 'InceptionV3','Xception'],
                    help="The network architecture, optional: CNN, MobileNetV2, VGG16, Xception")

def CNN_model(learning_rate, input_shape):
    # Build model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
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
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], \
                  optimizer=Adam(learning_rate=learning_rate))
    return model

def MobileNetV2_model(learning_rate, input_shape):
    baseModel = MobileNetV2(include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], \
                  optimizer=Adam(learning_rate=learning_rate))
    return model

def VGG16_model(learning_rate, input_shape):
    baseModel = VGG16(include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], \
                  optimizer=Adam(learning_rate=learning_rate))
    return model

def Xception_model(learning_rate, input_shape):
    baseModel = Xception(include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile our model
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], \
                  optimizer=Adam(learning_rate=learning_rate))
    return model

if __name__ == "__main__":

    args = parser.parse_args()

    bs = args.batch_size
    lr = args.learning_rate
    size = (args.size, args.size)
    shape = (args.size, args.size, 3)
    epochs = args.epochs

    # Load and preprocess data
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    valid_dir = os.path.join(args.data_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, zoom_range=0.2, \
                                       shear_range=0.2, brightness_range=[0.9, 1.1], \
                                       horizontal_flip=True)                  
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=size, shuffle=True,
                                                        batch_size=bs, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=size, shuffle=True,
                                                        batch_size=bs, class_mode='binary')
    test_generator  = test_datagen.flow_from_directory(test_dir, target_size=size, shuffle=True,
                                                        batch_size=bs, class_mode='binary')

    print(train_generator.class_indices)
    print(train_generator.image_shape)

    # Build model
    net_type_to_model = {
        'CNN' : CNN_model,
        'MobileNetV2': MobileNetV2_model,
        'VGG16' : VGG16_model,
        'Xception' : Xception_model
    }
    model_name = args.net_type
    model_builder = net_type_to_model.get(model_name)
    model = model_builder(lr, shape)
    model.summary()

    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto')
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + f"-size-{size[0]}" + \
                                             f"-bs-{bs}" + f"-lr-{lr}.h5"),    \
                                              monitor='val_accuracy',save_best_only=True, verbose=1)
    # Train model
    model_train = model.fit(train_generator, epochs=epochs, validation_data=valid_generator,
                            batch_size=bs, callbacks=[earlystop, tensorboard, checkpoint], shuffle=True)
    print(model_train)

    test_loss,test_accuracy = model.evaluate(test_generator)
    print('test_loss: ', test_loss)
    print('test_accuracy: ', test_accuracy)

    # serialize the model to disk
    print("saving mask detector model...")
    model.save(args.model, save_format="h5")

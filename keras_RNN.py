# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation,Bidirectional,SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from livelossplot.keras import PlotLossesCallback
import tensorflow as tf
from keras import regularizers
import numpy as np
import matplotlib.image as mpimg
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
from keras.layers import Activation, Dropout, Flatten, Dense,Input,LSTM, GlobalMaxPooling2D, Conv2D, MaxPooling2D,ConvLSTM1D

training_data_dir = '../data/train_set'
validation_data_dir = "../data/valid_set"
test_data_dir = '../data/test_set'
RNN_WEIGHTS_DOGS_VS_CATS = r'output/RNN/cats_vs_dogs_rnn.h5'

IMAGE_SIZE = 256
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
nb_train_samples = 1500
nb_test_samples = 500
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001
TRAINING_LOGS_FILE = "output/RNN/training_logs_rnn.csv"
MODEL_SUMMARY_FILE = "output/RNN/model_summary_rnn.txt"

############################
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ConvLSTM1D(256,3,activation='relu',return_sequences=True))
model.add(ConvLSTM1D(128,3,activation='relu',return_sequences=True))
model.add(ConvLSTM1D(64,3,activation='relu',return_sequences=True))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = adam_v2.Adam(learning_rate=LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
with open(MODEL_SUMMARY_FILE, "w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# 数据增强
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)
test_data_generator = ImageDataGenerator(rescale=1. / 255)

# 数据准备
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    class_mode="binary",
    shuffle=False)

H = model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    verbose=1
)
model.save_weights(RNN_WEIGHTS_DOGS_VS_CATS)

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('output/RNN/rnn_plot.png')
plt.show()


def predict(TEST_SIZE):
    probabilities = model.predict_generator(test_generator, TEST_SIZE)
    for index, probability in enumerate(probabilities):
        image_path = test_data_dir + "/" + test_generator.filenames[index]
        img = mpimg.imread(image_path)
        plt.imshow(img)
        if probability > 0.5:
            plt.title("%.2f" % (probability[0] * 100) + "% dog")
        else:
            plt.title("%.2f" % ((1 - probability[0]) * 100) + "% cat")
        plt.show()

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import sys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x_data = np.load(sys.argv[1])
y_data = np.load(sys.argv[2])

train_size = 8000

x_train = x_data[:train_size].astype(float) / 255.0
y_train = tf.keras.utils.to_categorical(y_data[:train_size], 3)
x_test = x_data[train_size:].astype(float) / 255.0
y_test = tf.keras.utils.to_categorical(y_data[train_size:], 3)


# if __name__ == "__main__":
#     cv2.imshow("win", x_train[5])
#     if cv2.waitKey(0) > 0:
#         exit()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, 
                           kernel_size=(3, 3), activation='relu', 
                           input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test, verbose=2)
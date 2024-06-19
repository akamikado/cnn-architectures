import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train/255.0


def pad_image(image):
    return np.pad(image, ((2, 2), (2, 2)), 'constant')


# Requires 4 pixels of padding on because size of images are 28x28 pixels
x_train_padded = np.array([pad_image(img) for img in x_train])

# One hot encoding the labels so that they work better with softmax classifier
# y_train = tf.one_hot(y_train.astype(np.int32), depth=10)

model = models.Sequential()

# C1 Convolution layer
model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)))

# S2 Subsampling layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# C3 Convolution layer
model.add(layers.Conv2D(16, (5, 5), activation='tanh'))

# S4 Subsampling layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# C5 Convolution layer
model.add(layers.Conv2D(120, (5, 5), activation='tanh'))

# Flatten to 1D array to connect it with fully connected layer
model.add(layers.Flatten())

# F6 Fully connected layer
model.add(layers.Dense(84, activation='tanh'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
), metrics=['accuracy'])

epochs = 10

model.fit(x_train_padded, y_train, epochs=epochs, batch_size=128)

model.save('lenet5.keras')

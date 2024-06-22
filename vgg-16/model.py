import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

dataset_dir = '../datasets/caltech256/train'

train_ds = image_dataset_from_directory(
    dataset_dir,
    subset='training',
    validation_split=0.2,
    seed=128,
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    subset='validation',
    validation_split=0.2,
    seed=128,
    image_size=(224, 224),
    batch_size=32
)


def create_vgg16(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1,
              activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


model = create_vgg16((224, 224, 3), 232)

epochs = 50

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save('vgg16.keras')

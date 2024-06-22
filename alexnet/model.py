import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to training and validation directories
train_dir = '../datasets/tiny_imagenet/train'
val_dir = '../datasets/tiny_imagenet/val'

# Create ImageDataGenerator instances for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical'
)


def create_alexnet(input_shape, num_classes):

    model = models.Sequential()
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=4,
                            activation='relu', input_shape=input_shape))

    model.add(layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2)))

    model.add(layers.Conv2D(256, kernel_size=(
        5, 5), strides=1, activation='relu'))

    model.add(layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2)))

    model.add(layers.Conv2D(384, kernel_size=(
        3, 3), strides=1, activation='relu'))

    model.add(layers.Conv2D(384, kernel_size=(
        3, 3), strides=1, activation='relu'))

    model.add(layers.Conv2D(256, kernel_size=(
        3, 3), strides=1, activation='relu'))

    model.add(layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


input_shape = (224, 224, 3)

model = create_alexnet(input_shape, 200)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=50
)

model.save('alexnet.keras')

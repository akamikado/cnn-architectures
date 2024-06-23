import tensorflow as tf
from tensorflow.keras import layers, models

train_dir = '../datasets/flower_photos'

img_size = (224, 224)
batch_size = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)


def conv_layer(input, filters, kernel_size, strides):
    input = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding='same')(input)
    input = layers.BatchNormalization()(input)
    input = layers.Activation('relu')(input)
    return input


def residual_block(input, filters, kernel_sizes, strides, is_first_block_of_stage=False):
    shortcut = input

    x = conv_layer(input, filters[0], kernel_sizes[0], strides[0])
    x = conv_layer(x, filters[1], kernel_sizes[1], strides[1])
    x = conv_layer(x, filters[2], kernel_sizes[2], strides[2])

    if is_first_block_of_stage or input.shape[-1] != filters[2]:
        shortcut = layers.Conv2D(
            filters[2], (1, 1), strides=strides[0], padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50():
    input = layers.Input(shape=(224, 224, 3))

    x = conv_layer(input, 64, (7, 7), 2)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 2
    x = residual_block(x, [64, 64, 256], [(1, 1), (3, 3), (1, 1)], [
                       1, 1, 1], is_first_block_of_stage=True)
    x = residual_block(x, [64, 64, 256], [(1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [64, 64, 256], [(1, 1), (3, 3), (1, 1)], [1, 1, 1])

    # Stage 3
    x = residual_block(x, [128, 128, 512], [(1, 1), (3, 3), (1, 1)], [
                       2, 1, 1], is_first_block_of_stage=True)
    x = residual_block(x, [128, 128, 512], [(1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [128, 128, 512], [(1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [128, 128, 512], [(1, 1), (3, 3), (1, 1)], [1, 1, 1])

    # Stage 4
    x = residual_block(x, [256, 256, 1024], [(1, 1), (3, 3), (1, 1)], [
                       2, 1, 1], is_first_block_of_stage=True)
    x = residual_block(x, [256, 256, 1024], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [256, 256, 1024], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [256, 256, 1024], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [256, 256, 1024], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [256, 256, 1024], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])

    # Stage 5
    x = residual_block(x, [512, 512, 2048], [(1, 1), (3, 3), (1, 1)], [
                       2, 1, 1], is_first_block_of_stage=True)
    x = residual_block(x, [512, 512, 2048], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])
    x = residual_block(x, [512, 512, 2048], [
                       (1, 1), (3, 3), (1, 1)], [1, 1, 1])

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(1000, activation='softmax')(x)

    model = models.Model(inputs=input, outputs=output)

    return model


model = ResNet50()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=validation_ds, epochs=15)

model.save('resnet50.keras')

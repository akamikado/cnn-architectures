import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, ReLU, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

dataset_dir = '../datasets/flower_photos'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    subset='training',
    validation_split=0.2,
    seed=128,
    image_size=(299, 299),
    batch_size=8
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    subset='validation',
    validation_split=0.2,
    seed=128,
    image_size=(299, 299),
    batch_size=8
)


def conv_layer(input, filters, kernel_size, strides=(1, 1)):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding='same')(input)
    return x


def sp_conv_layer(input, filters, kernel_size):
    x = tf.keras.layers.SeparableConv2D(
        filters, kernel_size, padding='same')(input)
    return x


def entry_flow(input):
    x = conv_layer(input, 32, (3, 3), strides=(2, 2))
    x = ReLU()(x)

    x = conv_layer(x, 64, (3, 3))
    x = ReLU()(x)

    x1 = conv_layer(x, 128, (1, 1), strides=2)

    x = sp_conv_layer(x, 128, (3, 3))
    x = ReLU()(x)
    x = sp_conv_layer(x, 128, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x1, x])

    x1 = conv_layer(x, 256, (1, 1), strides=2)

    x = ReLU()(x)
    x = sp_conv_layer(x, 256, (3, 3))
    x = ReLU()(x)
    x = sp_conv_layer(x, 256, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x1, x])

    x1 = conv_layer(x, 728, (1, 1), strides=2)

    x = ReLU()(x)
    x = sp_conv_layer(x, 728, (3, 3))
    x = ReLU()(x)
    x = sp_conv_layer(x, 728, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x1, x])

    return x


def middle_flow(input):
    x1 = input
    for _ in range(8):
        x = ReLU()(x1)
        x = sp_conv_layer(x, 728, (3, 3))
        x = ReLU()(x)
        x = sp_conv_layer(x, 728, (3, 3))
        x = ReLU()(x)
        x = sp_conv_layer(x, 728, (3, 3))

        x1 = Add()([x1, x])

    return x1


def exit_flow(input):
    x = ReLU()(input)
    x = sp_conv_layer(x, 728, (3, 3))
    x = ReLU()(x)
    x = sp_conv_layer(x, 1024, (3, 3))

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x1 = conv_layer(input, 1024, (1, 1), strides=2)

    x = Add()([x1, x])

    x = ReLU()(x)
    x = sp_conv_layer(x, 1536, (3, 3))

    x = ReLU()(x)
    x = sp_conv_layer(x, 2048, (3, 3))

    x = GlobalAveragePooling2D()(x)

    x = Dense(1000, activation='softmax')(x)

    return x


def Xception():
    input = Input((299, 299, 3))
    x = entry_flow(input)
    x = middle_flow(x)
    x = exit_flow(x)

    model = Model(inputs=input, outputs=x)

    return model


model = Xception()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15)

model.save('xception.keras')

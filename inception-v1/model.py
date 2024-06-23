import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D

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


def Inception(input_layer, f1, f2_1, f2_3, f3_1, f3_5, f4):
    # path 1
    path1 = Conv2D(f1, (1, 1,), activation='relu', padding='same')(input_layer)

    # path 2
    path2 = Conv2D(f2_1, (1, 1), activation='relu',
                   padding='same')(input_layer)
    path2 = Conv2D(f2_3, (3, 3), activation='relu', padding='same')(path2)

    # path 3
    path3 = Conv2D(f3_1, (1, 1), activation='relu',
                   padding='same')(input_layer)
    path3 = Conv2D(f3_5, (5, 5), activation='relu', padding='same')(path3)

    # path 4
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(f4, (1, 1), activation='relu', padding='same')(path4)

    output = tf.keras.layers.Concatenate([path1, path2, path3, path4], axis=-1)

    return output


def GoogleNet():
    input_layer = tf.keras.Input(shape=(224, 224, 3))

    m = Conv2D(64, (7, 7), strides=2, activation='relu',
               padding='same')(input_layer)
    m = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(m)

    m = Conv2D(192, (3, 3), activation='relu', padding='same')(m)
    m = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(m)

    m = Inception(m, 64, 96, 128, 16, 32, 32)

    m = Inception(m, 128, 128, 192, 32, 96, 64)

    m = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(m)

    m = Inception(m, 192, 96, 208, 16, 48, 64)

    m1 = AveragePooling2D((5, 5), strides=3)(m)
    m1 = Conv2D(120, (1, 1), activation='relu', padding='same')(m1)
    m1 = Flatten()(m1)
    m1 = Dense(128, activation='relu')(m1)
    m1 = Dense(10, activation='softmax')(m1)

    m = Inception(m, 160, 112, 224, 24, 64, 64)

    m = Inception(m, 128, 128, 256, 24, 64, 64)

    m = Inception(m, 112, 144, 288, 32, 64, 64)

    m2 = AveragePooling2D((5, 5), strides=3)(m)
    m2 = Conv2D(120, (1, 1), activation='relu', padding='same')(m2)
    m2 = Flatten()(m2)
    m2 = Dense(128, activation='relu')(m2)
    m2 = Dense(10, activation='softmax')(m2)

    m = Inception(m, 256, 160, 320, 32, 128, 128)

    m = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(m)

    m = Inception(m, 256, 160, 320, 32, 128, 128)

    m = Inception(m, 384, 192, 389, 48, 128, 128)

    m = GlobalAveragePooling2D()(m)

    m = Dropout(0.4)(m)

    m = Dense(1000, activation='softmax')(m)

    model = tf.keras.Model(inputs=input_layer, outputs=[m, m1, m2])

    return model


model = GoogleNet()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=validation_ds, epochs=15)

model.save('googlenet.keras')

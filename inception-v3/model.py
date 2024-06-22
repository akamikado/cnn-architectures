import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dense, Concatenate, Dropout


def ConvLayer(input, filters, kernel_size=(1, 1), strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, activation='relu')(input)

    return x


def StemBlock(input):
    x = ConvLayer(input, 32, (3, 3), 2)
    x = ConvLayer(x, 32, (3, 3))
    x = ConvLayer(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = ConvLayer(x, 80)
    x = ConvLayer(x, 192, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    return x


def InceptionBlock_A(input, filters):
    x1 = ConvLayer(input, 64)
    x1 = ConvLayer(x1, 96, (3, 3))
    x1 = ConvLayer(x1, 96, (3, 3))

    x2 = ConvLayer(input, 48)
    x2 = ConvLayer(x2, 64, (3, 3))

    x3 = AveragePooling2D((3, 3), strides=(1, 1))(input)
    x3 = ConvLayer(x3, filters)

    x4 = ConvLayer(input, 64)

    x = Concatenate([x1, x2, x3, x4], axis=3)

    return x


def InceptionBlock_B(input, filters):
    x1 = ConvLayer(input, filters)
    x1 = ConvLayer(x1, filters, (7, 1))
    x1 = ConvLayer(x1, filters, (1, 7))
    x1 = ConvLayer(x1, filters, (7, 1))
    x1 = ConvLayer(x1, 192, (1, 7))

    x2 = ConvLayer(input, filters)
    x2 = ConvLayer(x2, filters, (1, 7))
    x2 = ConvLayer(x2, 192, (7, 1))

    x3 = AveragePooling2D()(input)
    x3 = ConvLayer(x3, 192)

    x4 = ConvLayer(input, 192)

    x = Concatenate([x1, x2, x3, x4], axis=3)

    return x


def InceptionBlock_C(input):
    x1 = ConvLayer(input, 448)
    x1 = ConvLayer(x1, 384, (3, 3))
    x1_1 = ConvLayer(x1, 384, (1, 3))
    x1_2 = ConvLayer(x1, 384, (3, 1))
    x1 = Concatenate([x1_1, x1_2], axis=3)

    x2 = ConvLayer(input, 384)
    x2_1 = ConvLayer(x2, 384, (1, 3))
    x2_2 = ConvLayer(x2, 384, (3, 1))
    x2 = Concatenate([x2_1, x2_2], axis=3)

    x3 = AveragePooling2D((3, 3), strides=(1, 1))(input)
    x3 = ConvLayer(x3, 192)

    x4 = ConvLayer(input, 320)

    x = Concatenate([x1, x2, x3, x4], axis=3)

    return x


def ReductionBlock_A(input):
    x1 = ConvLayer(input, 64)
    x1 = ConvLayer(x1, 96, (3, 3))
    x1 = ConvLayer(x1, 96, (3, 3), (2, 2))

    x2 = ConvLayer(input, 384, (3, 3), (2, 2))

    x3 = AveragePooling2D((3, 3), strides=(2, 2))(input)

    x = Concatenate([x1, x2, x3], axis=3)

    return x


def ReductionBlock_B(input):
    x1 = ConvLayer(input, 192)
    x1 = ConvLayer(x1, 192, (1, 7))
    x1 = ConvLayer(x1, 192, (7, 1))
    x1 = ConvLayer(x1, 192, (3, 3), (2, 2))

    x2 = ConvLayer(input, 192)
    x2 = ConvLayer(x2, 320, (3, 3), (2, 2))

    x3 = AveragePooling2D((3, 3), strides=(2, 2))(input)

    x = Concatenate([x1, x2, x3], axis=3)

    return x


def AuxillaryClassifier(input):
    x = AveragePooling2D((5, 5), strides=(3, 3))(input)
    x = ConvLayer(x, 128)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(768, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='softmax')(x)

    return x


def InceptionV3():
    input = tf.keras.Input(shape=(299, 299, 3))

    x = StemBlock(input)

    x = InceptionBlock_A(x, 32)
    x = InceptionBlock_A(x, 64)
    x = InceptionBlock_A(x, 64)

    x = ReductionBlock_A(x)

    x = InceptionBlock_B(x, 128)
    x = InceptionBlock_B(x, 160)
    x = InceptionBlock_B(x, 160)
    x = InceptionBlock_B(x, 192)

    Aux = AuxillaryClassifier(x)

    x = ReductionBlock_B(x)

    x = InceptionBlock_C(x)
    x = InceptionBlock_C(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='softmax')(x)

    model = tf.keras.models.Model(input, [x, Aux], name='Inception-v3')

    return model

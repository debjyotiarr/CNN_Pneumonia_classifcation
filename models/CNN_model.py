import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

def CNNModel(input_shape = (228,228,3), num_classes=2, kernel_size = 3, strides=1, padding = 'same'):
    model = Sequential([
        layers.Input(shape = input_shape, name = 'image_input'),

        #first block
        layers.Conv2D(32, kernel_size=kernel_size, activation='relu', strides = strides,padding = 'valid'),
        layers.Conv2D(32, kernel_size=kernel_size, activation='relu', strides = strides, padding=padding),
        layers.MaxPool2D(pool_size=2),
        layers.Dropout(0.2),

        #scond block
        layers.Conv2D(16, kernel_size=kernel_size, activation='relu', strides = strides, padding=padding),
        layers.Conv2D(16, kernel_size=kernel_size, activation='relu', strides = strides, padding=padding),
        layers.MaxPool2D(pool_size=2),
        layers.Dropout(0.2),

        #third block
        layers.Conv2D(8, kernel_size=kernel_size, activation='relu', strides = strides, padding=padding),
        layers.Conv2D(8, kernel_size=kernel_size, activation='relu', strides = strides, padding=padding),
        layers.MaxPool2D(pool_size=2),

        layers.Flatten(),

        #Dense Layers
        layers.Dense(128, activation='relu'),
        layer.Dropout(0.2),
        layers.Dense(num_classes-1, activation='sigmoid')
    ], name = 'cnnSequential')
    return model

def CNNModel2(input_size=(228,228,3), num_classes = 2, kernel_size = 3, strides=1, padding = 'same'):
    inputs = layers.Input(shape=input_size, name = 'input_image')
    #first
    x = layers.Conv2D(32, kernel_size=kernel_size, activation='relu', strides=strides, padding='valid')(inputs)
    x = layers.Conv2D(32, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.2)(x)

    #second
    x = layers.Conv2D(16, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = layers.Conv2D(16, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.2)(x)

    #third
    x = layers.Conv2D(8, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = layers.Conv2D(8, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Flatten()(x)

    #Dense layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
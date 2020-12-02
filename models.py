import tensorflow as tf
import keras
from keras import layers, optimizers


def load_resnet_based(input_shape=(100, 100, 3), learning_rate=1e-3):
    """
    Load pretrained ResNet-50 + Dense model

    Parameters
    ----------
    input_shape : tuple
        Input image shape (height, width, channels)

    learning_rate : float
        Model learning rate

    Returns
    -------
    Tensorflow model

    """

    from tensorflow.keras.applications import ResNet50

    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Make last 4 ResNet-50 layers trainable
    resnet.trainable = False

    for x in resnet.layers[-4:]:
        x.trainable = True
        # print(f'Layer: {x.name} | Parameters: {x.count_params()}')

    # Add Dense layers
    model = tf.keras.Sequential([
        resnet,
        layers.Dropout(0.5),
        layers.Flatten(),

        layers.Dense(100, activation='elu'),
        layers.Dropout(0.5),

        layers.Dense(50, activation='elu'),
        layers.Dropout(0.5),

        layers.Dense(10, activation='elu'),
        layers.Dropout(0.5),

        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mse']
    )
    return model


def load_nvidia(input_shape=(100, 100, 3), learning_rate=1e-3):
    """
    Load NVIDIA model

    Parameters
    ----------
    input_shape : tuple
        Input image shape (height, width, channels)

    learning_rate : float
        Model learning rate

    Returns
    -------
    Tensorflow model

    """

    model = keras.Sequential([
        layers.Conv2D(24, kernel_size=5, strides=2, padding='same', input_shape=input_shape, activation='relu'),
        layers.Conv2D(36, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.Conv2D(48, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['mse']
    )
    return model


if __name__ == '__main__':
    pass

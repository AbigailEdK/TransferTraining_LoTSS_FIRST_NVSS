# region ABOUT
# ===================================================================================================
# > This script defines the ConvXpress model architecture and a function to create the model based on a given configuration. The ConvXpress model is a convolutional neural network designed for image classification tasks, consisting of multiple convolutional layers with ReLU activation, max pooling, and dropout for regularization. The model is structured in blocks of convolutional layers with increasing filter sizes, followed by a fully connected layer and a softmax output layer for classification. The function get_model takes a configuration object, input shape, and number of classes as arguments and returns a compiled Keras Sequential model ready for training.

# > Adjusted from models.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer, LSTM, Bidirectional #type: ignore
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras import initializers, regularizers #type: ignore
# endregion

# region FUNCTIONS
def get_model(cf, input_shape, num_classes):
    return ConvXpress(cf, input_shape, num_classes)

def ConvXpress(cf, input_shape, num_classes):
    """Author of original: https://github.com/ezrafielding/Galaxy10-convXpress/blob/main/define_models.py"""

    '''
    ARGUMENTS:
    ----------
        > cf: Config object containing the configuration for the experiment,
        including hyperparameters and settings.
        > input_shape: The shape of the input data for the model.
        > num_classes: The number of output classes for the classification
        task.

    RETURNS:
    --------
        > model: A compiled Keras Sequential model based on the ConvXpress
        architecture, ready for training.

    DESCRIPTION:
    ------------
        The ConvXpress model is a convolutional neural network architecture designed for image classification tasks. It consists of multiple convolutional layers with ReLU activation, followed by max pooling and dropout for regularization. The architecture is structured in blocks of convolutional layers, with increasing filter sizes, and ends with a fully connected layer and a softmax output layer for classification. The model is compiled with the Adam optimizer, using the learning rate and regularization specified in the configuration object.

        The model architecture is as follows:
            - Input layer with the specified input shape.
            - Three convolutional layers with 32 filters, kernel size of
              (3, 3), ReLU activation, and 'same' padding, followed by max
              pooling and dropout.
            - Three convolutional layers with 64 filters, kernel size of
              (3, 3), ReLU activation, and 'same' padding, followed by max
              pooling and dropout.
            - Three convolutional layers with 128 filters, kernel size of
              (3, 3), ReLU activation, and 'same' padding, followed by max
              pooling and dropout.
            - Flatten layer to convert the 3D feature maps to a 1D vector.
            - Dense layer with 500 units, linear activation, and L2
              regularization.
            - Dropout layer for regularization.
            - Output dense layer with 'num_classes' units and softmax
              activation for classification.

    '''


    pool_stride = (2,2)

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3),strides=pool_stride,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))

    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))

    model.add(MaxPooling2D(pool_stride,padding='same'))
    
    model.add(Dropout(0.25, seed=cf.SEED))

    model.add(Conv2D(64, (3, 3),strides=pool_stride, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    
    model.add(Dropout(0.25, seed=cf.SEED))

    
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(MaxPooling2D(pool_size=pool_stride,padding='same'))
    
    model.add(Dropout(0.25, seed=cf.SEED))

    
    model.add(Flatten())
    dense_val = 500
    
    model.add(Dense(dense_val, activation='linear',kernel_regularizer=regularizers.l2(cf.REGULARISATION), kernel_initializer=initializers.glorot_uniform(seed=cf.SEED),bias_initializer='zeros'))
    
    model.add(Dropout(0.5, seed=cf.SEED))
    
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=cf.SEED),bias_initializer='zeros'))

    print("Returning ConvXpress")
    
    return model

# endregion
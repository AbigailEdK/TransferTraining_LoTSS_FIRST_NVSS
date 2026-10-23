# region ABOUT
# ===================================================================================================
# > This script defines the ConvXpress model architecture and a function to create the model based on a given configuration. The ConvXpress model is a convolutional neural network designed for image classification tasks, consisting of multiple convolutional layers with ReLU activation, max pooling, and dropout for regularization. The model is structured in blocks of convolutional layers with increasing filter sizes, followed by a fully connected layer and a softmax output layer for classification. The function get_model takes a configuration object, input shape, and number of classes as arguments and returns a compiled Keras Sequential model ready for training.

# > Adjusted from models.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer, LSTM, Bidirectional #type: ignore
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras import initializers, regularizers, models #type: ignore
import os
from tensorflow.keras import backend as K #type: ignore
import gc
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

    # print("Returning ConvXpress")
    
    return model

# def load_model_from_base(cf, input_shape, num_classes, fold):
#     '''
#     Loads the base model for transfer learning based on the configuration.

#     ARGUMENTS:
#     ----------
#         > cf: Config object containing the configuration for the experiment,
#         including hyperparameters and settings.

#     RETURNS:
#     --------
#         > model: A Keras Sequential model loaded with the weights of the
#         specified base model, ready for transfer learning.

#     DESCRIPTION:
#         This function loads the base model for transfer learning based on the
#         configuration provided. It constructs the file path to the saved base
#         model using the parameters specified in the configuration object, such
#         as the base model name, model architecture, dataset, number of epochs,
#         learning rate, and regularization. The function then loads the model
#         weights from the specified file path and returns the loaded model ready
#         for transfer learning.

#     '''

#     base_model_dir = os.path.join(cf.MODEL_DIR, f"baseNONE_model{cf.BASE_MODEL}_set{cf.SURVEYS}_base{cf.BASE_EPOCHS}e_{cf.EPOCHS}e_{cf.LEARNING_RATE*10}l_{cf.REGULARISATION}r", f"model_fold_{fold}.keras")

#     print(f"Loading base model from: {base_model_dir}")

#     # = cf.SAVE_DIR = process.folder_construct(base_model=cf.BASE_MODEL, model=cf.SURVEYS, epochs=cf.EPOCHS, surveys=cf.SURVEYS, lr=cf.LEARNING_RATE, reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Training/")
#     K.clear_session()

#     base_model = models.load_model(base_model_dir)  # Implement this function to load the pre-trained base model for transfer learning

#     base_model_clone = models.clone_model(base_model)  # Clone the base model to create a new instance for training

#     base_model_clone.set_weights(base_model.get_weights())  # Copy the weights from the base model to the cloned model
    
#     model = models.Sequential(base_model_clone.layers[:-1])  # Replace the final layer with a new one for our specific task

#     model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=cf.SEED),bias_initializer='zeros', name=f'output_layer_fold_{fold}'))  # Add a new output layer with the appropriate number of classes and activation function

#     return model

def load_model_from_base(cf, input_shape, num_classes, fold):
  # 1. Define paths safely
  base_model_dir = os.path.join(cf.MODEL_DIR, f"baseNONE_target{cf.BASE_MODEL}_dataset{cf.BASE_MODEL}_{cf.LEARNING_RATE*10}l_base{cf.BASE_REGULARISATION}r_{cf.BASE_REGULARISATION}r_ceiling{cf.EPOCH_CEILING}e", f"model_fold_{fold}.keras")
  
  print(f"Loading base model weights from: {base_model_dir}")

  # 2. Load the source pre-trained model to extract weights
  # We load this in a temporary variable so we can safely delete it after extraction
  pretrained_model = models.load_model(base_model_dir)

  # 3. Instantiate a completely fresh, separate model from your source code
  # (Replace 'build_untrained_model' with whatever function builds your model architecture)
  K.clear_session()
  new_model = ConvXpress(cf, input_shape=input_shape, num_classes=num_classes)

  # 4. Deep-copy weights layer-by-layer up to the penultimate layer
  # This leaves the final Dense layer untouched with its fresh, random initialization
  for i in range(len(new_model.layers) - 1):
    pretrained_weights = pretrained_model.layers[i].get_weights()
    new_model.layers[i].set_weights(pretrained_weights)

  # 5. Clean up the temporary pretrained model from memory
  del pretrained_model
  gc.collect()

  return new_model
# endregion
from tensorflow import keras

def conv_block(input_layer,filters):
    """Function to create a convolutional block in the UNet architecture
    Input:
       - input_layer: Input tensor representing the previous layer
       - filters: Number of filters for the convolutional layers
    Output:
       - pool: Output tensor after applying max pooling
       - conv: Output tensor of the last convolutional layer
    """
    conv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(input_layer)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(conv)
    conv = keras.layers.BatchNormalization()(conv)
    pool = keras.layers.MaxPooling2D((2, 2))(conv)
    return pool, conv

def middle_block(input_layer,filters):
    """Function to create the middle block of the UNet architecture
    Input:
       - input_layer: Input tensor representing the previous layer
       - filters: Number of filters for the convolutional layers
    Output:
       - conv: Output tensor of the last convolutional layer 
    """
    conv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(input_layer)
    conv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(conv)
    return conv


def deconv_block(input_layer, filters, conv_concat):
    """
    Function to create a deconvolutional (upsampling) block in the UNet architecture
    Input:
      - input_layer: Input tensor representing the previous layer
      - filters: Number of filters for the deconvolutional and convolutional layers
      - conv_concat: Tensor from the corresponding convolutional layer in the contracting path
    Output:
      - upconv: Output tensor of the last convolutional layer
    """
    deconv = keras.layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(input_layer)
    deconv = keras.layers.BatchNormalization()(deconv)
    concat = keras.layers.concatenate([deconv, conv_concat])
    upconv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(concat)
    upconv = keras.layers.BatchNormalization()(upconv)
    upconv = keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(upconv )
    upconv = keras.layers.BatchNormalization()(upconv)
    return upconv

def build_unet(img_shape,filters_sequence = [32,64,128]):
    """
    Function to build the UNet architecture
    Input:
       - img_shape:  Shape of the input image h,w,c
       - filters_sequence: List of numbers of filters for each convolutional block in the contracting path
    Output:
       - seg_model: Keras Model representing the UNet architecture (seg_model segmentation_model)
    """
    # Input layer
    input_layer = keras.layers.Input(img_shape, name = 'RGB_Input')
    # Layer to x for convenience
    x = input_layer
    # Collect layers for later concatenation
    conv_layers = []
    for filters in filters_sequence[:-1]:
        # building unet by x, also returning convolutions for concatenations
        x, conv = conv_block(x,filters)
        # collect layers
        conv_layers.append(conv)

    # middle block with final filters. filters_sequence[-1] 
    x = middle_block(x,filters_sequence[-1])

    # Now, using upsampling blocks. 
    # filters_sequence[::-1][1:] - reverse filters list and start from pre-last element
    # conv layers[::-1] - also reversed
    for filters, conv in zip(filters_sequence[::-1][1:],conv_layers[::-1]):
        x = deconv_block(x,filters,conv)
    output_layer = keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(x)
    seg_model = keras.models.Model(inputs=[input_layer], outputs=[output_layer])
    return seg_model

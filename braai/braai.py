import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import argparse

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import datetime
import numpy as np


def vgg6(input_shape=(63, 63, 3), n_classes: int = 1):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """

    # # batch norm momentum
    # batch_norm_momentum = 0.2

    model = tf.keras.models.Sequential(name='VGG6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    # model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # todo: try replacing FC with average pooling? see A. Karpaty's blog post from April 2019

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation='relu', name='fc_1'))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(tf.keras.layers.Dense(n_classes, activation=activation, name='fc_out'))

    return model


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                               name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.add([X_shortcut, X])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                               name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                        name=conv_name_base + '1',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut, training=1)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.add([X_shortcut, X])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50(input_shape=(63, 63, 3), n_classes: int = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape -- shape of the images of the dataset
    n_classes -- integer, number of classes. if = 1, sigmoid is used in the output layer; softmax otherwise
    Returns:
    model -- a Model() instance in Keras
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name='bn_conv1')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 1024], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 1024], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 1024], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = tf.keras.layers.Flatten()(X)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    X = tf.keras.layers.Dense(n_classes, activation=activation, name='fcOUT',
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def ResNet18(input_shape=(63, 63, 3), n_classes: int = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape -- shape of the images of the dataset
    n_classes -- integer, number of classes. if = 1, sigmoid is used in the output layer; softmax otherwise
    Returns:
    model -- a Model() instance in Keras
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name='bn_conv1')(X, training=1)
    X = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    # X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    # X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    # X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='b')
    # X = identity_block(X, 3, [256, 256, 512], stage=4, block='c')
    # X = identity_block(X, 3, [256, 256, 512], stage=4, block='d')
    # X = identity_block(X, 3, [256, 256, 512], stage=4, block='e')
    # X = identity_block(X, 3, [256, 256, 512], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 1024], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 1024], stage=5, block='b')
    # X = identity_block(X, 3, [512, 512, 1024], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = tf.keras.layers.Flatten()(X)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    X = tf.keras.layers.Dense(n_classes, activation=activation, name='fcOUT',
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet18')

    return model


def dense_block(x, blocks, name):
    """A dense block for DenseNet.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block for DenseNet.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    bn_axis = 3
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           epsilon=1.001e-5,
                                           momentum=batch_norm_momentum,
                                           name=name + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Conv2D(int(tf.keras.backend.int_shape(x)[bn_axis] * reduction), 1,
                               use_bias=False,
                               name=name + '_conv')(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    bn_axis = 3
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                            epsilon=1.001e-5,
                                            momentum=batch_norm_momentum,
                                            name=name + '_0_bn')(x)
    x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1,
                                use_bias=False,
                                name=name + '_1_conv')(x1)
    x1 = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                            epsilon=1.001e-5,
                                            momentum=batch_norm_momentum,
                                            name=name + '_1_bn')(x1)
    x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, 3,
                                padding='same',
                                use_bias=False,
                                name=name + '_2_conv')(x1)
    x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(input_shape=(63, 63, 3), n_classes: int = 1, include_top: bool = True):

    # batch norm momentum
    batch_norm_momentum = 0.2

    # densenet121
    blocks = [6, 12, 24, 16]
    # densenet169
    # blocks = [6, 12, 32, 32]
    # densenet201
    # blocks = [6, 12, 48, 32]

    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(X_input)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                           momentum=batch_norm_momentum, name='conv1/bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, momentum=batch_norm_momentum, name='bn')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)

    if include_top:
        # output layer
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        activation = 'sigmoid' if n_classes == 1 else 'softmax'
        x = tf.keras.layers.Dense(n_classes, activation=activation, name='fc_out')(x)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = tf.keras.Model(X_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = tf.keras.Model(X_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = tf.keras.Model(X_input, x, name='densenet201')
    else:
        model = tf.keras.Model(X_input, x, name='densenet')

    return model


def load_data(path: str = '/data', t_stamp: str = None, test_size=0.1, verbose: bool = True, random_state=None):

    # data:
    x = np.load(os.path.join(path, f'triplets.{t_stamp}.npy'))
    # classifications:
    y = np.load(os.path.join(path, f'labels.{t_stamp}.npy'))

    if random_state is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if verbose:
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


def save_report(path: str = '/data', stamp: str = None, report: dict = dict()):
    f_name = os.path.join(path, f'report.{stamp}.json')
    # import codecs
    # json.dump(report, codecs.open(f_name, 'w', encoding='utf-8'), separators=(',', ':'), indent=2)
    with open(f_name, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepStreaks')
    parser.add_argument('--t_stamp', type=str,
                        help='Dataset time stamp',
                        default='20190412_195358')
    parser.add_argument('--path_data', type=str,
                        help='Local path to data',
                        default='/data')
    parser.add_argument('--model', type=str,
                        help='Choose model to train: VGG6, ResNet18, ResNet50, DenseNet121',
                        default='VGG6')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size',
                        default=32)
    parser.add_argument('--loss', type=str,
                        help='Loss function: binary_crossentropy or categorical_crossentropy',
                        default='binary_crossentropy')
    parser.add_argument('--optimizer', type=str,
                        help='Optimized to use: adam or sgd',
                        default='adam')
    parser.add_argument('--epochs', type=int,
                        help='Number of train epochs',
                        default=300)
    parser.add_argument('--patience', type=int,
                        help='Early stop training if no val_acc improvement after this many epochs',
                        default=200)
    parser.add_argument('--test_split', type=int,
                        help='Training+validation/test split',
                        default=0.1)
    parser.add_argument('--validation_split', type=int,
                        help='Training/validation split',
                        default=0.1)
    parser.add_argument('--class_weight', action='store_true',
                        help='Weight training data by class depending on number of examples')
    parser.add_argument('--edge_tpu', action='store_true',
                        help='Perform quantization-aware training for further Edge TPU deployment')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose')

    args = parser.parse_args()
    t_stamp = args.t_stamp
    path_data = args.path_data

    # use same seed for repeatability
    random_state = 42

    # known models:
    models = {'VGG6': {'model': vgg6, 'grayscale': False},
              'ResNet18': {'model': ResNet18, 'grayscale': False},
              'ResNet50': {'model': ResNet50, 'grayscale': False},
              'DenseNet121': {'model': DenseNet, 'grayscale': False}
              }
    assert args.model in models, f'Unknown model: {args.model}'
    # grayscale = models[args.model]['grayscale']

    tf.keras.backend.clear_session()

    save_model = True

    ''' load data '''
    loss = args.loss
    binary_classification = True if loss == 'binary_crossentropy' else False
    n_classes = 1 if binary_classification else 2

    # load data
    X_train, Y_train, X_test, Y_test = load_data(path=path_data,
                                                 t_stamp=t_stamp,
                                                 test_size=args.test_split,
                                                 verbose=args.verbose,
                                                 random_state=random_state)

    candids = np.load(os.path.join(path_data, f'candids.{t_stamp}.npy'))
    labels = np.load(os.path.join(path_data, f'labels.{t_stamp}.npy'))
    _, _, mask_train, mask_test = train_test_split(labels, list(range(len(labels))),
                                                   test_size=args.test_split, random_state=random_state)
    masks = {'training': mask_train, 'test': mask_test}

    # training data weights
    if args.class_weight:
        # weight data class depending on number of examples?
        if not binary_classification:
            num_training_examples_per_class = np.sum(Y_train, axis=0)
        else:
            num_training_examples_per_class = np.array([len(Y_train) - np.sum(Y_train), np.sum(Y_train)])

        assert 0 not in num_training_examples_per_class, 'found class without any examples!'

        # fewer examples -- larger weight
        weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
        normalized_weight = weights / np.max(weights)

        class_weight = {i: w for i, w in enumerate(normalized_weight)}

    else:
        class_weight = {i: 1 for i in range(2)}

    print(f'Class weights: {class_weight}\n')

    # image shape:
    image_shape = X_train.shape[1:]
    print('Input image shape:', image_shape)

    print("Number of training examples = " + str(X_train.shape[0]))
    print("Number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    ''' build model '''
    model = models[args.model]['model'](input_shape=image_shape, n_classes=n_classes)
    # model = vgg4(input_shape=image_shape, n_classes=n_classes)

    # quantization-aware training for Edge TPU:
    if args.edge_tpu:
        print('quantization-aware training for Edge TPU')
        sess = tf.keras.backend.get_session()
        tf.contrib.quantize.create_training_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

    # set up optimizer:
    if args.optimizer == 'adam':
        # optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
        #                                      epsilon=None, decay=0.0, amsgrad=False)
        optimizer = tf.keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                             epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        # optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0)
    else:
        print('Could not recognize optimizer, using Adam')
        optimizer = tf.keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999,
                                             epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

    # print(model.summary())

    run_t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'deeprb_{t_stamp}_{model.name}_{run_t_stamp}'

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'/data/logs/{model_name}')

    patience = args.patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    batch_size = args.batch_size

    epochs = args.epochs

    # training without data augmentation
    # model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
    #           class_weight=class_weight,
    #           validation_split=0.05,
    #           verbose=1, callbacks=[tensorboard, early_stopping])

    # training with data augmentation:
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
    #                                                           vertical_flip=True,
    #                                                           validation_split=args.validation_split)
    data_augmentation = {'horizontal_flip': True,
                         'vertical_flip': True,
                         'rotation_range': 0,
                         'fill_mode': 'constant',
                         'cval': 1e-9}
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=data_augmentation['horizontal_flip'],
                                                              vertical_flip=data_augmentation['vertical_flip'],
                                                              rotation_range=data_augmentation['rotation_range'],
                                                              fill_mode=data_augmentation['fill_mode'],
                                                              cval=data_augmentation['cval'],
                                                              validation_split=args.validation_split)

    training_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='training')
    validation_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='validation')

    h = model.fit_generator(training_generator,
                            steps_per_epoch=len(X_train) // batch_size,
                            validation_data=validation_generator,
                            validation_steps=(len(X_train)*args.validation_split) // batch_size,
                            class_weight=class_weight,
                            epochs=epochs,
                            verbose=1, callbacks=[tensorboard, early_stopping])

    print('Evaluating on training set to check misclassified samples:')
    # preds = model.evaluate(X_train, Y_train, batch_size=batch_size)
    # print("Loss in prediction mode = " + str(preds[0]))
    # print("Training Accuracy in prediction mode = " + str(preds[1]))

    labels_training_pred = model.predict(X_train, batch_size=batch_size)
    # XOR will show misclassified samples
    misclassified_train_mask = np.array(list(map(int, labels[masks['training']]))).flatten() ^ \
                               np.array(list(map(int, np.rint(labels_training_pred)))).flatten()
    misclassified_train_mask = [ii for ii, mi in enumerate(misclassified_train_mask) if mi == 1]

    misclassifications_train = {int(c): [int(l), float(p)]
                                for c, l, p in zip(candids[masks['training']][misclassified_train_mask],
                                                   labels[masks['training']][misclassified_train_mask],
                                                   labels_training_pred[misclassified_train_mask])}
    # print(misclassifications_train)

    print('Evaluating on test set')
    preds = model.evaluate(X_test, Y_test, batch_size=batch_size)
    test_loss = float(preds[0])
    test_accuracy = float(preds[1])
    print("Loss = " + str(test_loss))
    print("Test Accuracy = " + str(test_accuracy))

    # save the full model [h5] and also separately weights [h5] and architecture [json]:
    model_save_name = f'/data/{model_name}'
    if True:
        model_save_name_h5 = f'{model_save_name}.h5'
        model.save(model_save_name_h5)

        model.save_weights(f'{model_save_name}.weights.h5')
        model_json = model.to_json()
        with open(f'{model_save_name}.architecture.json', 'w') as json_file:
            json_file.write(model_json)

    print(f'Batch size: {batch_size}')
    preds = model.predict(x=X_test, batch_size=batch_size)

    # XOR will show misclassified samples
    misclassified_test_mask = np.array(list(map(int, labels[masks['test']]))).flatten() ^ \
                              np.array(list(map(int, np.rint(preds)))).flatten()
    misclassified_test_mask = [ii for ii, mi in enumerate(misclassified_test_mask) if mi == 1]

    misclassifications_test = {int(c): [int(l), float(p)]
                               for c, l, p in zip(candids[masks['test']][misclassified_test_mask],
                                                  labels[masks['test']][misclassified_test_mask],
                                                  preds[misclassified_test_mask])}
    # print(misclassifications_test)

    # round probs to nearest int (0 or 1)
    labels_pred = np.rint(preds)
    confusion_matr = confusion_matrix(Y_test, labels_pred)
    confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

    print('Confusion matrix:')
    print(confusion_matr)

    print('Normalized confusion matrix:')
    print(confusion_matr_normalized)

    # generate training report in json format
    print('Generating report...')
    r = {'Dataset time stamp': t_stamp,
         'Run time stamp': run_t_stamp,
         'Model name': model_name,
         'Model trained': args.model,
         'Batch size': batch_size,
         'Optimizer': args.optimizer,
         'Requested number of train epochs': epochs,
         'Early stopping after epochs': patience,
         'Training+validation/test split': args.test_split,
         'Training/validation split': args.validation_split,
         'Weight training data by class': args.class_weight,
         'Quantization-aware training for Edge TPU': args.edge_tpu,
         'Random state': random_state,
         'Number of training examples': X_train.shape[0],
         'Number of test examples': X_test.shape[0],
         'X_train shape': X_train.shape,
         'Y_train shape': Y_train.shape,
         'X_test shape': X_test.shape,
         'Y_test shape': Y_test.shape,
         'Data augmentation': data_augmentation,
         'Test loss': test_loss,
         'Test accuracy': test_accuracy,
         'Confusion matrix': confusion_matr.tolist(),
         'Normalized confusion matrix': confusion_matr_normalized.tolist(),
         'Misclassified test candids': list(misclassifications_test.keys()),
         'Misclassified training candids': list(misclassifications_train.keys()),
         'Test misclassifications': misclassifications_test,
         'Training misclassifications': misclassifications_train,
         'Training history': h.history
         }
    for k in r['Training history'].keys():
        r['Training history'][k] = np.array(r['Training history'][k]).tolist()

    # print(r)

    save_report(path='/data', stamp=run_t_stamp, report=r)
    print('Done.')

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.regularizers import l2

from skreg.regularizers import GaussianReg
from skreg.initializers import GaussianInit


def load_cov(cov_dir, fname):
    return np.load(os.path.join(cov_dir,fname)).astype(np.float32)

def cnn(input_shape, nb_classes, use_fc=True, scale_conv=1., scale_fc=1.):
    layers = [
        # Conv, Pool
        Conv2D(5, kernel_size=5, strides=2, padding='same', activation='relu',
               kernel_regularizer=l2(scale_conv*0.05),
               input_shape=input_shape),
        MaxPooling2D(pool_size=3, strides=3),
        # Conv, Pool
        Conv2D(10, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(scale_conv*0.05)),
        MaxPooling2D(pool_size=3, strides=2),
        # Conv, Pool
        Conv2D(8, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(scale_conv*0.05)),
        MaxPooling2D(pool_size=3, strides=1),
        # Flatten
        Flatten()
    ]
    if use_fc:
        layers.append(Dropout(0.2))
        layers.append(Dense(128, activation='relu', kernel_regularizer=l2(scale_fc*0.01)))
    layers.append(Dropout(0.5))
    layers.append(Dense(nb_classes, activation='softmax'))
    # construct model
    model = Sequential(layers)

    return model

def cnn_sk(input_shape, nb_classes, cov_dir, use_fc=True,
           scale_conv=1., scale_fc=1., cg_init=True):
    # conv1
    #cov1 = load_cov(cov_dir, 'conv1_cov_rank3_sparse.npy')
    cov1 = load_cov(cov_dir, 'conv1_cov_rank3.npy')
    reg1 = GaussianReg(scale_conv*0.05, Cov=cov1, rank=3)
    # conv2
    cov2 = load_cov(cov_dir, 'conv2_cov_rank2_sparse.npy')
    #cov2 = load_cov(cov_dir, 'conv2_cov_rank2.npy')
    reg2 = GaussianReg(scale_conv*0.05, Cov=cov2, rank=2)
    # conv3
    cov3 = load_cov(cov_dir, 'conv3_cov_rank2_sparse.npy')
    #cov3 = load_cov(cov_dir, 'conv3_cov_rank2.npy')
    reg3 = GaussianReg(scale_conv*0.05, Cov=cov3, rank=2)
    if cg_init:
        init1 = GaussianInit(Cov=cov1, rank=3)
        init2 = GaussianInit(Cov=cov2, rank=2)
        init3 = GaussianInit(Cov=cov3, rank=2)
    else:
        init1, init2, init3 = 3*['glorot_uniform']
    layers = [
        # Conv, Pool
        Conv2D(5, kernel_size=5, strides=2, padding='same', activation='relu',
               kernel_regularizer=reg1, kernel_initializer=init1,
               input_shape=input_shape),
        MaxPooling2D(pool_size=3, strides=3),
        # Conv, Pool
        Conv2D(10, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_regularizer=reg2, kernel_initializer=init2),
        MaxPooling2D(pool_size=3, strides=2),
        # Conv, Pool
        Conv2D(8, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_regularizer=reg3, kernel_initializer=init3),
        MaxPooling2D(pool_size=3, strides=1),
        # Flatten
        Flatten()
    ]
    if use_fc:
        layers.append(Dropout(0.2))
        layers.append(Dense(128, activation='relu', kernel_regularizer=l2(scale_fc*0.01)))
    layers.append(Dropout(0.5))
    layers.append(Dense(nb_classes, activation='softmax'))
    # construct model
    model = Sequential(layers)

    return model
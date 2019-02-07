from __future__ import division, print_function
import os
import time
import shutil
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from skreg.util import get_image_dataset
from skreg.util import preprocess_images
from skreg.util import shuffle_images
from skreg.models import cnn, cnn_sk


def train_phase2(nb_epochs, results_dir, data_dir, gpu_id, correlated,
                 scale_conv=1., scale_fc=1.):

    # set TF session
    gpu_options = tf.GPUOptions(visible_device_list=gpu_id)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    time.sleep(0.5)
    print('')

    # reset results dir
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # load data
    X_train, Y_train = get_image_dataset(os.path.join(data_dir, 'train'))
    X_valid, Y_valid = get_image_dataset(os.path.join(data_dir, 'valid'))
    X_test, Y_test = get_image_dataset(os.path.join(data_dir, 'test'))
    # add noise to valid & test sets
    X_valid, Y_valid = shuffle_images(X_valid, Y_valid, rep=5)
    X_test, Y_test = shuffle_images(X_test, Y_test, rep=5)

    print('X_train shape: ', X_train.shape)
    print('Y_train shape: ', Y_train.shape)
    print('X_valid shape: ', X_valid.shape)
    print('Y_valid shape: ', Y_valid.shape)
    print('X_test shape: ', X_test.shape)
    print('Y_test shape: ', Y_test.shape)
    img_shape = X_train.shape[1:]
    nb_classes = Y_train.shape[1]

    # pre-process
    X_train = preprocess_images(X_train)
    X_valid = preprocess_images(X_valid)
    X_test = preprocess_images(X_test)

    # set other params
    bsize = 8 #4
    steps_per_epoch = X_train.shape[0]/bsize
    total_steps = steps_per_epoch*nb_epochs
    if nb_epochs >= 100:
        decay = (1/total_steps)*(1/0.5 - 1)
    else:
        decay = 0.
    print('batch size: %i' % bsize)
    print('learning rate decay: %0.6f' % decay)
    print('scale_conv: %0.2f' % scale_conv)
    # print('scale_fc: %0.2f' % scale_fc)

    # build CNN
    if correlated:
        print('Using correlated Gaussian prior')
        model = cnn_sk(
            input_shape=img_shape, nb_classes=nb_classes, use_fc=False,
            cov_dir='../data/gaussian_fit',
            scale_conv=scale_conv, scale_fc=scale_fc, cg_init=True
        )
    else:
        print('Using independent Gaussian prior')
        model = cnn(
            input_shape=img_shape, nb_classes=nb_classes, use_fc=False,
            scale_conv=scale_conv, scale_fc=scale_fc
        )
    model.compile(
        optimizer=Adam(decay=decay),
        loss='categorical_crossentropy',
        metrics=['ce', 'accuracy']
    )

    # model saving checkpoint
    weights_file = os.path.join(results_dir, 'cnn.h5')
    checkpoint = ModelCheckpoint(
        filepath=weights_file,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

    # train the model
    datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range=20,
        height_shift_range=20
    )
    flow_train = datagen.flow(
        X_train, Y_train, batch_size=bsize, shuffle=True, seed=5
    )
    start_time = time.time()
    hist = model.fit_generator(
        generator=flow_train,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epochs,
        validation_data=(X_valid, Y_valid),
        verbose=2,
        callbacks=[checkpoint]
    )
    end_time = time.time()
    print('Training took %0.1fs' % (end_time - start_time))

    # consolidate results
    train_losses = np.array(hist.history['loss'])
    train_CCEs = np.array(hist.history['ce'])
    train_accs = np.array(hist.history['acc'])
    valid_losses = np.array(hist.history['val_loss'])
    valid_CCEs = np.array(hist.history['val_ce'])
    valid_accs = np.array(hist.history['val_acc'])
    best_ix = np.argmin(valid_losses)
    print('best_ix: %i' % best_ix)
    print('BEST - loss: %0.4f - ce: %0.4f - acc: %0.4f' %
          (train_losses[best_ix], train_CCEs[best_ix], train_accs[best_ix]))
    print('BEST - val_loss: %0.4f - val_ce: %0.4f - val_acc: %0.4f' %
          (valid_losses[best_ix], valid_CCEs[best_ix], valid_accs[best_ix]))

    # load best model, test on holdout set
    model.load_weights(weights_file)
    _, test_CCE, test_acc = model.evaluate(X_test, Y_test, verbose=False)
    print('test_ce: %0.4f - test_acc: %0.4f \n' % (test_CCE, test_acc))

    # save results
    np.save(os.path.join(results_dir,'train_losses.npy'), train_losses)
    np.save(os.path.join(results_dir,'train_CCEs.npy'), train_CCEs)
    np.save(os.path.join(results_dir,'train_accs.npy'), train_accs)
    np.save(os.path.join(results_dir,'valid_losses.npy'), valid_losses)
    np.save(os.path.join(results_dir,'valid_CCEs.npy'), valid_CCEs)
    np.save(os.path.join(results_dir,'valid_accs.npy'), valid_accs)

    K.clear_session()

    return valid_CCEs[best_ix], valid_accs[best_ix], best_ix


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', default=300, type=int)
    parser.add_argument('--correlated', default=False, action='store_true')
    parser.add_argument('--scale_conv', default=1., type=float)
    parser.add_argument('--scale_fc', default=1., type=float)
    parser.add_argument('--data_dir', default='../data/silhouettes/phase2', type=str)
    parser.add_argument('--results_dir', default='./phase2_tmp', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()
    kwargs = vars(args)

    train_phase2(**kwargs)
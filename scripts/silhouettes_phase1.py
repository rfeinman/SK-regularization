from __future__ import division, print_function
import argparse
import getpass
import os
import shutil
import time
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from skreg.util import preprocess_images, preprocess_images_fb
from skreg.util import get_class_weights, shuffle_images
from skreg.models import cnn

def train_phase1(nb_epochs, results_dir, data_dir, gpu_id, fb, shuffle_val=False):
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
    if fb:
        print('using foreground-background colors')
        X = np.load(os.path.join(data_dir, 'X_phase1_fb.npy'))
        Y = np.load(os.path.join(data_dir, 'Y_phase1_fb.npy'))
        X = preprocess_images_fb(X)
    else:
        X = np.load(os.path.join(data_dir, 'X_phase1.npy'))
        Y = np.load(os.path.join(data_dir, 'Y_phase1.npy'))
        X = preprocess_images(X)
    img_shape = X.shape[1:]
    nb_classes = Y.shape[1]

    # train/val split
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=0.2, random_state=53, stratify=np.where(Y)[1]
    )
    if shuffle_val:
        X_valid, Y_valid = shuffle_images(X_valid, Y_valid, rep=5)
    print('X_train shape: ', X_train.shape)
    print('Y_train shape: ', Y_train.shape)
    print('X_valid shape: ', X_valid.shape)
    print('Y_valid shape: ', Y_valid.shape)

    # class weights
    print('Using class weighting to balance class counts')
    class_weights = get_class_weights(Y_train)

    # set other params
    bsize = min(32, int(X_train.shape[0]/5))
    steps_per_epoch = X_train.shape[0]/bsize
    total_steps = steps_per_epoch*nb_epochs
    if nb_epochs >= 100:
        decay = (1/total_steps)*(1/0.5 - 1)
    else:
        decay = 0.
    print('batch size: %i' % bsize)
    print('learning rate decay: %0.6f' % decay)

    # build CNN
    model = cnn(input_shape=img_shape, nb_classes=nb_classes)
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

    # data generator
    datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range=20,
        height_shift_range=20
    )
    flow_train = datagen.flow(X_train, Y_train, batch_size=bsize, shuffle=True)

    # train the model
    start_time = time.time()
    hist = model.fit_generator(
        generator=flow_train,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epochs,
        validation_data=(X_valid, Y_valid),
        verbose=2,
        callbacks=[checkpoint],
        class_weight=class_weights
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
    print('')

    # save results
    np.save(os.path.join(results_dir,'train_losses.npy'), train_losses)
    np.save(os.path.join(results_dir,'train_CCEs.npy'), train_CCEs)
    np.save(os.path.join(results_dir,'train_accs.npy'), train_accs)
    np.save(os.path.join(results_dir,'valid_losses.npy'), valid_losses)
    np.save(os.path.join(results_dir,'valid_CCEs.npy'), valid_CCEs)
    np.save(os.path.join(results_dir,'valid_accs.npy'), valid_accs)

    K.clear_session()


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', default=300, type=int)
    parser.add_argument('--fb', default=False, action='store_true')
    parser.add_argument('--shuffle_val', default=False, action='store_true')
    parser.add_argument('--results_dir', default='./phase1_tmp', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()
    kwargs = vars(args)

    # set data_dir location
    uname = getpass.getuser()
    if uname == 'rfeinman':
        kwargs['data_dir'] = '/Users/rfeinman/Data/BrownSilhouettes/preprocessed_200'
    elif uname == 'feinman':
        kwargs['data_dir'] = '/data/feinman/BrownSilhouettes/preprocessed_200'
    else:
        raise Exception

    train_phase1(**kwargs)
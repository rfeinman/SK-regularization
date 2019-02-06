from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


## Visualization ##

def view_kernels(w, scale=0.5, vlim='infer', title=None):
    if vlim == 'none':
        vlim = [None, None]
    elif vlim =='infer':
        vlim = [w.min(), w.max()]
    else:
        assert isinstance(vlim, list) or isinstance(vlim, tuple)
        assert len(vlim) == 2
    # record shape
    k, _, c, m = w.shape
    # set figure size
    width, height = (scale*m, scale*(c+1))
    fig, axes = plt.subplots(nrows=c, ncols=m, figsize=(width,height))
    for i in range(c):
        for j in range(m):
            axes[i,j].imshow(w[:,:,i,j],vmin=vlim[0],vmax=vlim[1],cmap='gray')
            axes[i,j].axis('off')
    if title is None:
        suptitle = 'mean: %0.3f (+/- %0.3f)' % (w.mean(), w.std())
    else:
        suptitle = title + '. mean: %0.3f (+/- %0.3f)' % (w.mean(), w.std())
    title_y = 1/(4.5*height) + 0.894
    title_y = np.minimum(title_y, 0.95)
    plt.suptitle(suptitle, y=title_y, fontsize=10)

def view_kernel_samples(w, scale=0.5, vlim='infer', gray=False):
    if vlim == 'none':
        vlim = [None, None]
    elif vlim =='infer':
        vlim = [w.min(), w.max()]
    else:
        assert isinstance(vlim, list) or isinstance(vlim, tuple)
        assert len(vlim) == 2
    # record shape
    m = w.shape[0]
    c = 1 if gray else w.shape[-1]
    # set figure size
    width, height = (scale*m, scale*(c+1))
    fig, axes = plt.subplots(nrows=c, ncols=m, figsize=(width,height))
    if gray:
        # w is shape (m, k, k)
        for i in range(m):
            axes[i].imshow(w[i], vmin=vlim[0], vmax=vlim[1], cmap='gray')
            axes[i].axis('off')
    else:
        # w is shape (m, k, k, c)
        for i in range(c):
            for j in range(m):
                axes[i,j].imshow(w[j,:,:,i], vmin=vlim[0], vmax=vlim[1], cmap='gray')
                axes[i,j].axis('off')
    suptitle = 'mean: %0.3f (+/- %0.3f)' % (w.mean(), w.std())
    title_y = 1/(4.5*height) + 0.894
    title_y = np.minimum(title_y, 0.95)
    plt.suptitle(suptitle, y=title_y, fontsize=10)

def view_covariance(Cov, figsize=(3,3)):
    plt.figure(figsize=figsize)
    plt.imshow(Cov)
    plt.colorbar()
    plt.title('Covariance')


## Covariance ##

def check_psd(X, eps=1e-8):
    """
    :param X: [(M,M) ndarray] matrix to check
    :param eps: [float] tolerance
    """
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1]
    assert np.all(np.abs(X - X.T) <= eps)
    eigvals, _ = np.linalg.eigh(X)
    assert np.all(eigvals >= -eps)

def cov(D):
    """
    :param D: (n, d) ndarray
        Centered data matrix of size nb_samples x nb_features
    """
    n = D.shape[0]
    return np.matmul(D.T, D) / (n-1)


## Image pre-processing ##

def normalize(X, mean, std):
    """
    Normalize RGB channels in place
    """
    # subtract mean
    X[..., 0] -= mean[0]
    X[..., 1] -= mean[1]
    X[..., 2] -= mean[2]
    # divide std-dev
    X[..., 0] /= std[0]
    X[..., 1] /= std[1]
    X[..., 2] /= std[2]

def preprocess_images(X):
    mean = np.asarray([0.8406, 0.8420, 0.8554], dtype=np.float32)
    std = np.asarray([0.3160, 0.3158, 0.2952], dtype=np.float32)
    assert len(X.shape) == 4 and X.shape[-1] == 3
    X_new = np.copy(X).astype(np.float32)

    # convert [0,255] scale to [0,1] if needed
    vmin = X_new[:20].min()
    vmax = X_new[:20].max()
    if vmax > 1:
        assert vmin >= 0 and vmax <= 255
        X_new = X_new / 255.

    # z-score
    normalize(X_new, mean, std)

    return X_new

def preprocess_images_fb(X):
    mean = np.asarray([0.4973, 0.4899, 0.5024], dtype=np.float32)
    std = np.asarray([0.2775, 0.2862, 0.2831], dtype=np.float32)
    assert len(X.shape) == 4 and X.shape[-1] == 3
    X_new = np.copy(X).astype(np.float32)

    # convert [0,255] scale to [0,1] if needed
    vmin = X_new[:20].min()
    vmax = X_new[:20].max()
    if vmax > 1:
        assert vmin >= 0 and vmax <= 255
        X_new = X_new / 255.

    # z-score
    normalize(X_new, mean, std)

    return X_new


## Misc ##

def get_class_weights(Y):
    y = np.where(Y)[1]
    classes = np.unique(y)
    # get counts for each class
    counts = {}
    for c in classes:
        counts[c] = np.count_nonzero(y==c)
    # get minimum count
    min_count = min(counts.values())
    # weight = min_count/count
    weights = {}
    for c in classes:
        weights[c] = min_count/counts[c]

    return weights

def shuffle_images(X_valid, Y_valid, rep=5, seed=12):
    assert X_valid.shape[0] == Y_valid.shape[0]
    n = X_valid.shape[0]
    datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range=20,
        height_shift_range=20
    )
    flow = datagen.flow(X_valid, Y_valid, batch_size=n, shuffle=False, seed=seed)
    flow.reset()
    X, Y = [], []
    for i, (X_batch, Y_batch) in enumerate(flow):
        if i >= rep:
            break
        X.append(X_batch)
        Y.append(Y_batch)
    X, Y = np.concatenate(X), np.concatenate(Y)

    return X, Y

def get_image_dataset(data_dir, img_size=(200,200), batch_size=32):
    datagen = ImageDataGenerator()
    flow = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, shuffle=False
    )
    nb_batches = int(np.ceil(flow.n/batch_size))
    flow.reset()
    X, Y = [], []
    for i, (X_batch, Y_batch) in enumerate(flow):
        if i >= nb_batches:
            break
        X.append(X_batch)
        Y.append(Y_batch)
    X, Y = np.concatenate(X), np.concatenate(Y)
    X /= 255.

    return X, Y
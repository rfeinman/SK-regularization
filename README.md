# SK-regularization

Code for the paper "SK-reg: A smooth kernel regularizer for convolutional neural networks" (Feinman & Lake, 2019).

## 1) Requirements & setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, add the path to the repository to your `PYTHONPATH` environment variable to enable imports from any folder:

    export PYTHONPATH="/path/to/SK-regularization:$PYTHONPATH"
    
## 2) Running the experiments

### Download the silhouettes dataset

First, download the pre-processed silhouettes image dataset from the following link:
<http://www.cns.nyu.edu/~reuben/files/silhouettes.zip>.

Unzip the folder and place it into the `data/` directory. The dataset contains two sub-folders: `phase1/` and `phase2/`. The `phase1/` folder contains an image dataset with the 20 Phase 1 classes we describe in our paper. The `phase2/` directory contains a dataset with the 10 Phase 2 classes that we describe. Phase 2 classes are distinct from Phase 1.

### Phase 1 silhouettes training

Once you've downloaded & unzipped the silhouettes folder and placed it in the `data/` directory, you will next train the CNN on the Phase 1 (20-way) classification task. From the `experiments/` directory, you can test a single training run with the following command:

    python silhouettes_phase1.py

This will train the CNN for 300 epochs using a single GPU (if available), and performance metrics for the train and validation sets will be reported. The model will be saved in a folder called `phase1_tmp/`. You can discard the save folder; this was simply a test.

Once you've tested the CNN, you can begin the phase 1 experiment loop. This loop will train the CNN 20 times, using a different random seed for each training run. The resulting CNN will be saved for each training run. To begin the training loop, run the following command from the `experiments/` directory:

    python silhouettes_phase1_loop.py
    
Results from the 20 trials will be saved to the `data/` directory in a folder called `kernel_dataset/`. You will be using the learned convolution kernels from these 20 training runs to determine the covariance parameters of SK-reg. 

### Gaussian fitting

Once you have completed Phase 1 training (with results saved in `data/kernel_dataset/`) you can now fit a multivariate Gaussian for each convolution layer of the CNN to obtain SK-reg parameters. To do so, cd to `experiments/` and open the Jupyter Notebook titled `fit_gaussians.ipynb`. Execute the notebook boxes in order. Once completed, you will have a new folder located at `data/gaussian_fit` containing the SK-reg parameters for each convolution layer. While executing this notebook, you can see some nice visualizations of the fitted Gaussians, and you can also see log-likelihood metrics for the Gaussian fits.

### Phase 2 silhouettes training

Once you've fitted the Gaussian distributions to the kernels from phase 1, you can now apply SK-reg to a new learning task in phase 2. To train the CNN on the new Phase 2 (10-way) classification task, cd to the `experiments/` directory and run the following command:

    python silhouettes_phase2.py --mode=<reg mode> --alpha=<reg weight>.
    
where `<reg mode>` is one of either `l2` or `sk` and, and `<reg weight>` is a float specifying how much to weight regularization vs. classification loss. With parameter `--mode=sk` you will apply SK-reg, using the Gaussian covariance matrices acquired from phase 1. With parameter `--mode=l2` you will apply baseline L2 regularization. The optimal regularization weights for `l2` and `sk`, determined via validated grid-search, are 4.29 and 2.57, respectively.

### Phase 2 Tiny Imagenet training

To do - code demo in progress.


## 3) Citing this work

Please use the following BibTeX entry when citing this paper:

```
@article{Feinman2019,
  title={SK-reg: A smooth kernel regularizer for convolutional neural networks},
  author={Reuben Feinman and Brenden M. Lake},
  journal={arXiv preprint arXiv:TODO},
  year={2019}
}
```
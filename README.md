# SK-regularization (code in progress)

Code for the paper "SK-reg: A smooth kernel regularizer for convolutional neural networks" (Feinman & Lake, 2019).

## 1) Requirements & setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/SK-regularization:$PYTHONPATH"
    
## 2) Running the experiments

### Download the silhouettes dataset

First, download the pre-processed silhouettes image dataset from the following link:
<http://www.cns.nyu.edu/~reuben/files/silhouettes.zip>.

Unzip the folder and place it into the `data/` sub-directory.

### Phase 1 training

Next, you will train the CNN on the Phase 1 20-way classification task. 
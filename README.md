# SK-regularization (code in progress)

Code for the paper "SK-reg: A smooth kernel regularizer for convolutional neural networks" (Feinman & Lake, 2019).

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/SK-regularization:$PYTHONPATH"
    
## Running the Experiments

### Download & pre-process silhouettes dataset

First, download the raw silhouettes image dataset from the following link:
<http://www.cns.nyu.edu/~reuben/files/silhouettes_raw.zip>.
This dataset is a cleaned version of the Brown LEMS binary shape dataset (a minor re-organization of the file names was performed). Unzip the folder and place it into the `data/` sub-directory.

Next, use the following notebook to pre-process the image dataset

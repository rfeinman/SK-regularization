# SK-reg

Code for the paper "SK-reg: A smooth kernel regularizer for convolutional neural networks" (Feinman & Lake, 2019).

## Requirements & Setup
This code repository requires Keras and TensorFlow. Keras must be
configured to use TensorFlow backend. A full list of requirements can be found
in `requirements.txt`. After cloning this repository, it is recommended that
you add the path to the repository to your `PYTHONPATH` environment variable
to enable imports from any folder:

    export PYTHONPATH="/path/to/SK-regularization:$PYTHONPATH"
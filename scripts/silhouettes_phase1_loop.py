from __future__ import division, print_function
import os
import shutil
import argparse

from silhouettes_phase1 import train_phase1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', default=400, type=int)
    parser.add_argument('--fb', default=False, action='store_true')
    parser.add_argument('--data_dir', default='../data/silhouettes/phase1', type=str)
    parser.add_argument('--nb_trials', default=20, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()

    # set kernel directory... this is were CNN kernels will be saved
    if args.fb:
        kernel_dir = '../data/kernel_dataset_fb'
    else:
        kernel_dir = '../data/kernel_dataset'
    if os.path.isdir(kernel_dir):
        shutil.rmtree(kernel_dir)
    os.mkdir(kernel_dir)

    # experiment params
    kwargs = {
        'nb_epochs': args.nb_epochs,
        'fb': args.fb,
        'gpu_id': args.gpu_id
    }

    # train loop
    for trial in range(args.nb_trials):
        kwargs['results_dir'] = os.path.join(kernel_dir, 'trial%0.3i'%trial)
        train_phase1(**kwargs)

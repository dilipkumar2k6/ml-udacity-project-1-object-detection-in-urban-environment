import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import shutil

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    source_dir = os.path.join(data_dir, 'training_and_validation')
    tf_records = os.listdir(source_dir)
    random.shuffle(tf_records)
    num_tf_records = len(tf_records)
    num_train = num_tf_records * .75
    num_validation = num_tf_records * .15
    num_test = num_tf_records * .10

    # iterate all tf records
    i = 0
    for tf_record in tf_records:
        tf_record_path = os.path.join(source_dir, tf_record) 
        # process for train data
        if i < num_train:
            shutil.move(tf_record_path, os.path.join(data_dir, 'train'))
        # process for validation data
        elif i >= num_train and i < num_train + num_validation:
            shutil.move(tf_record_path, os.path.join(data_dir, 'val'))
        # process for test data
        else:
            shutil.move(tf_record_path, os.path.join(data_dir, 'test'))
        i = i + 1

    print(f'Tf records split num_train={num_train}, num_validation={num_validation}, num_test={num_test}')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
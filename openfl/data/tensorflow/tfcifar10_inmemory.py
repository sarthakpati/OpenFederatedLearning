# Copyright (C) 2020 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from openfl.data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory
#from openfl.data import load_cifar10_shard

from math import ceil
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.datasets import cifar10

def _load_raw_datashards(shard_num, nb_collaborators):
    """Load the raw CIFAR10 dataset from the web 

    origin_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
    hash_value = 'c58f30108f718f92721af3b95e74349a' 
    path = get_file('cifar10.tar.gz', origin=origin_link, file_hash=hash_value) 
  
    Args: 
        shard_num (int): The index of the dataset shard 
        nb_collaborators (int): The number of collaborators in the federation
  
    Returns: 
        Two tuples (images, labels) for the training and validation datasets for this shard 
  
    """ 
  
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # fix the label dimension to be (N,) 
    y_train = y_train.reshape(-1) 
    y_test = y_test.reshape(-1) 
  
    # create the shards 
    X_train_shards = x_train[shard_num::nb_collaborators] 
    y_train_shards = y_train[shard_num::nb_collaborators] 
  
    X_test_shards = x_test[shard_num::nb_collaborators] 
    y_test_shards  = y_test[shard_num::nb_collaborators] 
    return (X_train_shards, y_train_shards), (X_test_shards, y_test_shards)

def load_cifar10_shard(shard_num, nb_collaborators, categorical=True, channels_last=False, **kwargs): 
    """Load the CIFAR10 dataset.
  
    Args: 
        shard_num (int): The index of the dataset shard 
        nb_collaborators (int): The number of collaborators in the federation
        categorical (bool): True = return the categorical labels as one-hot encoded (Default = True) 
        channels_last (bool): True = input images are channels first (Default = False) 
        **kwargs: Variable parameters to pass to function 
  
    Returns: 
        list: The input shape. 
        int: The number of classes 
        numpy.ndarray: The training data 
        numpy.ndarray: The training labels 
        numpy.ndarray: The validation data 
        numpy.ndarray: The validation labels 
    """ 
    img_rows, img_cols, img_channel = 32, 32, 3
    num_classes = 10 
  
    (X_train, y_train), (X_test, y_test) = _load_raw_datashards(shard_num, nb_collaborators)
  
    if channels_last: 
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channel) 
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channel) 
        input_shape = (img_rows, img_cols, img_channel) 
    else: 
        X_train = X_train.reshape(X_train.shape[0], img_channel, img_rows, img_cols) 
        X_test = X_test.reshape(X_test.shape[0], img_channel, img_rows, img_cols) 
        input_shape = (img_channel, img_rows, img_cols) 
  
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32') 
    X_train /= 255 
    X_test /= 255
  
    if categorical: 
        # convert class vectors to binary class matrices 
        y_train = keras.utils.to_categorical(y_train, num_classes) 
        y_test = keras.utils.to_categorical(y_test, num_classes) 
  
    return input_shape, num_classes, X_train, y_train, X_test, y_test 


class TensorFlowCIFAR10InMemory(TensorFlowFLDataInMemory):
    """TensorFlow data loader for CIFAR10 dataset
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """Initializer

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            **kwargs: Additional arguments,  passed to super init and load_cifar10_shard

        Returns:
            Data loader with BraTS data
        """

        super().__init__(batch_size, **kwargs)

        _, num_classes, X_train, y_train, X_val, y_val = load_cifar10_shard(shard_num=data_path, **kwargs)

        self.num_classes = num_classes
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

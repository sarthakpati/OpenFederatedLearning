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
from tensorflow.python.keras.utils.data_utils import get_file

from openfl.data import one_hot
from openfl.data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory

def _load_raw_datashards(shard_num, nb_collaborators):                                                                                                                                             
    """Load the raw data by shard                                                                                                                                                                  
                                                                                                                                                                                                   
    Returns tuples of the dataset shard divided into training and validation.                                                                                                                      
                                                                                                                                                                                                   
    Args:                                                                                                                                                                                          
        shard_num (int): The shard number to use                                                                                                                                                   
        nb_collaborators (int): The number of collaborators in the federation                                                                                                                      
                                                                                                                                                                                                   
    Returns:                                                                                                                                                                                       
        2 tuples: (image, label) of the training, validation dataset                                                                                                                               
    """                                                                                                                                                                                            
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'                                                                                                                 
    path = get_file('mnist.npz',                                                                                                                                                                   
                    origin=origin_folder + 'mnist.npz',                                                                                                                                            
                    file_hash='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')                                                                                                  
                                                                                                                                                                                                   
    with np.load(path) as f:                                                                                                                                                                       
        # get all of mnist                                                                                                                                                                         
        X_train_tot = f['x_train']                                                                                                                                                                 
        y_train_tot = f['y_train']                                                                                                                                                                 
                                                                                                                                                                                                   
        X_test_tot = f['x_test']                                                                                                                                                                   
        y_test_tot = f['y_test']                                                                                                                                                                   
                                                                                                                                                                                                   
    # create the shards                                                                                                                                                                            
    X_train = X_train_tot[shard_num::nb_collaborators]                                                                                                                                             
    y_train = y_train_tot[shard_num::nb_collaborators]                                                                                                                                             
                                                                                                                                                                                                   
    X_test = X_test_tot[shard_num::nb_collaborators]                                                                                                                                               
    y_test = y_test_tot[shard_num::nb_collaborators]                                                                                                                                               
                                                                                                                                                                                                   
    return (X_train, y_train), (X_test, y_test)

def load_mnist_shard(shard_num, nb_collaborators, categorical=True, channels_last=True, **kwargs):                                                                                                                                 
    """                                                                                                                                                                                                                            
    Load the MNIST dataset.                                                                                                                                                                                                        
                                                                                                                                                                                                                                   
    Args:                                                                                                                                                                                                                          
        shard_num (int): The shard to use from the dataset                                                                                                                                                                         
        nb_collaborators (int): The number of collaborators in the federation                                                                                                                                                      
        categorical (bool): True = convert the labels to one-hot encoded vectors (Default = True)                                                                                                                                  
        channels_last (bool): True = The input images have the channels last (Default = True)                                                                                                                                      
        **kwargs: Additional parameters to pass to the function                                                                                                                                                                    
                                                                                                                                                                                                                                   
    Returns:                                                                                                                                                                                                                       
        list: The input shape                                                                                                                                                                                                      
        int: The number of classes                                                                                                                                                                                                 
        numpy.ndarray: The training data                                                                                                                                                                                           
        numpy.ndarray: The training labels                                                                                                                                                                                         
        numpy.ndarray: The validation data                                                                                                                                                                                         
        numpy.ndarray: The validation labels                                                                                                                                                                                       
    """                                                                                                                                                                                                                            
    img_rows, img_cols = 28, 28                                                                                                                                                                                                    
    num_classes = 10                                                                                                                                                                                                               
                                                                                                                                                                                                                                   
    (X_train, y_train), (X_test, y_test) = _load_raw_datashards(shard_num, nb_collaborators)                                                                                                                                       
                                                                                                                                                                                                                                   
    if channels_last:                                                                                                                                                                                                              
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)                                                                                                                                                         
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)                                                                                                                                                            
        input_shape = (img_rows, img_cols, 1)                                                                                                                                                                                      
    else:                                                                                                                                                                                                                          
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)                                                                                                                                                         
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)                                                                                                                                                            
        input_shape = (1, img_rows, img_cols)                                                                                                                                                                                      
                                                                                                                                                                                                                                   
    X_train = X_train.astype('float32')                                                                                                                                                                                            
    X_test = X_test.astype('float32')                                                                                                                                                                                              
    X_train /= 255                                                                                                                                                                                                                 
    X_test /= 255                                                                                                                                                                                                                  
    print('X_train shape:', X_train.shape)                                                                                                                                                                                         
    print('y_train shape:', y_train.shape)                                                                                                                                                                                         
    print(X_train.shape[0], 'train samples')                                                                                                                                                                                       
    print(X_test.shape[0], 'test samples')                                                                                                                                                                                         
                                                                                                                                                                                                                                   
    if categorical:                                                                                                                                                                                                                
        # convert class vectors to binary class matrices                                                                                                                                                                           
        y_train = one_hot(y_train, num_classes)                                                                                                                                                                                    
        y_test = one_hot(y_test, num_classes)                                                                                                                                                                                      
                                                                                                                                                                                                                                   
    return input_shape, num_classes, X_train, y_train, X_test, y_test

class TensorFlowMNISTInMemory(TensorFlowFLDataInMemory):
    """TensorFlow Data Loader for MNIST Dataset
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """Initializer

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """

        super().__init__(batch_size, **kwargs)

        _, num_classes, X_train, y_train, X_val, y_val = load_mnist_shard(shard_num=int(data_path), **kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.num_classes = num_classes

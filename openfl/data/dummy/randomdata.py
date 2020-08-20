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

from math import ceil

import numpy as np

from openfl.data import FLData


class RandomData(FLData):
    """A helper function to create a Data object with random data.
    """

    def __init__(self, feature_shape, label_shape=None, data_path=None, train_batches=1, val_batches=1, batch_size=32, n_classes=None, **kwargs):
        """The initializer method

        Args:
            feature_shape: shape of the input array
            label_shape: shape of the label array. If 'None' then label_shape is the same as feature_shape (Default=None)
            data_path: Not used
            train_batches (int): The number of batches to create for the training dataset (Default = 1)
            val_batches (int): The number of batches to create for the validation dataset (Default = 1)
            batch_size (int): the batch size (Default=32)
            **kwargs: Variable parameter list to pass to method

        """

        if label_shape is None:
            label_shape = feature_shape

        self.batch_size = batch_size
        self.X_train = np.random.random(size=tuple([train_batches * batch_size] + list(feature_shape))).astype(np.float32)
        self.y_train = np.random.random(size=tuple([train_batches * batch_size] + list(label_shape))).astype(np.float32)
        self.X_val = np.random.random(size=tuple([val_batches * batch_size] + list(feature_shape))).astype(np.float32)
        self.y_val = np.random.random(size=tuple([val_batches * batch_size] + list(label_shape))).astype(np.float32)
        self.n_classes = n_classes


    def get_feature_shape(self):
        """Get the shape of an example feature array

        Returns:
            tuple: shape of an example feature array
        """
        return self.X_train[0].shape

    def get_train_loader(self):
        """Get training data loader

        Returns:
            loader object
        """
        return self._get_batch_generator(X=self.X_train, y=self.y_train, batch_size=self.batch_size)

    def get_val_loader(self):
        """Get validation data loader

        Returns:
            loader object
        """
        return self._get_batch_generator(X=self.X_val, y=self.y_val, batch_size=self.batch_size)

    def get_training_data_size(self):
        """Get total number of training samples

        Returns:
            int: number of training samples
        """
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        """Get total number of validation samples

        Returns:
            int: number of validation samples
        """
        return self.X_val.shape[0]

    @staticmethod
    def _batch_generator(X, y, idxs, batch_size, num_batches):
        """Generate a batch and then yield

        Args:
            X: input data
            y: label data
            idxs: Indices of dataset
            batch_size (int): Batch size to return
            num_batches (int): Number of batches to return

        Yields:
            tuple: input data, label data
        """
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def _get_batch_generator(self, X, y, batch_size):
        """Returns a batch generator object

        Args:
            X: The input data array
            y: The label data array
            batch_size (int): The batch size

        Returns:
            A batch generator data object
        """
        if batch_size == None:
            batch_size = self.batch_size

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        # build the generator and return it
        return self._batch_generator(X, y, idxs, batch_size, num_batches)

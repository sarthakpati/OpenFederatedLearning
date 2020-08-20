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
import torch
import torch.utils.data

from openfl.data.fldata import FLData


class PyTorchFLDataInMemory(FLData):
    """PyTorch data loader for Federated Learning
    """

    def __init__(self, batch_size, feature_shape=None, **kwargs):
        """Instantiate the data object

        Args:
            batch_size (int): batch size for all data loaders
            feature_shape (tuple): shape of an example feature array
            kwargs: consumes all un-used kwargs


        """
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.inference_loader = None
        self.training_data_size = None
        self.validation_data_size = None
        self.feature_shape = feature_shape

        # Child classes should have init signature:
        # (self, data_path, batch_size, **kwargs), should call this __init__ and then
        # define loaders: self.train_loader and self.val_loader using the
        # self.create_loader provided here.

    def get_feature_shape(self):
        """Get the shape of an example feature array

        Returns:
            tuple: shape of an example feature array
        """
        if self.feature_shape is None:
            # find a loader that isn't None
            for loader in [self.train_loader, self.val_loader, self.inference_loader]:
                if loader is not None:
                    self.feature_shape = tuple(loader.dataset[0][0].shape)
                    break
        return self.feature_shape

    def get_train_loader(self):
        """Get training data loader

        Returns:
            loader object (class defined by inheritor)
        """
        return self.train_loader

    def get_val_loader(self):
        """Get validation data loader

        Returns:
            loader object (class defined by inheritor)
        """
        # TODO: Do we want to be able to modify batch size here?
        # If so will have to decide whether to replace the loader.
        return self.val_loader

    def get_inference_loader(self):
        """
        Get inferencing data loader 

        Returns
        -------
        loader object (class defined by inheritor)
        """
        return self.inference_loader

    def get_training_data_size(self):
        """Get total number of training samples

        Returns:
            int : number of training samples
        """
        return self.training_data_size

    def get_validation_data_size(self):
        """Get total number of validation samples

        Returns:
            int: number of validation samples
        """
        return self.validation_data_size


    def create_loader(self, X, y=None, shuffle=True):
        """Create the data loader using the Torch Tensor methods

        Args:
            X: the input data
            y: the label data
            shuffle: whether to shuffle in-between batch draws

        Returns:
            A `PyTorch DataLoader object <https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html`_
        """
        if isinstance(X[0], np.ndarray):
            tX = torch.stack([torch.Tensor(i) for i in X])
        else:
            tX = torch.Tensor(X)
        if y is None:
            return torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(tX), 
                                               batch_size=self.batch_size, 
                                               shuffle=shuffle)
        else:
            if isinstance(y[0], np.ndarray):
                ty = torch.stack([torch.Tensor(i) for i in y])
            else:
                ty = torch.Tensor(y)
            return torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(tX, ty), 
                                               batch_size=self.batch_size, 
                                               shuffle=shuffle)

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

from openfl.data import FLData


class EmptyData(FLData):
    """A data class with no data
    """

    def __init__(self, data_path, data_size_mean, data_size_std, batch_size=1, p_train=0.8, **kwargs):
        """A data class with no data

        """
        n = np.random.normal(loc=data_size_mean, scale=data_size_std)
        self.batch_size = batch_size
        self.train_size = int(max(1, n * p_train))
        self.val_size = int(max(1, n - self.train_size))

    def get_feature_shape(self):
        """Returns the shape of the input (feature) array
        """
        return 0

    def get_train_loader(self):
        """Returns the training data loader object
        """
        return 0

    def get_val_loader(self):
        """Returns the validation data loader object
        """
        return 0

    def get_training_data_size(self):
        """Return the size of the training data array
        """
        return self.train_size

    def get_validation_data_size(self):
        """Return the size of the validation data array
        """
        return self.val_size

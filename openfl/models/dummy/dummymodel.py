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

"""Dummy model that sleeps and returns random results
"""
import numpy as np
import time

from openfl.models import FLModel


class DummyModel(FLModel):
    """Generic (dummy) model
    """

    def __init__(self, data, layer_shapes, train_time_mean, train_time_std, val_time_mean, val_time_std, **kwargs):
        """Initializer

        Args:
            data: The dataloader object
            layer_shapes:
            train_time_mean:
            train_time_std:
            val_time_mean:
            val_time_std:
            **kwargs: Additional variable to pass to the function
        """

        super().__init__(data=data, **kwargs)

        self.data = data
        self.layer_shapes = layer_shapes
        self.train_time_mean = train_time_mean
        self.train_time_std = train_time_std
        self.val_time_mean = val_time_mean
        self.val_time_std = val_time_std

    def train_batches(self):
        """For this dummy model just randomly sleep for a few seconds
        """
        self.random_sleep(self.train_time_mean, self.train_time_std)
        return np.random.random()

    def validate(self):
        """For this dummy model just randomly sleep for a few seconds
        """
        self.random_sleep(self.val_time_mean, self.val_time_std)
        return np.random.random()

    def get_tensor_dict(self, with_opt_vars):
        """Get the tensor dictionary

        Args:
            with_opt_vars:
        """
        d = {}
        for name, shape in self.layer_shapes.items():
            d[name] = np.random.random(size=tuple(shape)).astype(np.float32)
        return d

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set tensor dictionary
        """
        pass

    def reset_opt_vars(self):
        """Reset optimizer variables
        """
        pass

    def initialize_globals(self):
        """Initial global variables
        """
        pass

    @staticmethod
    def random_sleep(mean, std):
        """Sleep for a random number of seconds
        """
        t = int(np.random.normal(loc=mean, scale=std))
        t = max(1, t)
        time.sleep(t)

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

import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from openfl.models.tensorflow import KerasFLModel

class KerasCNN(KerasFLModel):
    """A basic convolutional neural network model.

    """
    def __init__(self, **kwargs):
        """Initializer

        Args:
            **kwargs: Additional parameters to pass to the function
            
        """
        super().__init__(**kwargs)

        self.model = self.build_model(self.feature_shape, self.data.num_classes, **kwargs)

        self.set_logger()

        print(self.model.summary())
        if self.data is not None:
            print("Training set size: %d; Validation set size: %d" % (self.get_training_data_size(), self.get_validation_data_size()))

    def build_model(self,
                    input_shape,
                    num_classes,
                    conv_kernel_size=(4, 4),
                    conv_strides = (2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        model = Sequential()
        model.add(Conv2D(conv1_channels_out,
                        kernel_size=conv_kernel_size,
                        strides=conv_strides,
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(conv2_channels_out,
                        kernel_size=conv_kernel_size,
                        strides=conv_strides,
                        activation='relu'))
        model.add(Flatten())
        model.add(Dense(final_dense_inputsize, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()
        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model

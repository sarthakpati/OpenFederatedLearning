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

"""Base classes for developing a keras.Model() Federated Learning model.

You may copy this file as the starting point of your own keras model.
"""
import logging
import numpy as np
import tqdm
import tensorflow as tf

from openfl.models import FLModel

import tensorflow.keras as keras
from tensorflow.keras import backend as K

class KerasFLModel(FLModel):
    """The base model for Keras models in the federation.
    """
    def __init__(self, **kwargs):
        """Initializer

        Args:
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.model = keras.Model()

        NUM_PARALLEL_EXEC_UNITS = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                                inter_op_parallelism_threads=1,
                                allow_soft_placement=True,
                                device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
        config.gpu_options.allow_growth=True

        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

    def train_batches(self, num_batches):
        """Train the model on a specified number of batches

        Perform the training for a specified number of batches. Is expected to perform draws randomly, without
        replacement until data is exausted. Then data is replaced and shuffled and draws continue.

        Returns:
            float: loss metric
        """

        # keras model fit method allows for partial batches
        batches_per_epoch = int(np.ceil(self.data.get_training_data_size()/self.data.batch_size))

        if num_batches % batches_per_epoch != 0:
            raise ValueError('KerasFLModel does not support specifying a num_batches corresponding to partial epochs.')
        else:
            num_epochs = num_batches // batches_per_epoch

        history = self.model.fit(self.data.X_train,
                                 self.data.y_train,
                                 batch_size=self.data.batch_size,
                                 epochs=num_epochs,
                                 verbose=0,)

        loss = np.mean([history.history['loss']])
        return loss

    def validate(self):
        """Validate the model on the local dataset

        """
        vals = self.model.evaluate(self.data.X_val, self.data.y_val, verbose=0)
        metrics_names = self.model.metrics_names
        ret_dict = dict(zip(metrics_names, vals))
        return ret_dict['acc']

    @staticmethod
    def _get_weights_dict(obj):
        """Get the dictionary of weights.

        Args:
            obj (Model or Optimizer): The target object that we want to get the weights.

        Returns:
            dict: The weight dictionary.
        """
        weights_dict = {}
        weight_names = [weight.name for weight in obj.weights]
        weight_values = obj.get_weights()
        for name, value in zip(weight_names, weight_values):
            weights_dict[name] = value
        return weights_dict

    @staticmethod
    def _set_weights_dict(obj, weights_dict):
        """Set the object weights with a dictionary.

        The obj can be a model or an optimizer.

        Args:
            obj (Model or Optimizer): The target object that we want to set the weights.
            weights_dict (dict): The weight dictionary.

        Returns:
            None
        """
        weight_names = [weight.name for weight in obj.weights]
        weight_values = [weights_dict[name] for name in weight_names]
        obj.set_weights(weight_values)

    def initialize_globals(self):
        """Initialize global variables
        """
        self.sess.run(tf.global_variables_initializer())

    def get_tensor_dict(self, with_opt_vars):
        """Get the model weights as a tensor dictionary.

        Args:
            with_opt_vars (bool): True = include the optimizer's status.

        Returns:
            dict: The tensor dictionary.
        """
        model_weights = self._get_weights_dict(self.model)

        if with_opt_vars:
            opt_weights = self._get_weights_dict(self.model.optimizer)

            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                self.logger.debug("WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Sets the model weights with a tensor dictionary.

        Args:
            tensor_dict: the tensor dictionary
            with_opt_vars (bool): True = include the optimizer's status.
        """

        if with_opt_vars is False:
            self._set_weights_dict(self.model, tensor_dict)
        else:
            model_weight_names = [weight.name for weight in self.model.weights]
            model_weights_dict = {name: tensor_dict[name] for name in model_weight_names}
            opt_weight_names = [weight.name for weight in self.model.optimizer.weights]
            opt_weights_dict = {name: tensor_dict[name] for name in opt_weight_names}
            self._set_weights_dict(self.model, model_weights_dict)
            self._set_weights_dict(self.model.optimizer, opt_weights_dict)

    def reset_opt_vars(self):
        """Reset optimizer variables

        Resets the optimizer variables

        """
        for weight in self.model.optimizer.weights:
            weight.initializer.run(session=self.sess)

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


"""Mixin class for FL models. No default implementation.

Each framework will likely have its own baseclass implementation (e.g. TensorflowFLModelBase) that uses this mixin.

You may copy use this file or the appropriate framework-specific base-class to port your own models.
"""

import logging


class FLModel(object):
    """Federated Learning Model Base Class
    """

    def __init__(self, data, tensor_dict_split_fn_kwargs=None,**kwargs):
        """Intializer

        Args:
            data: The data object
            tensor_dict_split_fn_kwargs: (Default=None)
            **kwargs: Additional parameters to pass to the function
        """

        self.data = data
        self.feature_shape = self.data.get_feature_shape()

        # key word arguments for determining which parameters to hold out from aggregation.
        # If set to none, an empty dict will be passed, currently resulting in the defaults:
        # holdout_types=['non_float'] # all param np.arrays of this type will be held out
        # holdout_tensor_names=[]     # params with these names will be held out
        # TODO: params are restored from protobufs as float32 numpy arrays, so
        # non-floats arrays and non-arrays are not currently supported for passing to and
        # from protobuf (and as a result for aggregation) - for such params in current examples,
        # aggregation does not make sense anyway, but if this changes support should be added.
        self.tensor_dict_split_fn_kwargs = tensor_dict_split_fn_kwargs


    def set_logger(self):
        """Sets up the log object
        """
        self.logger = logging.getLogger(__name__)

    def get_data(self):
        """Get the data object.

        Serves up batches and provides infor regarding data.

        Returns:
            data object
        """
        return self.data

    def set_data(self, data):
        """Set data object.

        Args:
            data: Data object to set
        Returns:
            None
        """
        if data.get_feature_shape() != self.data.get_feature_shape():
            raise ValueError('Data feature shape is not compatible with model.')
        self.data = data

    def get_training_data_size(self):
        """Get the number of training examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of training examples.
        """
        return self.data.get_training_data_size()

    def get_validation_data_size(self):
        """Get the number of examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of validation examples.
        """
        return self.data.get_validation_data_size()

    def train_batches(self, num_batches, use_tqdm=False):
        """Perform the training for a specified number of batches.

        Is expected to perform draws randomly, without
        replacement until data is exausted. Then data is replaced and
        shuffled and draws continue.

        Args:
            num_batches: Number of batches to train
            use_tdqm (bool): True = use tqdm progress bar (Default=False)

        Returns:
            dict: {<metric>: <value>}
        """
        raise NotImplementedError

    def validate(self):
        """Run validation.

        Returns"
            dict: {<metric>: <value>}
        """
        raise NotImplementedError

    def get_tensor_dict(self, with_opt_vars):
        """Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}
        """
        raise NotImplementedError

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the model weights with a tensor dictionary: {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables of the optimizer.

        Returns:
            None
        """
        raise NotImplementedError

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        raise NotImplementedError

    def initialize_globals(self):
        """Initialize all global variables

        Returns:
            None
        """
        raise NotImplementedError

    def load_native(self, filepath, **kwargs):
        """Loads model state from a filepath in ML-framework "native" format, e.g. PyTorch pickled models. May load from multiple files. Other filepaths may be derived from the passed filepath, or they may be in the kwargs.

        Args:
            filepath (string): Path to frame-work specific file to load. For frameworks that use multiple files, this string must be used to derive the other filepaths.
            kwargs           : For future-proofing 

        Returns:
            None
        """
        raise NotImplementedError

    def save_native(self, filepath, **kwargs):
        """Saves model state in ML-framework "native" format, e.g. PyTorch pickled models. May save one file or multiple files, depending on the framework.

        Args:
            filepath (string): If framework stores a single file, this should be a single file path. Frameworks that store multiple files may need to derive the other paths from this path.
            kwargs           : For future-proofing

        Returns:
            None
        """
        raise NotImplementedError

    def run_inference_and_store_results(self, **kwargs):
        """Runs inference over the inference_loader in the data object, then calls the data object to store the results.
        Args:
            kwargs: For write_outputs method of self.data
        
        Returns:
            List of outputs from the data.write_output calls
        """
        
        # what comes out of the model object? generally numpy arrays
        # what metadata is specific to the batch? for example, simpleITK image objects of input and filenames

        # loop over inference data
        ret = []
        for sample in self.data.get_inference_loader():
            # FIXME: Need something cleaner, and metadata should be consumed as kwargs
            if isinstance(sample, dict) and "features" in sample and "metadata" in sample:
                features = sample["features"]
                metadata = sample["metadata"]
            else:
                # FIXME: this abstraction needs love. The inference loader should take care of all of this.
                if isinstance(sample, list):
                    features = sample[0]
                else:
                    features = sample
                metadata = None
            outputs = self.infer_volume(features)
            ret.append(self.data.write_outputs(outputs, metadata, **kwargs))
        return ret
    
    def infer_volume(self, X):
        """Runs inference on a batch and returns the output of the model.

        Args:
            X: Input for batch

        Returns:
            Model output for batch
        """
        raise NotImplementedError

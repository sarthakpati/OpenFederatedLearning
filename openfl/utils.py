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

import yaml
import importlib
import hashlib
import numpy as np

def load_yaml(path):
    """Load a YAML file

    Args:
        path (string): The directory path for the federation plan (FL Plan)

    Returns:
        A YAML object of the federation plan.

    """
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan

def get_object(module_name, class_name, **kwargs):
    """Import a class from a Python module

    Args:
        module_name: The name of the Python module
        class_name: The class name within the Python module
        **kwargs: Additional parameters to pass to the function

    Returns:
        A Python class module

    """
    module = importlib.import_module(module_name)
    return module.__getattribute__(class_name)(**kwargs)

def split_tensor_dict_into_floats_and_non_floats(tensor_dict):
    """Splits the tensor dictionary into float and non-floating point values

    Splits a tensor dictionary into float and non-float values.

    Args:
        tensor_dict: A dictionary of tensors

    Returns:
        Two dictionaries: the first contains all of the floating point tensors and the second contains all of the non-floating point tensors

    """

    float_dict = {}
    non_float_dict = {}
    for k, v in tensor_dict.items():
        if np.issubdtype(v.dtype, np.floating):
            float_dict[k] = v
        else:
            non_float_dict[k] = v
    return float_dict, non_float_dict


def split_tensor_dict_for_holdouts(logger, tensor_dict, holdout_types=['non_float'], holdout_tensor_names=[]):
    """Splits a tensor according to tensor types.

    Args:
        logger: The log object
        tensor_dict: A dictionary of tensors
        holdout_types: A list of types to extract from the dictionary of tensors
        holdout_tensor_names: A list of tensor names to extract from the dictionary of tensors

    Returns:
        Two dictionaries: the first is the original tensor dictionary minus the holdout tenors and the second is a tensor dictionary with only the holdout tensors

    """
    # initialization
    tensors_to_send = tensor_dict.copy()
    holdout_tensors = {}

    # filter by-name tensors from tensors_to_send and add to holdout_tensors
    # (for ones not already held out becuase of their type)
    for tensor_name in holdout_tensor_names:
        if tensor_name not in holdout_tensors.keys():
            try:
                holdout_tensors[tensor_name] = tensors_to_send.pop(tensor_name)
            except KeyError:
                logger.warn('tried to remove tensor: {} not present in the tensor dict'.format(tensor_name))
                continue

    # filter holdout_types from tensors_to_send and add to holdout_tensors
    for holdout_type in holdout_types:
        if holdout_type == 'non_float':
            # filter non floats from tensors_to_send and add to holdouts
            tensors_to_send, non_float_dict = split_tensor_dict_into_floats_and_non_floats(tensors_to_send)
            holdout_tensors = {**holdout_tensors, **non_float_dict}
        else:
            raise ValueError('{} is not a currently suported parameter type to hold out from a tensor dict'.format(holdout_type))


    return tensors_to_send, holdout_tensors

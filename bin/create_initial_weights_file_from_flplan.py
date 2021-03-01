#!/usr/bin/env python3

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

import argparse
import os
import sys
import logging
import importlib

from openfl import load_yaml, get_object, split_tensor_dict_for_holdouts, hash_string
from openfl.collaborator.collaborator import OptTreatment
from openfl.flplan import parse_fl_plan, create_data_object, create_model_object, create_compression_pipeline
from openfl.proto.protoutils import dump_proto, numpy_array_to_tensor_proto
from openfl.proto.collaborator_aggregator_interface_pb2 import ModelHeader, ExtraModelInfo
from setup_logging import setup_logging


def main(plan, native_model_weights_filepath, collaborators_file, feature_shape, n_classes, data_config_fname, logging_config_path, logging_default_level, model_device):
    """Creates a protobuf file of the initial weights for the model

    Uses the federation (FL) plan to create an initial weights file
    for the federation.

    Args:
        plan: The federation (FL) plan filename
        native_model_weights_filepath: A framework-specific filepath. Path will be relative to the working directory.
        collaborators_file:
        feature_shape: The input shape to the model
        data_config_fname: The data configuration file (defines where the datasets are located)
        logging_config_path: The log path
        logging_default_level (int): The default log level

    """

    setup_logging(path=logging_config_path, default_level=logging_default_level)

    logger = logging.getLogger(__name__)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    # ensure the weights dir exists
    if not os.path.exists(weights_dir):
        print('creating folder:', weights_dir)
        os.makedirs(weights_dir)

    # parse the plan and local config
    flplan = parse_fl_plan(os.path.join(plan_dir, plan))
    local_config = load_yaml(os.path.join(base_dir, data_config_fname))

    # get the output directory
    directory = os.path.join(weights_dir, flplan['aggregator_object_init']['init_kwargs']['model_directory'], 'initial')

    # create the data object for models whose architecture depends on the feature shape
    if feature_shape is None:
        if collaborators_file is None:
            sys.exit("You must specify either a feature shape or a collaborator list in order for the script to determine the input layer shape")
        # FIXME: this will ultimately run in a governor environment and should not require any data to work
        # pick the first collaborator to create the data and model (could be any)
        collaborator_common_name = load_yaml(os.path.join(base_dir, 'collaborator_lists', collaborators_file))['collaborator_common_names'][0]
        data = create_data_object(flplan, collaborator_common_name, local_config, n_classes=n_classes)
    else:
        data = get_object('openfl.data.dummy.randomdata', 'RandomData', feature_shape=feature_shape)
        logger.info('Using data object of type {} and feature shape {}'.format(type(data), feature_shape))

    # create the model object
    wrapped_model = create_model_object(flplan, data, model_device=model_device)

    # determine if we need to store the optimizer variables
    # FIXME: what if this key is missing?
    try:
        opt_treatment = OptTreatment[flplan['collaborator_object_init']['init_kwargs']['opt_treatment']]
    except KeyError:
        # FIXME: this error message should use the exception to determine the missing key and the Enum to display the options dynamically
        sys.exit("FL plan must specify ['collaborator_object_init']['init_kwargs']['opt_treatment'] as [RESET|CONTINUE_LOCAL|CONTINUE_GLOBAL]")

    # FIXME: this should be an "opt_treatment requires parameters type check rather than a magic string"
    with_opt_vars = opt_treatment == OptTreatment['CONTINUE_GLOBAL']

    if native_model_weights_filepath is not None:
        wrapped_model.load_native(native_model_weights_filepath)
    
    tensor_dict_split_fn_kwargs = wrapped_model.tensor_dict_split_fn_kwargs or {}

    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(logger,
                                                                 wrapped_model.get_tensor_dict(with_opt_vars=with_opt_vars),
                                                                 **tensor_dict_split_fn_kwargs)
    logger.warn('Following paramters omitted from global initial model, '\
                'local initialization will determine values: {}'.format(list(holdout_params.keys())))

    os.makedirs(directory, exist_ok=True)

    model_header = ModelHeader(id=wrapped_model.__class__.__name__, version=0)
    dump_proto(model_header, os.path.join(directory, 'ModelHeader.pbuf'))

    extra_model_info = ExtraModelInfo(tensor_names=list(tensor_dict.keys()))
    dump_proto(extra_model_info, os.path.join(directory, 'ExtraModelInfo.pbuf'))

    for k, v in tensor_dict.items():
        t_hash = hash_string(k)
        proto = numpy_array_to_tensor_proto(v, k)
        dump_proto(proto, os.path.join(directory, '{}.pbuf'.format(t_hash)))

    logger.info("Created initial weights files in directory: {}".format(directory))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--native_model_weights_filepath', '-nmwf', type=str, default=None)
    parser.add_argument('--collaborators_file', '-c', type=str, default=None, help="Name of YAML File in /bin/federations/collaborator_lists/")
    parser.add_argument('--feature_shape', '-fs', type=int, nargs='+', default=None)
    parser.add_argument('--n_classes', '-nc', type=int, default=None)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    # FIXME: this kind of commandline configuration needs to be done in a consistent way
    parser.add_argument('--model_device', '-md', type=str, default='cpu')
    args = parser.parse_args()
    main(**vars(args))

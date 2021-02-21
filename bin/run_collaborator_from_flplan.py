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
import sys
import os
import logging
import importlib

from openfl.collaborator.collaborator import Collaborator
from openfl.flplan import create_collaborator_object_from_flplan, parse_fl_plan, load_yaml
from setup_logging import setup_logging


def main(plan, 
         collaborator_common_name, 
         single_col_cert_common_name, 
         data_config_fname, 
         data_dir,
         validate_on_patches,
         data_in_memory, 
         data_queue_max_length, 
         data_queue_num_workers,  
         logging_config_path, 
         logging_default_level, 
         logging_directory, 
         model_device):
    """Runs the collaborator client process from the federation (FL) plan

    Args:
        plan                        : The filename for the federation (FL) plan YAML file
        collaborator_common_name    : The common name for the collaborator node
        single_col_cert_common_name : The SSL certificate for this collaborator
        data_config_fname           : The dataset configuration filename (YAML)
        data_dir                    : parent directory holding the patient data subdirectories(to be split into train and val)
        validate_on_patches         : model init kwarg
        data_in_memory              : data init kwarg 
        data_queue_max_length       : data init kwarg 
        data_queue_num_workers      : data init kwarg
        logging_config_fname        : The log file
        logging_default_level       : The log level
        model_device                : gets passed to model 'init' function as "device"
    """
    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')
    metadata_dir = os.path.join(base_dir, 'metadata')
    logging_config_path = os.path.join(script_dir, logging_config_path)
    logging_directory = os.path.join(script_dir, logging_directory)

    setup_logging(path=logging_config_path, default_level=logging_default_level, logging_directory=logging_directory)
    
    flplan = parse_fl_plan(os.path.join(plan_dir, plan))

    # FIXME: Find a better solution for passing model and data init kwargs
    model_init_kwarg_keys = ['validate_on_patches', 'torch_threads', 'kmp_affinity']
    model_init_kwarg_vals = [validate_on_patches, torch_threads, kmp_affinity]
    for key, value in zip(model_init_kwarg_keys, model_init_kwarg_vals):
        if value is not None:
            flplan['model_object_init']['init_kwargs'][key] = value

    data_init_kwarg_keys = ['data_in_memory', 'data_queue_max_length', 'data_queue_num_workers']
    data_init_kwarg_vals = [data_in_memory,data_queue_max_length, data_queue_num_workers]
    for key, value in zip(data_init_kwarg_keys, data_init_kwarg_vals):
        if value is not None:
            flplan['data_object_init']['init_kwargs'][key] = value

    local_config = load_yaml(os.path.join(base_dir, data_config_fname))

    try:
        collaborator = create_collaborator_object_from_flplan(flplan,
                                                            collaborator_common_name,
                                                            local_config,
                                                            base_dir,
                                                            weights_dir,
                                                            metadata_dir,
                                                            single_col_cert_common_name,
                                                            data_dir=data_dir,
                                                            model_device=model_device)

        collaborator.run()
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).exception(repr(e))
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborator_common_name', '-col', type=str, required=True)
    parser.add_argument('--single_col_cert_common_name', '-scn', type=str, default=None)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    # FIXME: data_dir should be data_path
    parser.add_argument('--data_dir', '-d', type=str, default=None)
    # FIXME: a more general solution of passing model and data kwargs should be provided
    parser.add_argument('--validate_on_patches', '-vp', type=bool, default=None)
    parser.add_argument('--data_in_memory', '-dim', type=bool, default=None)
    parser.add_argument('--data_queue_max_length', '-dqml', type=int, default=None)
    parser.add_argument('--data_queue_num_workers', '-dqnw', type=int, default=None)
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    parser.add_argument('--logging_directory', '-ld', type=str, default="logs")
    # FIXME: this kind of commandline configuration needs to be done in a consistent way
    parser.add_argument('--model_device', '-md', type=str, default='cpu')
    args = parser.parse_args()
    main(**vars(args))

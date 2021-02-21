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
import logging

from openfl            import load_yaml
from openfl.flplan     import parse_fl_plan
from single_proc_fed    import federate
from setup_logging      import setup_logging


def main(plan, 
         resume, 
         collaborators_file, 
         data_config_fname, 
         validate_without_patches_flag,
         data_in_memory_flag, 
         data_queue_max_length, 
         data_queue_num_workers,
         torch_threads, 
         kmp_affinity_flag,
         logging_config_path, 
         logging_default_level, 
         logging_directory, 
         model_device, 
         **kwargs):
    """Run the federation simulation from the federation (FL) plan.

    Runs a federated training from the federation (FL) plan but creates the
    aggregator and collaborators on the same compute node. This allows
    the developer to test the model and data loaders before running
    on the remote collaborator nodes.

    Args:
        plan                    : The Federation (FL) plan (YAML file)
        resume                  : Whether or not the aggregator is told to resume from previous best
        collaborators_file      : The file listing the collaborators
        data_config_fname       : The file describing where the dataset is located on the collaborators
        validate_on_patches     : model init kwarg
        data_in_memory          : data init kwarg 
        data_queue_max_length   : data init kwarg 
        data_queue_num_workers  : data init kwarg
        torch_threads           : number of threads to set in torch 
        kmp_affinity            : whether or not to include a hard-coded KMP AFFINITY setting
        logging_config_path     : The log file
        logging_default_level   : The log level
        **kwargs                : Variable parameters to pass to the function

    """
    # FIXME: consistent filesystem (#15)
    # establish location for fl plan as well as
    # where to get and write model protobufs
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')
    metadata_dir = os.path.join(base_dir, 'metadata')
    collaborators_dir = os.path.join(base_dir, 'collaborator_lists')
    logging_config_path = os.path.join(script_dir, logging_config_path)
    logging_directory = os.path.join(script_dir, logging_directory)

    setup_logging(path=logging_config_path, default_level=logging_default_level, logging_directory=logging_directory)

    # load the flplan, local_config and collaborators file
    flplan = parse_fl_plan(os.path.join(plan_dir, plan))

    # FIXME: Find a better solution for passing model and data init kwargs
    model_init_kwarg_keys = ['validate_without_patches', 'torch_threads', 'kmp_affinity']
    model_init_kwarg_vals = [validate_without_patches_flag, torch_threads, kmp_affinity_flag]
    for key, value in zip(model_init_kwarg_keys, model_init_kwarg_vals):
        if value is not None:
            flplan['model_object_init']['init_kwargs'][key] = value

    data_init_kwarg_keys = ['in_memory', 'q_max_length', 'q_num_workers']
    data_init_kwarg_vals = [data_in_memory_flag,data_queue_max_length, data_queue_num_workers]
    for key, value in zip(data_init_kwarg_keys, data_init_kwarg_vals):
        if value is not None:
            flplan['data_object_init']['init_kwargs'][key] = value
    
    local_config = load_yaml(os.path.join(base_dir, data_config_fname))
    collaborator_common_names = load_yaml(os.path.join(collaborators_dir, collaborators_file))['collaborator_common_names']
  
    # TODO: Run a loop here over various parameter values and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(flplan=flplan,
             resume=resume,
             local_config=local_config,
             collaborator_common_names=collaborator_common_names,
             base_dir=base_dir,
             weights_dir=weights_dir,
             metadata_dir=metadata_dir,
             model_device=model_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--resume', '-r', type=bool, default=False)
    parser.add_argument('--collaborators_file', '-c', type=str, required=True, help="Name of YAML File in /bin/federations/collaborator_lists/")
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    # FIXME: a more general solution of passing model and data kwargs should be provided
    parser.add_argument('--validate_without_patches_flag', '-vwop', type=bool, default=None)
    parser.add_argument('--data_in_memory_flag', '-dim', type=bool, default=None)
    parser.add_argument('--data_queue_max_length', '-dqml', type=int, default=None)
    parser.add_argument('--data_queue_num_workers', '-dqnw', type=int, default=None)
    parser.add_argument('--torch_threads', '-tt', type=int, default=None)
    parser.add_argument('--kmp_affinity_flag', '-ka', type=bool, default=None)
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    parser.add_argument('--logging_directory', '-ld', type=str, default="logs")
    # FIXME: this kind of commandline configuration needs to be done in a consistent way
    parser.add_argument('--model_device', '-md', type=str, default='cpu')
    args = parser.parse_args()
    main(**vars(args))

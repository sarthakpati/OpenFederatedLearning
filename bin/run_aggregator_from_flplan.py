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

from openfl.flplan import parse_fl_plan, load_yaml, create_aggregator_object_from_flplan, create_aggregator_server_from_flplan, get_serve_kwargs_from_flpan
from setup_logging import setup_logging


def main(plan, collaborators_file, single_col_cert_common_name, logging_config_path, logging_default_level, logging_directory):
    """Runs the aggregator service from the Federation (FL) plan

    Args:
        plan: The Federation (FL) plan
        collaborators_file: The file listing the collaborators
        single_col_cert_common_name: The SSL certificate
        logging_config_path: The log configuration file
        logging_default_level: The log level

    """
    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')
    collaborators_dir = os.path.join(base_dir, 'collaborator_lists')
    metadata_dir = os.path.join(base_dir, 'metadata')
    logging_config_path = os.path.join(script_dir, logging_config_path)
    logging_directory = os.path.join(script_dir, logging_directory)

    setup_logging(path=logging_config_path, default_level=logging_default_level, logging_directory=logging_directory)

    flplan = parse_fl_plan(os.path.join(plan_dir, plan))
    collaborator_common_names = load_yaml(os.path.join(collaborators_dir, collaborators_file))['collaborator_common_names']

    agg             = create_aggregator_object_from_flplan(flplan, collaborator_common_names, single_col_cert_common_name, weights_dir, metadata_dir)
    server          = create_aggregator_server_from_flplan(agg, flplan)
    serve_kwargs    = get_serve_kwargs_from_flpan(flplan, base_dir)
    
    server.serve(**serve_kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborators_file', '-c', type=str, required=True, help="Name of YAML File in /bin/federations/collaborator_lists/")
    parser.add_argument('--single_col_cert_common_name', '-scn', type=str, default=None)
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    parser.add_argument('--logging_directory', '-ld', type=str, default="logs")
    args = parser.parse_args()
    main(**vars(args))

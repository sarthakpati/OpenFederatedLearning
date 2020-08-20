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

import os
import logging
import hashlib
import yaml

from . import load_yaml, get_object

def get_data_path_from_local_config(local_config, collaborator_common_name, data_name_in_local_config):
    data_names_to_paths = local_config['collaborators']

    if collaborator_common_name not in data_names_to_paths:
        raise ValueError("Could not find collaborator id \"{}\" in the local data config file.".format(collaborator_common_name))
    
    data_names_to_paths = data_names_to_paths[collaborator_common_name]
    if data_name_in_local_config not in data_names_to_paths:
        raise ValueError("Could not find data path for collaborator id \"{}\" and dataset name \"{}\".".format(collaborator_common_name, data_name_in_local_config))

    return data_names_to_paths[data_name_in_local_config]

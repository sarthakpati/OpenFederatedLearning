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

from openfl import load_yaml, get_object
from openfl.localconfig import get_data_path_from_local_config
from openfl.tensor_transformation_pipelines import NoCompressionPipeline
from openfl.aggregator.aggregator import Aggregator
from openfl.collaborator.collaborator import Collaborator
from openfl.comms.grpc.aggregatorgrpcserver import AggregatorGRPCServer
from openfl.comms.grpc.collaboratorgrpcclient import CollaboratorGRPCClient


def parse_fl_plan(plan_path, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    flplan = load_yaml(plan_path)

    # ensure 'init_kwargs' appears in each top-level block
    for k in flplan.keys():
        if 'init_kwargs' not in flplan[k]:
            flplan[k]['init_kwargs'] = {}

    # collect all the plan filepaths used
    plan_files = [plan_path]

    # walk the top level keys for defaults_file in sorted order
    for k in sorted(flplan.keys()):
        defaults_file = flplan[k].get('defaults_file')
        if defaults_file is not None:
            defaults_file = os.path.join(os.path.dirname(plan_path), defaults_file)
            logger.info("Using FLPlan defaults for section '{}' from file '{}'".format(k, defaults_file))
            defaults = load_yaml(defaults_file)
            if 'init_kwargs' in defaults:
                defaults['init_kwargs'].update(flplan[k]['init_kwargs'])
                flplan[k]['init_kwargs'] = defaults['init_kwargs']
            defaults.update(flplan[k])
            flplan[k] = defaults
            plan_files.append(defaults_file)

    # create the hash of these files
    flplan_fname = os.path.splitext(os.path.basename(plan_path))[0]
    flplan_hash = hash_files(plan_files, logger=logger)

    federation_uuid = '{}_{}'.format(flplan_fname, flplan_hash[:8])
    aggregator_uuid = 'aggregator_{}'.format(federation_uuid)

    flplan['aggregator_object_init']['init_kwargs']['aggregator_uuid'] = aggregator_uuid
    flplan['aggregator_object_init']['init_kwargs']['federation_uuid'] = federation_uuid
    flplan['collaborator_object_init']['init_kwargs']['aggregator_uuid'] = aggregator_uuid
    flplan['collaborator_object_init']['init_kwargs']['federation_uuid'] = federation_uuid
    flplan['hash'] = flplan_hash

    logger.info("Parsed plan:\n{}".format(yaml.dump(flplan)))

    return flplan


def init_object(flplan_block, **kwargs):
    if 'class_to_init' not in flplan_block:
        raise ValueError("FLPLAN ERROR")

    init_kwargs = flplan_block.get('init_kwargs', {})
    init_kwargs.update(**kwargs)

    class_to_init   = flplan_block['class_to_init']
    class_name      = class_to_init.split('.')[-1]
    module_name     = '.'.join(class_to_init.split('.')[:-1])

    return get_object(module_name, class_name, **init_kwargs)


def create_compression_pipeline(flplan):
    if flplan.get('compression_pipeline_object_init') is not None:
        compression_pipeline = init_object(flplan.get('compression_pipeline_object_init'))
    else:
        compression_pipeline = NoCompressionPipeline()
    return compression_pipeline


def create_model_object(flplan, data_object, model_device='cpu'):
    return init_object(flplan['model_object_init'], data=data_object, device=model_device)


def resolve_autoport(flplan):
    config = flplan['network_object_init']
    flplan_hash_8 = flplan['hash'][:8]
 
    # check for auto_port convenience settings
    if config.get('auto_port', False) == True:
        # replace the port number with something in the range of min-max
        # default is 49152 to 60999
        port_range = config.get('auto_port_range', (49152, 60999))
        port = (int(flplan_hash_8, 16) % (port_range[1] - port_range[0])) + port_range[0]
        config['init_kwargs']['agg_port'] = port


def create_aggregator_server_from_flplan(agg, flplan):
    # FIXME: this is currently only the GRPC server which takes no init kwargs!
    return AggregatorGRPCServer(agg)


def get_serve_kwargs_from_flpan(flplan, base_dir):
    config = flplan['network_object_init']

    resolve_autoport(flplan)

    # find the cert to use
    cert_dir         = os.path.join(base_dir, config.get('cert_folder', 'pki')) # default to 'pki
    cert_common_name = config['init_kwargs']['agg_addr']

    cert_chain_path = os.path.join(cert_dir, 'cert_chain.crt')
    certificate     = os.path.join(cert_dir, 'server', '{}.crt'.format(cert_common_name))
    private_key     = os.path.join(cert_dir, 'server', '{}.key'.format(cert_common_name))

    cert_common_name = config['init_kwargs']['agg_addr']

    serve_kwargs = config['init_kwargs']

    # patch in kwargs for certificates
    serve_kwargs['ca']          = cert_chain_path
    serve_kwargs['certificate'] = certificate
    serve_kwargs['private_key'] = private_key
    
    return serve_kwargs


def create_aggregator_object_from_flplan(flplan, collaborator_common_names, single_col_cert_common_name, weights_dir, metadata_dir):
    init_kwargs = flplan['aggregator_object_init']['init_kwargs']

    # FIXME: this sort of hackery should be handled by a filesystem abstraction
    # patch in the collaborators file and single_col_cert_common_name
    init_kwargs['collaborator_common_names']    = collaborator_common_names
    init_kwargs['single_col_cert_common_name']  = single_col_cert_common_name

    # FIXME: this sort of hackery should be handled by a filesystem abstraction
    # path in the full model filepaths
    for p in ['init', 'latest', 'best']:
        init_kwargs['{}_model_fpath'.format(p)] = os.path.join(weights_dir, init_kwargs['{}_model_fname'.format(p)])

    # FIXME: this sort of hackery should be handled by a filesystem abstraction
    # patch in full metadata filepaths
    for p in ['init', 'latest']:
        k = '{}_metadata_fname'.format(p)
        if k in init_kwargs and init_kwargs[k] is not None: 
            init_kwargs[k] = os.path.join(metadata_dir, init_kwargs[k])

    compression_pipeline = create_compression_pipeline(flplan)

    return Aggregator(compression_pipeline=compression_pipeline,
                      **init_kwargs)


def create_collaborator_network_object(flplan, collaborator_common_name, single_col_cert_common_name, base_dir):
    config = flplan['network_object_init']

    resolve_autoport(flplan)

    # find the cert to use
    cert_dir = os.path.join(base_dir, config.get('cert_folder', 'pki')) # default to 'pki

    # if a single cert common name is in use, then that is the certificate we must use
    if single_col_cert_common_name is None:
        cert_common_name = collaborator_common_name
    else:
        cert_common_name = single_col_cert_common_name

    cert_chain_path = os.path.join(cert_dir, 'cert_chain.crt')
    certificate     = os.path.join(cert_dir, 'client', '{}.crt'.format(cert_common_name))
    private_key     = os.path.join(cert_dir, 'client', '{}.key'.format(cert_common_name))

    # FIXME: support network objects other than GRPC
    return CollaboratorGRPCClient(ca=cert_chain_path,
                                  certificate=certificate,
                                  private_key=private_key,
                                  **config['init_kwargs'])


# FIXME: data_dir should be data_path
def create_collaborator_object_from_flplan(flplan, 
                                           collaborator_common_name, 
                                           local_config,
                                           base_dir,
                                           weights_dir,
                                           metadata_dir,
                                           single_col_cert_common_name=None,
                                           data_dir=None,
                                           data_object=None,
                                           model_object=None,
                                           compression_pipeline=None,
                                           network_object=None,
                                           model_device=None):
    if data_object is None:
        if data_dir is None:
            data_object = create_data_object(flplan, collaborator_common_name, local_config)
        else:
            data_object = create_data_object_with_explicit_data_path(flplan, data_path=data_dir)

    if model_object is None:
        model_object = create_model_object(flplan, data_object, model_device=model_device)
    
    if compression_pipeline is None:
        compression_pipeline = create_compression_pipeline(flplan)

    if network_object is None:
        network_object = create_collaborator_network_object(flplan, collaborator_common_name, single_col_cert_common_name, base_dir)

    # FIXME: filesystem "workspace" can fix this
    init_kwargs = {}
    # patch in weights dir for native model filepath and metadata_dir for saving metadata
    if 'init_kwargs' in flplan['collaborator_object_init']:
        init_kwargs = flplan['collaborator_object_init']['init_kwargs']
        if 'save_best_native_path' in init_kwargs:
            init_kwargs['save_best_native_path'] = os.path.join(weights_dir, init_kwargs['save_best_native_path'])
        if 'save_metadata_path' in init_kwargs:
            init_kwargs['save_metadata_path'] = os.path.join(metadata_dir, init_kwargs['save_metadata_path'])

    return Collaborator(collaborator_common_name=collaborator_common_name,
                        wrapped_model=model_object, 
                        channel=network_object,
                        compression_pipeline=compression_pipeline,
                        single_col_cert_common_name=single_col_cert_common_name,  
                        **flplan['collaborator_object_init']['init_kwargs'])


# FIXME: Should these two functions be one with different behavior based on parameters?
def create_data_object(flplan, collaborator_common_name, local_config, **kwargs):
    data_path = get_data_path_from_local_config(local_config,
                                                collaborator_common_name,
                                                flplan['data_object_init']['data_name_in_local_config'])
    return create_data_object_with_explicit_data_path(flplan, data_path=data_path, **kwargs)


def create_data_object_with_explicit_data_path(flplan, data_path, **kwargs):
    return init_object(flplan['data_object_init'], data_path=data_path, **kwargs)


def hash_files(paths, logger=None):
    md5 = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as f:
            md5.update(f.read())
        if logger is not None:
            logger.info("After hashing {}, hash is {}".format(p, md5.hexdigest()))
    return md5.hexdigest()

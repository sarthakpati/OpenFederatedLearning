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

from openfl import hash_string
from openfl.proto import load_proto, dump_proto
from openfl.proto.collaborator_aggregator_interface_pb2 import LegacyModelProto, ModelHeader, ExtraModelInfo, TensorProto

def main(input_pbuf_file, output_directory):
    # load the input pbuf file
    model = load_proto(input_pbuf_file, LegacyModelProto)
    
    os.makedirs(output_directory, exist_ok=True)

    model_header = ModelHeader(id=model.header.id, version=model.header.version)
    tensors = []
    tensor_names = []
    for t in model.tensors:
        name = t.name
        data_bytes = t.data_bytes
        shape = list(t.transformer_metadata[0].int_list)
        tensors.append(TensorProto(name=name, data_bytes=data_bytes, shape=shape))
        tensor_names.append(name)
    extra_model_info = ExtraModelInfo(tensor_names=tensor_names)
    
    dump_proto(model_header, os.path.join(output_directory, 'ModelHeader.pbuf'))
    dump_proto(extra_model_info, os.path.join(output_directory, 'ExtraModelInfo.pbuf'))

    for t_proto in tensors:
        t_hash = hash_string(t_proto.name)
        dump_proto(t_proto, os.path.join(output_directory, '{}.pbuf'.format(t_hash)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pbuf_file', '-i', type=str, required=True)
    parser.add_argument('--output_directory', '-o', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))

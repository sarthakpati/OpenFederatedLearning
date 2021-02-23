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

from setuptools import setup

setup(name='openfl',
      version='0.0.1',
      packages=['openfl',
                'openfl.aggregator',
                'openfl.collaborator',
                'openfl.tensor_transformation_pipelines',
                'openfl.proto',
                'openfl.comms.grpc',
                'openfl.models',
                'openfl.models.dummy', 
                'openfl.data', 
                'openfl.data.dummy',
                ],
      exclude =['openfl.models.pytorch', 
                'openfl.models.pytorch.pt_2dunet', 
                'openfl.models.pytorch.pt_cnn',
                'openfl.models.pytorch.pt_resnet',
                'openfl.data.pytorch',
                'openfl.models.tensorflow', 
                'openfl.models.tensorflow.keras_cnn', 
                'openfl.models.tensorflow.keras_resnet', 
                'openfl.models.tensorflow.tf_2dunet',
                'openfl.data.tensorflow',
                ],
      install_requires=['protobuf', 'pyyaml', 'grpcio==1.30.0', 'tqdm', 'coloredlogs', 'numpy']
)

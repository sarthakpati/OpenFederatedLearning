.. # Copyright (C) 2020 Intel Corporation
.. # Licensed under the Apache License, Version 2.0 (the "License");
.. # you may not use this file except in compliance with the License.
.. # You may obtain a copy of the License at
.. #
.. #     http://www.apache.org/licenses/LICENSE-2.0
.. #
.. # Unless required by applicable law or agreed to in writing, software
.. # distributed under the License is distributed on an "AS IS" BASIS,
.. # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. # See the License for the specific language governing permissions and
.. # limitations under the License.

*****************
Design Philosophy
*****************

The overall design is that all of the scripts are built off of the
federation plan. The plan is just a `YAML <https://en.wikipedia.org/wiki/YAML>`_
file that defines the
collaborators, aggregator, connections, models, data,
and any other parameters that describes how the training will evolve.
In the “Hello Federation” demos, the plan will be located in the
YAML file: *bin/federations/plans/keras_cnn_mnist_2.yaml*.
As you modify the demo to meet your needs, you’ll effectively
just be modifying the plan along with the Python code defining
the model and the data loader in order to meet your requirements.
Otherwise, the same scripts will apply. When in doubt,
look at the FL plan’s YAML file.

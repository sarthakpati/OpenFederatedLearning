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


.. _create_initial_weights:

************************
Creating Initial Weights
************************

At the start of federated training, the aggregator shares the initial global model with all collaborators to use for their model initializations.
Collaborator model state is held locally in the form of a 'tensor dictionary', providing a map of parameter names to numpy array values. These tensor dictionaries are split by our framework code (using default logic, which can be changed via the flplan) into a portion for global sharing (parameters to be aggregation for example) and a portion for holding-out from sharing. The shared portion is converted into a protobuf file for serialization, in order to be sent over the network or saved to file. The initial weights file is such a file, holding the initial global state of the model to be used for a particular federation. 

The initial weights file can be derived from another file produced by a native ML framework (such as PyTorch, Tensorflow, ...) or can be generated without such a file - using random initialization of the model to generate the state.

In both cases, the initial weights file is generated using the script, '/bin/create_initial_weihgts_file_from_flplan.py', which is run from the bin directory using:

.. code-block:: console

  $ ../venv/bin/python create_initial_weights_file_from_flplan.py -p <fplan fname> -c <collaborator list fname>




This script creates an instance of the model provided in the plan, populates it's state (via a native weights file or random initialization), then pulls and saves a serialized form of the globally shared state to disk in the folder 'bin/federations/weights/' under the filename provided in the plan. 

Because a model is instantiated, dependencies of model instantiation (such as an example feature shape and number of output classes) need to be provided. Currently such information can passed as arguments to the script directly, or inferred from the data that is defined in the plan.

Passing of the feature shape and number of classes to the script is done via the arguments -fs and -nc respectively (-fs should be followed by a list of integers separated by spaces. Allowing the script to infer this information requires that a collaborators list file name is passed to the script (using the -c argument), as well as that an entry can be found in the local_data_config corresponding to the first collaborator in the corresponding list.

Finally, if utilizing a native weights file, pass the absolute path of the file to the script using the -nmfw argument. *Note*: Only PyTorch native weights are currently supported.


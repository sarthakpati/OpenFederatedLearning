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


************************************
Running a Federation (MNIST Example)
************************************


We will be training an MNIST classifier using federated learning and two collaborators. We will use an flplan (keras_cnn_mnist_2.yaml) already provided in the repo inside /bin/federations/plans, as well as a provided collaborators list file (cols_2.yaml inside /bin/federations/collaborator_lists) containing two collaborator names, 'col_0' and 'col_1'. Both collaborator names are already provided in the default local_data_config file so that the framework can locate the collaborator specific data information (which in this case consists of a shard number to use for hard-coded sharding logic that is performed after grabbing MNIST from a hard coded online location).

.. toctree::

   running_federation.configure
   running_federation.run
   

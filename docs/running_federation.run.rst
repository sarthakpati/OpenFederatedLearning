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

Running the Federation
######################



On the Aggregator
~~~~~~~~~~~~~~~~~


1.	To start the aggregator, run the following script from the bin directory. Note that we will need to pass in the shared single collaborator cert common name in order to specify that we are running in single cert test mode (**this mode should only be used for testing purposes**).

.. code-block:: console

   $ ../venv/bin/python3 ./run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml -scn test --c cols_2.yaml



At this point, the aggregator is running and waiting
for the collaborators to connect. When all of the collaborators
connect, the aggregator starts training. When the last round of
training is complete, the aggregator stores the final weights in
the protobuf file that was specified in the YAML file
(in this case *keras_cnn_mnist_latest.pbuf*).


On each Collaborator [col_number = '0' and '1']:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2. From the bin directory, run the collaborator using the following script, passing in the shared single collaborator cert common name in order to specify that we are running in single cert test mode (**this mode should only be used for testing purposes**).

.. code-block:: console

   $ ../venv/bin/python3 ./run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml -col col_<col_number> -scn test

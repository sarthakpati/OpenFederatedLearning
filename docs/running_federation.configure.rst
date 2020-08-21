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

Setup and Configuration
#############################

We will show you how to set up |productName| using a simple `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_
dataset and a `TensorFlow/Keras <https://www.tensorflow.org/>`_
CNN model as an example.


On the Aggregator
~~~~~~~~~~~~~~~~~

1. Go through :ref:`installation_and_setup` on the aggregator machine with the following in mind. In this case, we require make install_openfl and make install_openfl_tensorflow. The tensorflow piece is required, as we will have the aggregator create the initial weights using the model code. The aggregator does not require this otherwise,and indeed the creation of initial weights can be done on a collaborator machine and copied to the aggregator (who needs it to start the federation) if you wish. Though the collaborator list we use here is already provided in the repository (and is already compatible with the flplan and local_data_config default file), you will also need to make sure you create a copy of the network config as part of this setup and enter in the appropriate FQDN for the machine that is running the aggregator. Finally the test PKI will also be needed, and we will set up a single cert (use the common name, 'test', to be shared by the two collaborators).


2.	Create the initial weights file on the aggregator machine by running the following command from the bin directory:

.. code-block:: console

   $ ../venv/bin/python3 create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml -c cols_2.yaml


On each Collaborator Running on a Different Machine than the Aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3. Go through installation and setup (see Installation and Setup), with the following in mind. In this case, we require make install_openfl and make install_openfl_tensorflow. The tensorflow piece is required, as we will be running the models here. Finally, copy over the following files from the aggregator machine to these machines: 

 

+-----------------------------------+--------------------------------------------------------------+
| File Type                         | Filename                                                     |
+===================================+==============================================================+
| default network file              | bin/federations/plans/defaults/network.yaml                  |
+-----------------------------------+--------------------------------------------------------------+
| certificate authority cert        | bin/federations/pki/cert_chain.crt                           |
+-----------------------------------+--------------------------------------------------------------+
| shared client test public key     | bin/federations/pki/col_test/col_test.crt                    |
+-----------------------------------+--------------------------------------------------------------+
| shared client test private key    | bin/federations/pki/col_test/col_test.key                    |                                                     
+-----------------------------------+--------------------------------------------------------------+




    
    
 

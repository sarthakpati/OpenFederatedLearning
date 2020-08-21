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

Running a Federation Simulation (no network, single process)
-------------------------------------------

When exploring the convergence properties of federated learning for a particular use-case, it is helpful to run several federations in parallel, each of which runs the aggregator and collaborators (round-robin) in a single process avoiding the need for network communication. We describe here how to run one of these simulations.

Note that much of the code used for simulation (ex. collaborator and aggregator objects) is the
same as for the multiprocess solution with grpc. Since the collaborator calls the aggregator object 
methods via the grpc channel object, simulation is performed by simply replacing the channel object
provided to each collaborator with the aggregator object.

Simulations are run from an flplan, and in fact the same flplan that is used for a multi-process federation can be used.  

**Note**: Simulations utilize a single model, with each new collaborator taking control of the model when it is their turn in the round-robin. It is therefore critical that the model, 'set_tensor_dict' method completely overwrites all substantive model state in order that state does not leak from the collabotor who previously held the model.

The steps for running a simulation
----------------------------------


1. Go through the steps for project :ref:`installation_and_setup`  with the following in mind. In this case, we require make install_openfl as well as installation of ML framework libraries (supported now are make install_openfl_tensorflow and make install_openfl_pytorch) as we will be running both the aggregator and all collaborators (and thus models) here. Though the network will not be used here, we still currently require that the flplan default network file is present even when running simulations, however it's contents can be exactly the same as the example file as far as running simulation goes. Finally, the PKI creation step can be skipped as we will not be using the network here.

2. From the bin directory, run the following command (see the notes on :ref:`creating_initial_weights` from the flplan for further options on parameters to this script) :

.. code-block:: console

  $ ../venv/bin/python create_initial_weights_file_from_flplan.py -p <flpan filename> -c <collaborators list filename>

3. Again from the bin directory, kick off the simulation by running the following: 

.. code-block:: console

  $ ../venv/bin/python run_simulation_from_flplan.py -p <flpan filename> -c <collaborators list filename>



4. You'll find the output from the aggregator in bin/logs/aggregator.log. Grep this file to see results. You can check the progress as the simulation runs, if desired. The aggregator.log is always appended to, so will include results from previous runs.





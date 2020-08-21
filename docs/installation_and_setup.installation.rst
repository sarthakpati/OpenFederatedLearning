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

************
Installation
************

|productName| consists of a core package set and 3 optional package sets:

1. The core |prod| packages require no machine learning frameworks (all numpy-based). This includes the logic for the aggregagtor, collaborator, network, and model/data interfaces.
2. The |pt| packages for model and data baseclasses to simplify porting |pt| models to |prod|. (Optional)
3. The |tf| packages for model and data baseclasses to simplify porting |tf| models to |prod|. (Optional)
4. The |fets| packages, as a submodule, that contains the |fets| model and data classes. (Optional, requires submodule init)

Our scripts create a Python 3 virtual environment at ./venv which we use to run our python scripts. You can use the make file to either install these packages in this virtual-environment, or to create wheel files for you to install in another environment.

Requirements
############

On each machine in your federation, you will need:

1. Python 3.5+
2. Python virtual environments

.. note::
   You can install virtual environment support in your Python3 installation via:
   
   .. code-block:: console

      $ python3 -m pip install --user virtualenv

   If you have trouble installing the virtual environment, make sure you have Python 3 installed on your OS. For example, on Ubuntu:

   .. code-block:: console

     $ sudo apt-get install python3-pip
   
   See the official `Python website <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv>`_ for more details.


Installing In The OpenFL Virtual environment
############################################

To install the core |prod| package in ./venv, navigate to the root |prod| directory and run:


.. code-block:: console
   :substitutions:

   $ make install_|pkg|

This will create the virtual environment install the core |prod| packages.

.. note::
   The Python version used will be the same Python version referenced as Python3 by your system.


For the optional |pt| packages, run:

.. code-block:: console
   :substitutions:

   $ make install_|pkg|_pytorch

.. note::
   You will need to install pytorch and torchvision as detailed here:
   `Pytorch website <https://pytorch.org/get-started/locally/>`_.
   To install in the virtual environment, use pip as:
   venv/bin/pip

For the optional |tf| packages, run:

.. code-block:: console
   :substitutions:

   $ make install_|pkg|_tensorflow

Finally, to download the |fets| algorithms, need to initialize the submodule and run the make recipe:

.. code-block:: console

   $ git submodule update --init --recursive
   $ make install_fets


(Optional) Building Wheel Files
###############################

If you want to install |prod| and related optional packages in another Python3 environment, you can build the wheel files with the make commands:

.. code-block:: console
   :substitutions:

   $ make |pkg|_whl
   $ make |pkg|_pytorch_whl
   $ make |pkg|_tensorflow_whl
   $ make fets_whl


.. note::
   Running |prod| in containers (e.g. Docker, Singularity) is natural solution to simplify deployment, and fairly straight-forward. We welcome contributions towards such a solution. 

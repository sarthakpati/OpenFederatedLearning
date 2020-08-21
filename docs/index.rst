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


.. Documentation master file, created by
   sphinx-quickstart on Thu Oct 24 15:07:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************************
Welcome to |productName| documentation!
***************************************

|productName| is a Python3 library for federated learning.
It enables organizations to collaborately train a
model without sharing sensitive information with each other.

There are basically two components in the library:
the *collaborator* which uses local sensitive dataset to fine-tune
the aggregated model and the *aggregator* which receives
model updates from collaborators and distribute the aggregated
models.

The *aggregator* is framework-anostic, while the *collaborator*
can use any deep learning frameworks, such as Tensorflow or
Pytorch.

|productName| is developed by Intel Labs and Intel Internet of Things Group.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   manual


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

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

# Models for Federated Learning

The folder contains runnable code for different models (and a generic test module to validate the code) in the FL framework. 

As our tool develops, the collaborators will eventually download the code to join federated learning. 


## Interface
The current examples are classes, whose constructors take a data object along with other key word arguments. In order to run federations (or single process simulations), the model class should have the attributes:

* get_data()
* set_data(data_object)
* train_epoch()
* get_training_data_size()
* validate()
* get_validation_data_size()
* get_tensor_dict(Boolean with_opt_vars)
* set_tensor_dict(tensor_dict)
* reset_opt_vars()
* tensor_dict_split_fn_kwargs   
    Determines which params to hold out from aggregation, 
    Ex: {holdout_types=['non_float'], holdout_tensor_names=[]}


We may also add this for the aggregator:

* export_initial_weights(fpath)


And for local test:
* load_weights(weight_fpath)
    to use the trained weights.


## Proposal of Revised Interface

* train_epoch() --> train(iterations=5)
> so that we can support flexible training iterations.
* validate() returns a dictionary of {metric: value} so that we can report and aggregate multiple metircs.


## TODO

* multiple metrics to calculate in validate()
* get_*_data_size(): 
* validate() returns dict()


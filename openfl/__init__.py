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

from .utils import load_yaml, get_object, split_tensor_dict_for_holdouts, hash_string


def check_type(obj, expected_type, logger):
    if not isinstance(obj, expected_type):
        exception = TypeError("Expected type {}, got type {}".format(type(obj), str(expected_type)))
        logger.exception(repr(exception))
        raise exception


def check_equal(x, y, logger):
    if x != y:
        exception = ValueError("{} != {}".format(x, y))
        logger.exception(repr(exception))
        raise exception


def check_not_equal(x, y, logger, name='None provided'):
    if x == y:
        exception = ValueError("Name {}. Expected inequality, but {} == {}".format(name, x, y))
        logger.exception(repr(exception))
        raise exception

def check_is_in(element, _list, logger):
    if element not in _list:
        exception = ValueError("Expected sequence memebership, but {} is not in {}".format(element, _list))
        logger.exception(repr(exception))
        raise exception

def check_not_in(element, _list, logger):
    if element in _list:
        exception = ValueError("Expected not in sequence, but {} is in {}".format(element, _list))
        logger.exception(repr(exception))
        raise exception

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

import os
import logging
import logging.config
import coloredlogs
import yaml

def setup_logging(path="logging.yaml", default_level='info', logging_directory=None):
    """Defines the various log levels

    * 'notset': logging.NOTSET
    * 'debug': logging.DEBUG
    * 'info': logging.INFO
    * 'warning': logging.WARNING
    * 'error': logging.ERROR
    * 'critical': logging.CRITICAL

    Args:
        path: path for logging file configuration (Default = "logging.yaml")
        default_level: Default level for logging (Default = "info")

    """
    logging_level_dict = {
     'notset': logging.NOTSET,
     'debug': logging.DEBUG,
     'info': logging.INFO,
     'warning': logging.WARNING,
     'error': logging.ERROR,
     'critical': logging.CRITICAL
    }

    default_level = default_level.lower()
    if default_level not in logging_level_dict:
        raise Exception("Not supported logging level: %s", default_level)
    default_level = logging_level_dict[default_level]

    if os.path.isfile(path):
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f.read())
                # create directories (if needed) to hold file handler output files
                for handler_name in config["handlers"]:
                    handler_config = config["handlers"][handler_name]
                    if "filename" in handler_config:
                        # FIXME: quick fix to enable overriding the log directory. This is a near-term hack. We need to rethink logging in general.
                        if logging_directory is not None:
                            handler_config['filename'] = os.path.join(logging_directory, os.path.basename(handler_config['filename']))
                        file_path = os.path.abspath(handler_config["filename"])
                        logdir_path = os.path.dirname(file_path)
                        if not os.path.exists(logdir_path):
                            print("Creating log directory: ", logdir_path)
                            os.makedirs(logdir_path)
                logging.config.dictConfig(config)
                coloredlogs.install()
                print("Loaded logging configuration: %s" % path)
            except Exception as e:
                print("Trouble loading logging configuration with file [%s]." % path)
                print(e)
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        print("Logging configuration file [%s] not found." % path)
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)

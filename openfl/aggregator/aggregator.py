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

import time
import os
import logging
import time
from collections import namedtuple

import numpy as np
import yaml
from threading import Lock

from .. import check_equal, check_not_equal, check_is_in, check_not_in, load_yaml, hash_string
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader, ModelHeader, TensorProto
from ..proto.collaborator_aggregator_interface_pb2 import GlobalTensor, JobReply, ResultsAck, ExtraModelInfo, RoundSummary
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_UPLOAD_RESULTS, JOB_SLEEP, JOB_QUIT
from openfl.proto.protoutils import dump_proto, load_proto, tensor_proto_to_numpy_array, numpy_array_to_tensor_proto


# FIXME: simpler "stats" collection/handling
# FIXME: remove the round tracking/job-result tracking stuff from this?
# Feels like it conflates model aggregation with round management
# FIXME: persistence of the trained weights.
class Aggregator(object):
    """An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (string)                    : UUID of this object.
        federation_uuid (string)                    : Federation UUID.
        collaborator_common_names (list of str)     : The list of approved collaborator IDs. These IDs should match the common_names in the collaborator certificates, unless "single_col_cert_common_name" is specified.
        init_model_fpath (string)                   : The filepath of the initial weights file.
        latest_model_fpath (string)                 : The filepath to store the latest aggregated weights
        best_model_fpath (string)                   : The filepath to store the weight of the best model.
        rounds_to_train (int)                       : Number of rounds to train (default: 256)
        minimum_reporting (int)                     : Aggregator will not end a round early until this number of clients has reported in. (default: -1)
        straggler_cutoff_time (scalar)              : Aggregator will not end a round early until this number of seconds has passed. (default: np.inf)
        single_col_cert_common_name (string)        : (default: None)
        compression_pipeline                        : (default: None)
        init_metadata_fname (string)                : (default: None)
        latest_metadata_fname (string)              : (default: None)
        send_metadata_to_clients (bool)             : (default: False)
        model_selection_val_keys (list of string)   : (default: None)
        kwargs                                      : Currently unused
    """
    # FIXME: no selector logic is in place
    def __init__(self,
                 aggregator_uuid,
                 federation_uuid,
                 collaborator_common_names,
                 model_directory,
                 initial_model='initial',
                 rounds_to_train=256,
                 minimum_reporting=-1,
                 straggler_cutoff_time=np.inf,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,
                 send_metadata_to_clients=False,
                 backup_path=None,
                 runtime_aggregator_config_dir=None,
                 runtime_configurable_params=[],
                 collaborator_sleep_time=10,
                 best_model_metric='shared_model_validation',
                 enrollment_period=np.inf,
                 model_selection_val_keys=None,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.per_example_validation_logger = logging.getLogger('openfl.per_example_validation')
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.model_directory = model_directory
        self.collaborator_common_names = collaborator_common_names
        self.rounds_to_train = rounds_to_train
        self.quit_job_sent_to = []
        self.minimum_reporting = minimum_reporting
        self.straggler_cutoff_time = straggler_cutoff_time
        self.round_start_time = None
        self.single_col_cert_common_name = single_col_cert_common_name
        self.collaborator_sleep_time = collaborator_sleep_time
        self.enrollment_period = enrollment_period
        self.enrolled = []

        self.model_selection_val_keys = model_selection_val_keys

        self.backup_path = backup_path
        self.runtime_aggregator_config_dir = runtime_aggregator_config_dir
        self.runtime_configurable_params = runtime_configurable_params

        self._GRACEFULLY_QUIT = False
        self.best_model_metric = best_model_metric
        self.ignore_list = []

        if self.runtime_aggregator_config_dir is not None:
            self.update_config_from_filesystem()

        if self.single_col_cert_common_name is not None:
            self.log_big_warning()
        else:
            self.single_col_cert_common_name = '' # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?

        self.tensors = {}
        self.load_model(os.path.join(model_directory, initial_model))

        self.round_num = self.model_header.version + 1

        self.round_summary = 'start of federation'

        self._do_quit = False

        self.initialize_round_results()
        self.best_model_score = None
        self.mutex = Lock()

    def create_task_list(self):
        self.metrics_tasks = ['shared_model_validation', 'loss', 'local_model_validation']
        # FIXME: this is fixed for now to the same each round
        self.tasks = self.metrics_tasks[0:2]
        self.tasks.extend(self.model_tensors)
        self.tasks.append(self.metrics_tasks[2])
        self.logger.info("Tasks set to {}".format(self.tasks))        

    def load_model(self, directory):
        self.model_header = load_proto(os.path.join(directory, 'ModelHeader.pbuf'), proto_type=ModelHeader)
        self.extra_model_info = load_proto(os.path.join(directory, 'ExtraModelInfo.pbuf'), proto_type=ExtraModelInfo)

        self.model_tensors = self.extra_model_info.tensor_names
        self.create_task_list()

        for t in self.extra_model_info.tensor_names:
            t_hash = hash_string(t)
            tensor_proto = load_proto(os.path.join(directory, '{}.pbuf'.format(t_hash)), proto_type=TensorProto)
            if t != tensor_proto.name:
                raise RuntimeError("Loaded the wrong tensor! Meant to load: {} did load: {} read file: {}".format(t, tensor_proto.name, t_hash))
            self.tensors[t] = tensor_proto_to_numpy_array(tensor_proto)

    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)

        dump_proto(self.model_header, os.path.join(directory, 'ModelHeader.pbuf'))
        dump_proto(self.extra_model_info, os.path.join(directory, 'ExtraModelInfo.pbuf'))

        for k, v in self.tensors.items():
            t_hash = hash_string(k)
            proto = numpy_array_to_tensor_proto(v, k)
            dump_proto(proto, os.path.join(directory, '{}.pbuf'.format(t_hash)))

    def save_local_update(self, directory, result):
        os.makedirs(directory, exist_ok=True)

        t_hash = hash_string(result.task)
        dump_proto(result, os.path.join(directory, '{}.pbuf'.format(t_hash)))

    def initialize_round_results(self):
        self.round_results = RoundTaskResults(self.collaborator_common_names, self.tasks, self.logger, self.metrics_tasks)

    def update_config_from_filesystem(self):
        if self.runtime_aggregator_config_dir is None:
            return

        # make the directory for convenience
        config_dir = os.path.join(self.runtime_aggregator_config_dir, self.federation_uuid)
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, 'agg_control.yaml')

        if os.path.exists(config_file):
            config = load_yaml(config_file)
            self.logger.info("Updating aggregator config from {}".format(config_file))
        else:
            self.logger.info("Aggregator did not find config file: {}".format(config_file))
            return

        # if "ignore_list" is not present, reset it to empty
        if 'ignore_list' not in config:
            self.ignore_list = []

        for k, v in config.items():
            if k == '_GRACEFULLY_QUIT':
                setattr(self, k, v)
                self.logger.info("Aggregator config {} updated to {}".format(k, v))
            elif k == 'ignore_list':
                setattr(self, k, v)
                self.logger.info("Aggregator config {} updated to {}".format(k, v))
            elif k not in self.runtime_configurable_params:
                self.logger.warning("Aggregator config file contains {}. This is not allowed by the flplan.".format(k))
            elif not hasattr(self, k):
                self.logger.warning("Aggregator config file contains {}. This is not a valid aggregator parameter".format(k))
            else:
                setattr(self, k, v)
                self.logger.info("Aggregator config {} updated to {}".format(k, v))

    def valid_collaborator_CN_and_id(self, cert_common_name, collaborator_common_name):
        """Determine if the collaborator certificate and ID are valid for this federation.

        Args:
            cert_common_name: Common name for security certificate
            collaborator_common_name: Common name for collaborator

        Returns:
            bool: True means the collaborator common name matches the name in the security certificate.

        """
        # if self.test_mode_whitelist is None, then the common_name must match collaborator_common_name and be in collaborator_common_names
        if self.single_col_cert_common_name == '':  # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?
            return cert_common_name == collaborator_common_name and collaborator_common_name in self.collaborator_common_names
        # otherwise, common_name must be in whitelist and collaborator_common_name must be in collaborator_common_names
        else:
            return cert_common_name == self.single_col_cert_common_name and collaborator_common_name in self.collaborator_common_names

    def time_to_quit(self):
        return self._do_quit or self.all_quit_jobs_sent()

    def all_quit_jobs_sent(self):
        """Determines if all collaborators have been sent the QUIT command.

        Returns:
            bool: True if all collaborators have been sent the QUIT command.

        """
        return sorted(self.quit_job_sent_to) == sorted(self.collaborator_common_names)

    def validate_header(self, message):
        """Validates the message is from valid collaborator in this federation.

        Returns:
            bool: True if the message is from a valid collaborator in this federation.

        """

        # validate that the message is for me
        check_equal(message.header.recipient, self.uuid, self.logger)

        # validate that the message is for my federation
        check_equal(message.header.federation_id, self.federation_uuid, self.logger)

        # validate that the sender is one of my collaborators
        check_is_in(message.header.sender, self.collaborator_common_names, self.logger)

        # check that we agree on single_col_cert_common_name
        check_equal(message.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

        # check that we are using the same model
        check_equal(message.header.model_header.id, self.model_header.id, self.logger)

    def straggler_time_expired(self):
        """Determines if there are still collaborators that have not returned past the expected round time.
        Returns:
            bool: True if training round limit has expired (i.e. there are straggler collaborators that have not returned in the expected time)

        """
        return self.round_start_time is not None and ((time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def minimum_collaborators_reported(self):
        """Determines if enough collaborators have returned to do the aggregation.

        Returns:
            bool: True if the number of collaborators that have finished is greater than the minimum threshold set.

        """
        return self.round_results.num_collaborators_done() >= self.minimum_reporting

    def straggler_cutoff_check(self):
        """Determines if we will end the round now, cutting off any remaining stragglers.

        Returns:
            bool: True if collaborator c is done.

        """
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported()
        if cutoff:
            collaborators_done = self.round_results.num_collaborators_done()
            self.logger.info('\tEnding round early due to straggler cutoff. Collaborators done: {}'.format(collaborators_done))
        return cutoff

    def end_of_round_check(self):
        # at least "minimum_reporting" collaborators must be enrolled
        if len(self.enrolled) < self.minimum_reporting:
            return

        # check if we need to end due to straggler cutoff or because all enrolled collaborators are done
        straggler_cutoff = self.straggler_cutoff_check()
        enrolled_collaborators_done = all([self.round_results.is_collaborator_done_for_round(c) for c in self.enrolled])

        if straggler_cutoff or enrolled_collaborators_done:
            self.end_of_round()

    def end_of_round(self):
        # FIXME: proper logging of metrics
        metrics_log_string = '\n**** END OF ROUND {} ****\n'.format(self.model_header.version)
        metrics_log_string += 'round results for model id/version {}/{}'.format(self.model_header.id, self.model_header.version) 
        
        # FIXME: dictionary handling is wonky
        for metric in self.metrics_tasks:
            r = self.round_results.task_results[metric]
            num_contributors = self.round_results.num_collaborators_done(task=metric)
            if isinstance(r.value, dict):
                metrics_log_string += '\n\t{}: total_samples: {} num_contributors: {}'.format(metric, r.weight, num_contributors)
                for k in sorted(r.value.keys()):
                    v = r.value[k]
                    metrics_log_string += '\n\t\t{}: {}'.format(k, v)
            else:
                metrics_log_string += '\n\t{}: {}, total_samples: {} num_contributors: {}'.format(metric, r.value, r.weight, num_contributors)
        self.logger.info(metrics_log_string)

        # update the shared tensor values
        for k in self.tensors.keys():
            self.tensors[k] = self.round_results.get_tensor(k)

        # update our model header
        self.model_header = ModelHeader(id=self.model_header.id,
                                        version=self.model_header.version + 1)

        # Save the new model as latest model
        self.save_model(os.path.join(self.model_directory, 'latest'))

        # if configured, also save to the backup location
        if self.backup_path is not None:
            self.save_model(os.path.join(self.backup_path, str(self.round_num)))

        # get the model score
        # if dictionary, we simply average their values (considering selection keys if appropriate)
        model_score = self.round_results.task_results[self.best_model_metric].value
        if isinstance(model_score, dict):
            if self.model_selection_val_keys is not None:
                model_subscores = [val for key, val in model_score.items() if key in self.model_selection_val_keys]
            else:
                model_subscores = list(model_score.values())
            model_score = np.average(model_subscores)

        new_best_model = False
        if self.best_model_score is None:
            new_best_model = True
        else:
            new_best_model = model_score > self.best_model_score

        if new_best_model:
            self.logger.info("Saved new best model with score {}.".format(model_score))
            self.best_model_score = model_score
            # Save a model proto version to file as current best model.
            self.save_model(os.path.join(self.model_directory, 'best'))

        # FIXME: use pprint?
        # create the collaborator round metrics
        self.round_summary = 'round: {}\n'.format(self.round_num)
        self.round_summary += 'round_start: {}\n'.format(self.round_start_time)
        if isinstance(self.best_model_score, dict):
            raise ValueError('Logical error, we should have replaced out all dictionary model_scores.')
        else:
            if self.model_selection_val_keys is not None:
                self.round_summary += 'best_model_score (according to keys: {}): {}\n'.format(self.model_selection_val_keys, self.best_model_score)
            else:
                self.round_summary += 'best_model_score: {}\n'.format(self.best_model_score)

        for k in self.metrics_tasks:
            t = self.round_results.task_results[k]
            self.round_summary += '{}:\n'.format(t.name)
            self.round_summary += '\tvalue: {}\n'.format(t.value)
            self.round_summary += '\tweight: {}\n'.format(t.weight)
            self.round_summary += '\tnum contributors: {}\n'.format(self.round_results.num_collaborators_done(task=k))

        # re-initialize our round results
        self.initialize_round_results()

        # set enrolled list to blank
        self.enrolled = []

        # if we have enabled runtime configuration updates, do that now
        if self.runtime_aggregator_config_dir is not None:
            self.update_config_from_filesystem()

        self.round_num += 1
        self.logger.debug("Start a new round %d." % self.round_num)
        self.round_start_time = None

        self._do_quit = self._GRACEFULLY_QUIT
    
    def _synchronized(func):
        def wrapper(self, *args, **kwargs):
            self.mutex.acquire(blocking=True)
            try:
                ret = func(self, *args, **kwargs)
            finally:
                self.mutex.release()
            return ret
        return wrapper

    @_synchronized
    def UploadResults(self, message):
        """Parses the collaborator reply message to get the collaborator results

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """
        self.validate_header(message)
        collaborator = message.header.sender
        task = message.task

        self.logger.debug("Received results for {} from {}".format(message.task, collaborator))

        # if this is for our special print task, simply log each entry in the uploaded dictionary
        if message.task == "___RESERVED_PRINT_TASK_STRING___":
            value = dict(message.value_dict.dictionary)
            self.logger.info("Received brats stats results for {} of {}".format(collaborator, value))
            return ResultsAck(header=self.create_reply_header(message), discard_round=False)

        # if this collaborator is out of date, we need to tell them to discard the remaining updates
        if self.collaborator_out_of_sync(message.header.model_header):
            self.logger.info("{} is out of sync. Replying with discard_round=True".format(collaborator))
            return ResultsAck(header=self.create_reply_header(message), discard_round=True)

        # if the collaborator has not already done this task, we aggregate and possibly also save
        if not self.round_results.has_collaborator_done(collaborator, task):
            # determine which of our oneofs is set
            if message.WhichOneof("extra") == 'tensor':
                value = tensor_proto_to_numpy_array(message.tensor)
            elif message.WhichOneof("extra") == 'value_dict':
                value = dict(message.value_dict.dictionary)
            elif message.WhichOneof("extra") == 'list_value_dict':
                per_example_value = {key: message.list_value_dict.list_dictionary[key].value for key in message.list_value_dict.list_dictionary}
                # TODO: Log the per_example values (provided above) using a second logger
                readable_per_example_values = ''
                for k, v in per_example_value.items():
                    l = [str(x) for x in v]
                    readable_per_example_values += '{} -\n\t{}\n'.format(k, '\n\t'.join(l))
                self.per_example_validation_logger.info('{} per example results for task {} round {}:\n{}'.format(collaborator, task, message.header.model_header.version, readable_per_example_values))
                value = {key: np.average(per_example_metric) for key, per_example_metric in per_example_value.items()}
            else:
                value = message.value

            self.round_results.update_from_collaborator(collaborator, task, value, message.weight)

            # if the collaborator has not alread
            if self.backup_path is not None:
                self.save_local_update(os.path.join(self.backup_path, collaborator, str(self.round_num)), message)

        reply = ResultsAck(header=self.create_reply_header(message), discard_round=False)

        self.end_of_round_check()

        return reply

    def determine_next_job(self, message):
        # get the collaborator who sent this message
        collaborator = message.header.sender

        # FIXME: we should really have each collaborator validate one last time
        # check if we are done
        if self.collaborators_should_quit():
            # we may need to send this again if the collaborator process has been restarted for some reason
            # thus we may not need to add it to the list again
            if message.header.sender not in self.quit_job_sent_to:
                self.quit_job_sent_to.append(collaborator)        
            return JobReply(header=self.create_reply_header(message),
                            job=JOB_QUIT)

        # if _do_quit is set, we should tell them to sleep
        # this occurs because the server has not yet actually quit
        if self._do_quit:
            return JobReply(header=self.create_reply_header(message),
                            job=JOB_SLEEP,
                            seconds=self.collaborator_sleep_time)

        # if this collaborator is in the ignore list, tell them to sleep
        if collaborator in self.ignore_list:
            return JobReply(header=self.create_reply_header(message),
                            job=JOB_SLEEP,
                            seconds=self.collaborator_sleep_time)

        # FIXME: this flow needs to depend on a job selection output for the round
        # for now, all jobs require an in-sync model, so it is the first check
        # check if the sender model is out of date
        if self.collaborator_out_of_sync(message.header.model_header):
            is_enrolled = collaborator in self.enrolled
            more_than_one_round_out_of_date = self.model_header.version != (message.header.model_header.version + 1)
            # if the collaborator is enrolled or more than one round out of date, tell them to download
            if is_enrolled or more_than_one_round_out_of_date:
                return JobReply(header=self.create_reply_header(message),
                                job=JOB_DOWNLOAD_MODEL,
                                extra_model_info=self.extra_model_info)
            # otherwise, the collaborator could be in a bad loop where they start each round late, 
            # then miss each enrollment, so they should just sleep for this round
            else:
                return JobReply(header=self.create_reply_header(message),
                                job=JOB_SLEEP,
                                seconds=self.collaborator_sleep_time)
        
        # the collaborator has the shared model, so now determine what the next upload should be
        next_upload = self.next_collaborator_upload(collaborator)
        if next_upload is not None:
            return JobReply(header=self.create_reply_header(message),
                            job=JOB_UPLOAD_RESULTS,
                            name=next_upload)
        # otherwise, the collaborator is done
        else:
            return JobReply(header=self.create_reply_header(message),
                            job=JOB_SLEEP,
                            seconds=self.collaborator_sleep_time)

    def next_collaborator_upload(self, collaborator):
        for task in self.tasks:
            if not self.round_results.has_collaborator_done(collaborator, task):
                return task
        return None

    @_synchronized
    def DownloadRoundSummary(self, message):
        self.validate_header(message)

        return RoundSummary(header=self.create_reply_header(message),
                            summary=str(self.round_summary))

    @_synchronized
    def RequestJob(self, message):
        """Parse message for job request and act accordingly.

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """
        self.validate_header(message)
        collaborator = message.header.sender

        # start round if it hasn't already
        if self.round_start_time is None:
            self.round_start_time = time.time()

        # check enrollment
        self.check_enrollment(collaborator)

        reply = self.determine_next_job(message)

        # we log download jobs as info, others as debug
        if reply.job is JOB_DOWNLOAD_MODEL:
            self.logger.info("Receive job request from %s and assign with %s" % (collaborator, Job.Name(reply.job)))
        else:
            self.logger.debug("Receive job request from %s and assign with %s" % (collaborator, Job.Name(reply.job)))

        if self.straggler_cutoff_time != np.inf and reply.job is JOB_SLEEP:
            # we have an idle collaborator and a straggler cutoff time, 
            # so we should check for early round end
            self.end_of_round_check()
        
        return reply

    @_synchronized
    def DownloadTensor(self, message):
        """Sends a tensor to the collaborator

        Args:
            message: Message from the collaborator

        Returns:
            A GlobalTensor

        """
        self.validate_header(message)

        self.logger.debug("Received DownloadTensor request from {} for {}".format(message.header.sender, message.tensor_name))

        if message.tensor_name not in self.tensors:
            raise RuntimeError("Requested tensor {} from {} not in our list of tensors!".format(message.tensor_name, message.header.sender))

        tensor_proto = numpy_array_to_tensor_proto(self.tensors[message.tensor_name],
                                                   message.tensor_name)

        reply = GlobalTensor(header=self.create_reply_header(message),
                             tensor=tensor_proto)

        return reply

    def check_enrollment(self, collaborator):
        # if already enrolled
        if collaborator in self.enrolled:
            return

        # check if enrollment is over
        # enrollment lasts until we have met the following criteria:
        # 1. more time has passed than the "enrollment period" value
        # 2. at least "minimum_reporting" collaborators are enrolled
        t = time.time() - self.round_start_time
        if t > self.enrollment_period and len(self.enrolled) >= self.minimum_reporting:
            return

        # enroll this collaborator
        self.logger.info("Enrolling collaborator {}".format(collaborator))
        self.enrolled.append(collaborator)

    def collaborators_should_quit(self):
        return self.round_num > self.rounds_to_train

    def create_reply_header(self, message):
        """Creates a header for the reply to the message

        Args:
            message: Message from the collaborator

        Returns:
            The message header.

        """
        return MessageHeader(sender=self.uuid,
                             recipient=message.header.sender,
                             federation_id=self.federation_uuid,
                             counter=message.header.counter,
                             single_col_cert_common_name=self.single_col_cert_common_name,
                             model_header=self.model_header)

    def collaborator_out_of_sync(self, model_header):
        """Determines if the collaborator has the wrong version of the model (aka out of sync)

        Args:
            model_header: Header for the model

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """
        # validate that this is the right model to be checking
        check_equal(model_header.id, self.model_header.id, self.logger)

        return model_header.version != self.model_header.version

    def log_big_warning(self):
        self.logger.warning("\n{}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS NOT PROPER PKI AND SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN WARNED!!!".format(the_dragon))


# this holds the streaming average results for the tasks, including both metrics and scalars
class RoundTaskResults(object):
    def __init__(self, collaborators, tasks, logger, log_these=None):
        self.collaborators = collaborators
        self.tasks = tasks
        self.logger = logger
        self.log_these = log_these
        # FIXME: This has all collaborators do all tasks for a round
        self.task_results = {t: StreamingAverage(t, logger) for t in tasks}

    def has_collaborator_done(self, collaborator, task):
        return self.task_results[task].has_collaborator_done(collaborator)

    def is_collaborator_done_for_round(self, collaborator):
        for task in self.tasks:
            if not self.has_collaborator_done(collaborator, task):
                return False
        return True

    def num_collaborators_done(self, task=None):
        if task is None:
            return sum([self.is_collaborator_done_for_round(c) for c in self.collaborators])
        else:
            return sum([self.has_collaborator_done(c, task) for c in self.collaborators])

    def is_task_done_for_round(self, task):
        for collaborator in self.collaborators:
            if not self.has_collaborator_done(collaborator, task):
                return False
        return True

    def update_from_collaborator(self, collaborator, task, value, weight):
        if self.log_these is not None and self.logger is not None and task in self.log_these:
            if isinstance(value, dict):
                message = "Received update from {} for {} with weight {} and values:".format(collaborator, task, weight)
                for k in sorted(value.keys()):
                    message += "\n\t{}: {}".format(k, value[k])
                self.logger.info(message)
            else:
                self.logger.info("Received update from {} for {} with value {} and weight {}".format(collaborator, task, value, weight))
        self.task_results[task].update_from_collaborator(collaborator, value, weight)
    
    def get_tensor(self, tensor_name):
        value = self.task_results[tensor_name].value
        if not isinstance(value, np.ndarray):
            raise RuntimeError("Tensor {} is not an ndarray!".format(tensor_name))
        return value

# stores the streaming average of an aggregated value during a round
class StreamingAverage(object):

    def __init__(self, name, logger):
        self.name = name
        self.value = None
        self.weight = None
        self.updated_by = []
        self.logger = logger

    def has_collaborator_done(self, collaborator):
        return collaborator in self.updated_by

    def _has_nans_inner(self, value):
        try:
            has_nans = np.isnan(value)
        except Exception as e:
            self.logger.critical("Failed isnan check for type {}".format(type(value)))
            return False
        try:
            has_nans = np.any(has_nans)
        except TypeError as te:
            pass
        return has_nans

    def _has_nans(self, value):
        if isinstance(value, dict):
            for v in value.values():
                if self._has_nans_inner(v):
                    return True
            return False
        else:
            return self._has_nans_inner(value)

    def update_from_collaborator(self, collaborator, value, weight):
        if self.has_collaborator_done(collaborator):
            return

        self.updated_by.append(collaborator)

        # check if value contains nans
        if self._has_nans(value):
            self.logger.critical("NANs detected in results for {} from {}".format(self.name, collaborator))
            return

        if self.value is None:
            self.value = value
            self.weight = weight
        else:
            # if a dictionary, we need to update each separately
            if isinstance(self.value, dict):
                for k, v in self.value.items():
                    self.value[k] = self.weighted_average([v, value[k]], [self.weight, weight])
            else:
                self.value = self.weighted_average([self.value, value], [self.weight, weight])
            self.weight += weight

    def weighted_average(self, values, weights):
        axis = None
        if isinstance(values[0], np.ndarray):
            axis = 0
        return np.average(values, weights=weights, axis=axis)

    # unformatted, parseable
    def __repr__(self):
        return "{}, weight: {}, value: {}".format(self.name, self.weight, self.value)

    # FIXME: make prettier
    def __str__(self):
        return self.__repr__()


the_dragon = """

 ,@@.@@+@@##@,@@@@.`@@#@+  *@@@@ #@##@  `@@#@# @@@@@   @@    @@@@` #@@@ :@@ `@#`@@@#.@
  @@ #@ ,@ +. @@.@* #@ :`   @+*@ .@`+.   @@ *@::@`@@   @@#  @@  #`;@`.@@ @@@`@`#@* +:@`
  @@@@@ ,@@@  @@@@  +@@+    @@@@ .@@@    @@ .@+:@@@:  .;+@` @@ ,;,#@` @@ @@@@@ ,@@@* @
  @@ #@ ,@`*. @@.@@ #@ ,;  `@+,@#.@.*`   @@ ,@::@`@@` @@@@# @@`:@;*@+ @@ @`:@@`@ *@@ `
 .@@`@@,+@+;@.@@ @@`@@;*@  ;@@#@:*@+;@  `@@;@@ #@**@+;@ `@@:`@@@@  @@@@.`@+ .@ +@+@*,@
  `` ``     ` ``  .     `     `      `     `    `  .` `  ``   ``    ``   `       .   `



                                            .**
                                      ;`  `****:
                                     @**`*******
                         ***        +***********;
                        ,@***;` .*:,;************
                        ;***********@@***********
                        ;************************,
                        `*************************
                         *************************
                         ,************************
                          **#*********************
                          *@****`     :**********;
                          +**;          .********.
                          ;*;            `*******#:                       `,:
                                          ****@@@++::                ,,;***.
                                          *@@@**;#;:         +:      **++*,
                                          @***#@@@:          +*;     ,****
                                          @*@+****           ***`     ****,
                                         ,@#******.  ,       ****     **;,**.
                                         * ******** :,       ;*:*+    **  :,**
                                        #  ********::      *,.*:**`   *      ,*;
                                        .  *********:      .+,*:;*:   :      `:**
                                       ;   :********:       ***::**   `       ` **
                                       +   :****::***  ,    *;;::**`             :*
                                      ``   .****::;**:::    *;::::*;              ;*
                                      *     *****::***:.    **::::**               ;:
                                      #     *****;:****     ;*::;***               ,*`
                                      ;     ************`  ,**:****;               ::*
                                      :     *************;:;*;*++:                   *.
                                      :     *****************;*                      `*
                                     `.    `*****************;  :                     *.
                                     .`    .*+************+****;:                     :*
                                     `.    :;+***********+******;`    :              .,*
                                      ;    ::*+*******************. `::              .`:.
                                      +    :::**********************;;:`                *
                                      +    ,::;*************;:::*******.                *
                                      #    `:::+*************:::;********  :,           *
                                      @     :::***************;:;*********;:,           *
                                      @     ::::******:*********************:         ,:*
                                      @     .:::******:;*********************,         :*
                                      #      :::******::******###@*******;;****        *,
                                      #      .::;*****::*****#****@*****;:::***;  ``  **
                                      *       ::;***********+*****+#******::*****,,,,**
                                      :        :;***********#******#******************
                                      .`       `;***********#******+****+************
                                      `,        ***#**@**+***+*****+**************;`
                                       ;         *++**#******#+****+`      `.,..
                                       +         `@***#*******#****#
                                       +          +***@********+**+:
                                       *         .+**+;**;;;**;#**#
                                      ,`         ****@         +*+:
                                      #          +**+         :+**
                                      @         ;**+,       ,***+
                                      #      #@+****      *#****+
                                     `;     @+***+@      `#**+#++
                                     #      #*#@##,      .++:.,#
                                    `*      @#            +.
                                  @@@
                                 #`@
                                  ,                                                        """

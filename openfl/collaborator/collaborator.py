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
import logging
import numpy as np

from .. import check_type, check_equal, check_not_equal, split_tensor_dict_for_holdouts
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader, ValueDictionary
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest, JobReply
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_UPLOAD_RESULTS, JOB_SLEEP, JOB_QUIT
from ..proto.collaborator_aggregator_interface_pb2 import ModelHeader, TensorProto, TensorDownloadRequest, ResultsUpload
from ..proto.protoutils import tensor_proto_to_numpy_array, numpy_array_to_tensor_proto


from enum import Enum

class OptTreatment(Enum):
    """Optimizer methods
    """

    RESET = 1
    """
    RESET tells each collaborator to reset the optimizer state at the beginning of each round.
    """
    CONTINUE_LOCAL = 2
    """
    CONTINUE_LOCAL tells each collaborator to continue with the local optimizer state from the previous round.
    """
    CONTINUE_GLOBAL = 3
    """
    CONTINUE_GLOBAL tells each collaborator to continue with the federally averaged optimizer state from the previous round.
    """

# FIXME: this is actually a tuple of a collaborator/flplan
# CollaboratorFLPlanExecutor?
class Collaborator(object):
    """The Collaborator object class

    Args:
        collaborator_common_name (string): The common name for the collaborator
        aggregator_uuid: The unique id for the aggregator
        federation_uuid: The unique id for the federation
        wrapped_model: The model
        channel (int): channel
        polling_interval (int) : The number of seconds to poll the network (Defaults to 4)
        opt_treatment (string): The optimizer state treatment (Defaults to "CONTINUE_GLOBAL", which is aggreagated state from previous round.)
        compression_pipeline: The compression pipeline (Defaults to None)
        epochs_per_round (float): Number of epochs per round (Defaults to 1.0. Note it is possible to perform a fraction of an epoch.)
        num_batches_per_round (int): Number of batches per round (Defaults to None)
        send_model_deltas (bool): True = Only model delta gets sent. False = Whole model gets sent to collaborator. (Defaults to False)
        single_col_cert_common_name: (Defaults to None)
        **kwargs : Additional parameters to pass to collaborator object
    """
    # FIXME: do we need a settable model version? Shouldn't col always start assuming out of sync?
    def __init__(self,
                 collaborator_common_name,
                 aggregator_uuid,
                 federation_uuid,
                 wrapped_model,
                 channel,
                 opt_treatment="CONTINUE_GLOBAL",
                 compression_pipeline=None,
                 epochs_per_round=1.0,
                 num_batches_per_round=None,
                 send_model_deltas = False,
                 single_col_cert_common_name=None,
                 num_retries=5,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.channel = channel
        self.num_retries = num_retries
        self.polling_interval = 10

        # this stuff is really about sanity/correctness checking to ensure the bookkeeping and control flow is correct
        self.common_name = collaborator_common_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.single_col_cert_common_name = single_col_cert_common_name
        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = '' # FIXME: this is just for protobuf compatibility. Cleaner solution?
        self.counter = 0
        self.model_header = ModelHeader(id=wrapped_model.__class__.__name__,
                                        version=-1)
        # number of epochs to perform per round of FL (is a float that is converted
        # to num_batches before calling the wrapped model train_batches method).
        # This is overridden by "num_batches_per_round"
        self.epochs_per_round = epochs_per_round
        self.num_batches_per_round = num_batches_per_round
        if num_batches_per_round is not None:
            self.logger.info("Collaborator {} overriding epochs_per_round of {} with num_batches_per_round of {}".format(self.common_name, self.epochs_per_round, self.num_batches_per_round))

        self.wrapped_model = wrapped_model
        self.tensor_dict_split_fn_kwargs = wrapped_model.tensor_dict_split_fn_kwargs or {}
        self.local_model_has_been_trained = False
        self.round_results = {}

        # RESET/CONTINUE_LOCAL/CONTINUE_GLOBAL
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error("Unknown opt_treatment: %s." % opt_treatment)
            raise NotImplementedError("Unknown opt_treatment: %s." % opt_treatment)

        # FIXME: this is a temporary fix for non-float values and other named params designated to hold out from aggregation.
        # Needs updated when we have proper collab-side state saving.
        self._remove_and_save_holdout_tensors(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))

    def _remove_and_save_holdout_tensors(self, tensor_dict):
        """Removes tensors from the tensor dictionary

        Takes the dictionary of tensors and removes the holdout_tensors.

        Args:
            tensor_dict: Dictionary of tensors

        Returns:
            Shared tensor dictionary

        """
        shared_tensors, self.holdout_tensors = split_tensor_dict_for_holdouts(self.logger, tensor_dict, **self.tensor_dict_split_fn_kwargs)
        if self.holdout_tensors != {}:
            self.logger.debug("{} removed {} from tensor_dict".format(self, list(self.holdout_tensors.keys())))
        return shared_tensors

    def create_message_header(self):
        """Create a message header to send to network

        Returns:
            Message header for network communications

        """
        header = MessageHeader(sender=self.common_name,
                               recipient=self.aggregator_uuid,
                               federation_id=self.federation_uuid,
                               counter=self.counter,
                               single_col_cert_common_name=self.single_col_cert_common_name,
                               model_header=self.model_header)
        return header

    def __repr__(self):
        """Print collaborator and federation names/uuids.
        """
        return 'collaborator {} of federation {}'.format(self.common_name, self.federation_uuid)

    def __str__(self):
        return self.__repr__()

    def validate_header(self, reply):
        """Validate message header from the aggregator.

        Checks the message against the federation certificates to ensure it commons from approved aggregator in current federation.

        Args:
            reply: Message reply from collaborator.

        Returns:
            bool: True if reply is valid for this federation.

        """
        # check message is from my agg to me
        check_equal(reply.header.sender, self.aggregator_uuid, self.logger)
        check_equal(reply.header.recipient, self.common_name, self.logger)

        # check that the federation id matches
        check_equal(reply.header.federation_id, self.federation_uuid, self.logger)

        # check that we agree on single_col_cert_common_name
        check_equal(reply.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

    def run(self):
        """Runs the collaborator code in a loop until federation quits.
        """
        time_to_quit = False
        while True:
            time_to_quit = self.run_to_yield_or_quit()
            if time_to_quit:
                print(self, 'quitting')
                break
            else:
                time.sleep(self.polling_interval)

    def run_to_yield_or_quit(self):
        """Runs the collaborator code in a loop until federation quits.

        Loops indefinitely looking for messages from the federation aggregator.
        It looks for the following network messages:

        .. code-block:: python

           if job is JOB_DOWNLOAD_MODEL:
               self.do_download_model_job()
           elif job is JOB_VALIDATE:
               self.do_validate_job()
           elif job is JOB_TRAIN:
               self.do_train_job()
           elif job is JOB_YIELD:
               return False
           elif job is JOB_QUIT:
               return True

        """

        self.logger.info("Collaborator [%s] connects to federation [%s] and aggegator [%s]." % (self.common_name, self.federation_uuid, self.aggregator_uuid))
        self.logger.debug("The optimizer variable treatment is [%s]." % self.opt_treatment)
        while True:
            # query for job and validate it
            reply = self.channel.RequestJob(JobRequest(header=self.create_message_header()))
            self.validate_header(reply)
            check_type(reply, JobReply, self.logger)
            job = reply.job

            self.logger.debug("%s - Got a job %s" % (self, Job.Name(job)))

            if job is JOB_DOWNLOAD_MODEL:
                self.do_download_model_job(reply.extra_model_info.tensor_names)
            elif job is JOB_UPLOAD_RESULTS:
                self.do_upload_results_job(reply.name)
            elif job is JOB_SLEEP:
                self.polling_interval = reply.seconds
                return False
            elif job is JOB_QUIT:
                return True

    def _with_opt_vars(self):
        """Determines optimizer operation to perform.

        Returns:
           bool: True means *CONTINUE_GLOBAL* method for optimizer.

        """
        if self.opt_treatment in (OptTreatment.CONTINUE_LOCAL, OptTreatment.RESET):
            self.logger.debug("Not share the optimization variables.")
            return False
        elif self.opt_treatment == OptTreatment.CONTINUE_GLOBAL:
            self.logger.debug("Share the optimization variables.")
            return True

    def do_upload_results_job(self, task):
        if task not in self.round_results:
            self.do_task(task)

        # now we can upload the result
        result, weight = self.round_results[task]

        request = ResultsUpload(header=self.create_message_header(),
                                weight=weight,
                                task=task)

        if isinstance(result, np.ndarray):
            request = ResultsUpload(header=self.create_message_header(),
                                    weight=weight,
                                    task=task,
                                    tensor=numpy_array_to_tensor_proto(result, task))
        elif isinstance(result, dict):
            request = ResultsUpload(header=self.create_message_header(),
                                    weight=weight,
                                    task=task,
                                    value_dict=ValueDictionary(dictionary=result))
        else:
            request = ResultsUpload(header=self.create_message_header(),
                                    weight=weight,
                                    task=task,
                                    value=result)

        self._request_with_retries(request, upload=True)

    def do_task(self, task):
        # FIXME: this should really not be hard-coded
        if task == 'shared_model_validation':
            # sanity check that we have not trained
            if self.local_model_has_been_trained:
                raise RuntimeError("Logic error! We should not have trained!")
            self.do_validation_task('shared_model_validation')
        elif task == 'loss':
            # sanity check that we have not trained
            if self.local_model_has_been_trained:
                raise RuntimeError("Logic error! We should not have trained!")
            self.do_train_task()
        elif task == 'local_model_validation':
            # sanity check that we have trained
            if not self.local_model_has_been_trained:
                raise RuntimeError("Logic error! We should have trained!")
            self.do_validation_task('local_model_validation')
        else:
            # if we are here, we reported a loss, then crashed before we uploaded remaining tensors
            # in this case, we are going to retrain
            self.logger.info("Retraining for remaining results.")
            self.do_train_task()

    def do_train_task(self):
        """Train the model.

        This is the code that actual runs the model training on the collaborator.

        """
        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # train the model
        # FIXME: model header "version" needs to be changed to "rounds_trained"
        # FIXME: We assume the models allow training on partial batches.
        # FIXME: Currently, num_batches_per_round overrides epochs per round. Is this the correct behavior?
        if self.num_batches_per_round is not None:
            num_batches = self.num_batches_per_round
        else:
            batches_per_epoch = int(np.ceil(data_size/self.wrapped_model.data.batch_size))
            num_batches = int(np.floor(batches_per_epoch * self.epochs_per_round))

        self.logger.debug("{} Begun training {} batches.".format(self, num_batches))

        train_info = self.wrapped_model.train_batches(num_batches=num_batches)

        self.local_model_has_been_trained = True

        self.logger.debug("{} Completed training {} batches.".format(self, num_batches))

        # allowing extra information regarding training to be logged (for now none of the extra info is sent to the aggregator)
        if isinstance(train_info, dict):
            loss = train_info["loss"]
            self.logger.debug("{} model is returning dictionary of training info: {}".format(self, train_info))
        else:
            loss = train_info

        self.round_results['loss'] = (loss, data_size)

        # get the trained tensor dict and store any designated to be held out from aggregation
        shared_tensors = self._remove_and_save_holdout_tensors(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))

        # store each resulting tensor
        for k, v in shared_tensors.items():
            self.round_results[k] = (v, data_size)

        self.logger.info("{} - Training complete".format(self))

    def do_validation_task(self, result_name):
        """Validate the model (locally)

        Runs the validation of the model on the local dataset.
        """
        self.logger.debug("{} - Beginning {}".format(self, result_name))
        results = self.wrapped_model.validate()
        self.logger.debug("{} - Completed {}".format(self, result_name))
        data_size = self.wrapped_model.get_validation_data_size()

        self.round_results[result_name] = (results, data_size)

    def _request_with_retries(self, request, upload=False):
        # FIXME: this needs to be a more robust response. The aggregator should actually have sent an error code, rather than an unhandled exception
        # an exception can happen in cases where we simply need to retry
        for i in range(self.num_retries):
            try:
                if upload:
                    reply = self.channel.UploadResults(request)
                else:
                    reply = self.channel.DownloadTensor(request)
                break
            except Exception as e:
                self.logger.exception(repr(e))
                # if final retry, raise exception
                if i + 1 == self.num_retries:
                    raise e
                else:
                    self.logger.warning("Retrying {}. Try {} of {}".format(request.__class__.__name__, i+1, self.num_retries))
        
        self.validate_header(reply)
        return reply

    def do_download_model_job(self, tensor_names):
        """Download model operation

        Asks the aggregator for the latest model to download and downloads it.

        """
        # time the download
        download_start = time.time()

        # set to the first header we receive
        new_model_header = None

        # download all of the tensors in the list
        downloaded_tensors = {}
        for tensor_name in tensor_names:
            request = TensorDownloadRequest(header=self.create_message_header(), tensor_name=tensor_name)
            global_tensor = self._request_with_retries(request, upload=False)

            # if this is our first tensor downloaded, we set first version
            if new_model_header is None:
                new_model_header = global_tensor.header.model_header
            # otherwise, if the tensor versions have changed, we need to exit and request new job
            elif new_model_header.version != global_tensor.header.model_header.version:
                self.logger("Tensor versions have changed. Canceling Download")
                return

            # ensure names match
            check_equal(tensor_name, global_tensor.tensor.name, self.logger)

            downloaded_tensors[tensor_name] = global_tensor.tensor

        # now we have downloaded all of the tensors
        self.logger.info("{} took {} seconds to download the model".format(self, round(time.time() - download_start, 3)))

        # check if our version has been reverted, possibly due to an aggregator reset
        version_reverted = self.model_header.version > new_model_header.version

        # extract all of our tensors
        agg_tensor_dict = {k: tensor_proto_to_numpy_array(downloaded_tensors[k]) for k, v in downloaded_tensors.items()}

        # set our model header
        self.model_header = new_model_header

        # restore any tensors held out from aggregation
        tensor_dict = {**agg_tensor_dict, **self.holdout_tensors}

        if self.opt_treatment == OptTreatment.CONTINUE_GLOBAL:
            with_opt_vars = True
        else:
            with_opt_vars = False

        # Ensuring proper initialization regardless of model state. Initial global models
        # do not contain optimizer state, and so cannot be used to reset the optimizer params.
        # Additionally, in any mode other than continue global, if we received an older model, we need to
        # reset our optimizer parameters
        if self.model_header.version == 0 or \
           (self.opt_treatment != OptTreatment.CONTINUE_GLOBAL and version_reverted):
            with_opt_vars = False
            self.logger.info("Resetting optimizer vars")
            self.wrapped_model.reset_opt_vars()

        self.wrapped_model.set_tensor_dict(tensor_dict, with_opt_vars=with_opt_vars)
        self.logger.debug("Loaded the model.")

        # FIXME: for the CONTINUE_LOCAL treatment, we need to store the status in case of a crash.
        if self.opt_treatment == OptTreatment.RESET:
            try:
                self.wrapped_model.reset_opt_vars()
            except:
                self.logger.exception("Failed to reset the optimization variables.")
                raise
            else:
                self.logger.debug("Reset the optimization variables.")

        # finally, we need to reset our trained and metrics states
        self.local_model_has_been_trained = False
        self.round_results = {}

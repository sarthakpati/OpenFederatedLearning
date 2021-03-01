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
from logging import getLogger

import grpc

from ...proto import datastream_to_proto, proto_to_datastream
from ...proto.collaborator_aggregator_interface_pb2_grpc import AggregatorStub
from ...proto.collaborator_aggregator_interface_pb2 import GlobalTensor


class ConstantBackoff:
    """Constant Backoff policy."""

    def __init__(self, logger, uri, reconnect_interval=10):
        """Initialize Constant Backoff."""
        self.reconnect_interval = reconnect_interval
        self.logger = logger
        self.uri = uri

    def sleep(self):
        """Sleep for specified interval."""
        self.logger.info(f'Attempting to connect to aggregator at {self.uri}')
        time.sleep(self.reconnect_interval)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor
):
    """Retry gRPC connection on failure."""

    def __init__(
        self,
        sleeping_policy,
        status_for_retry=None,
    ):
        """Initialize function for gRPC retry."""
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        """Intercept the call to the gRPC server."""
        while True:
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # If status code is not in retryable status codes
                if (
                    self.status_for_retry
                    and response.code() not in self.status_for_retry
                ):
                    return response

                self.sleeping_policy.sleep()
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Wrap intercept call for unary->unary RPC."""
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        """Wrap intercept call for stream->unary RPC."""
        return self._intercept_call(continuation, client_call_details, request_iterator)


class CollaboratorGRPCClient:
    """Collaboration over gRPC-TLS."""

    def __init__(self,
                 agg_addr,
                 agg_port,
                 disable_tls,
                 disable_client_auth,
                 ca,
                 certificate,
                 private_key,
                 aggregator_uuid=None,
                 federation_uuid=None,
                 single_col_cert_common_name=None,
                 **kwargs):
        """Initialize."""
        self.uri = f'{agg_addr}:{agg_port}'
        self.disable_tls = disable_tls
        self.disable_client_auth = disable_client_auth
        self.ca = ca
        self.certificate = certificate
        self.private_key = private_key

        self.channel_options = [
            ('grpc.max_metadata_size', 32 * 1024 * 1024),
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024)
        ]

        self.logger = getLogger(__name__)

        if self.disable_tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.ca,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        self.header = None
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.single_col_cert_common_name = single_col_cert_common_name

        # Adding an interceptor for RPC Errors
        self.interceptors = (
            RetryOnRpcErrorClientInterceptor(
                sleeping_policy=ConstantBackoff(
                    logger=self.logger,
                    reconnect_interval=int(kwargs.get('client_reconnect_interval', 1)),
                    uri=self.uri),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
        )
        self.stub = AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    def create_insecure_channel(self, uri):
        """
        Set an insecure gRPC channel (i.e. no TLS) if desired.
        Warns user that this is not recommended.
        Args:
            uri: The uniform resource identifier fo the insecure channel
        Returns:
            An insecure gRPC channel object
        """
        self.logger.warn(
            "gRPC is running on insecure channel with TLS disabled.")

        return grpc.insecure_channel(uri, options=self.channel_options)

    def create_tls_channel(self, uri, ca, disable_client_auth,
                           certificate, private_key):
        """
        Set an secure gRPC channel (i.e. TLS).
        Args:
            uri: The uniform resource identifier fo the insecure channel
            ca: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side
             authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator
             (signed by the certificate authority)
        Returns:
            An insecure gRPC channel object
        """
        with open(ca, 'rb') as f:
            root_certificates = f.read()

        if disable_client_auth:
            self.logger.warn('Client-side authentication is disabled.')
            private_key = None
            client_cert = None
        else:
            with open(private_key, 'rb') as f:
                private_key = f.read()
            with open(certificate, 'rb') as f:
                client_cert = f.read()

        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=client_cert
        )

        return grpc.secure_channel(
            uri, credentials, options=self.channel_options)

    def _set_header(self, collaborator_name):
        self.header = self.header = MessageHeader(
            sender=collaborator_name,
            receiver=self.aggregator_uuid,
            federation_uuid=self.federation_uuid,
            single_col_cert_common_name=self.single_col_cert_common_name or ''
        )

    def validate_response(self, reply, collaborator_name):
        """Validate the aggregator response."""
        # check that the message was intended to go to this collaborator
        check_equal(reply.header.receiver, collaborator_name, self.logger)
        check_equal(reply.header.sender, self.aggregator_uuid, self.logger)

        # check that federation id matches
        check_equal(
            reply.header.federation_uuid,
            self.federation_uuid,
            self.logger
        )

        # check that there is aggrement on the single_col_cert_common_name
        check_equal(
            reply.header.single_col_cert_common_name,
            self.single_col_cert_common_name or '',
            self.logger
        )

    def disconnect(self):
        """Close the gRPC channel."""
        self.logger.debug(f'Disconnecting from gRPC server at {self.uri}')
        self.channel.close()

    def reconnect(self):
        """Create a new channel with the gRPC server."""
        # channel.close() is idempotent. Call again here in case it wasn't issued previously
        self.disconnect()

        if self.disable_tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.ca,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        self.logger.debug(f'Connecting to gRPC at {self.uri}')

        self.stub = AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    def _atomic_connection(func):
        def wrapper(self, *args, **kwargs):
            self.reconnect()
            response = func(self, *args, **kwargs)
            self.disconnect()
            return response
        return wrapper

    @_atomic_connection
    def RequestJob(self, message):
        """Request Job

        Args:
            message: Message sent to the collaborator
        """
        return self.stub.RequestJob(message)

    @_atomic_connection
    def DownloadTensor(self, message):
        """Download Tensor

        Args:
            message: Message sent to the collaborator
        """
        stream = self.stub.DownloadTensor(message)
        # turn datastream into global model update
        return datastream_to_proto(GlobalTensor(), stream)

    @_atomic_connection
    def UploadResults(self, message):
        """Upload Local Model Update

        Args:
            message: Message sent to the collaborator
        """
        # turn local model update into datastream
        stream = []
        stream += proto_to_datastream(message, self.logger)
        return self.stub.UploadResults(iter(stream))

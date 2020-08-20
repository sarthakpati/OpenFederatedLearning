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

import grpc
import logging

from ...proto import datastream_to_proto, proto_to_datastream
from ...proto.collaborator_aggregator_interface_pb2_grpc import AggregatorStub
from ...proto.collaborator_aggregator_interface_pb2 import GlobalModelUpdate

class CollaboratorGRPCClient():
    """Collaboration over gRPC-TLS."""
    def __init__(self, agg_addr, agg_port, disable_tls, ca, disable_client_auth, certificate, private_key, **kwargs):
        self.logger = logging.getLogger(__name__)
        uri = "{addr:s}:{port:d}".format(addr=agg_addr, port=agg_port)

        self.channel_options=[('grpc.max_metadata_size', 32 * 1024 * 1024),
                              ('grpc.max_send_message_length', 128 * 1024 * 1024),
                              ('grpc.max_receive_message_length', 128 * 1024 * 1024)]

        if disable_tls:
            self.channel = self.create_insecure_channel(uri)
        else:
            self.channel = self.create_tls_channel(uri, ca, disable_client_auth, certificate, private_key)

        self.logger.debug("Connecting to gRPC at %s" % uri)
        self.stub = AggregatorStub(self.channel)

    def create_insecure_channel(self, uri):
        """Sets an insecure gRPC channel (i.e. no TLS) if desired (warns user that this is not recommended)

        Args:
            uri: The uniform resource identifier fo the insecure channel

        Returns:
            An insecure gRPC channel object

        """
        self.logger.warn("gRPC is running on insecure channel with TLS disabled.")
        return grpc.insecure_channel(uri, options=self.channel_options)

    def create_tls_channel(self, uri, ca, disable_client_auth, certificate, private_key):
        """Sets an secure gRPC channel (i.e. TLS)

        Args:
            uri: The uniform resource identifier fo the insecure channel
            ca: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator (signed by the certificate authority)

        Returns:
            An insecure gRPC channel object
        """

        with open(ca, 'rb') as f:
            root_certificates = f.read()

        if disable_client_auth:
            self.logger.warn("Client-side authentication is disabled.")
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
        return grpc.secure_channel(uri, credentials, options=self.channel_options)

    def RequestJob(self, message):
        """Request Job

        Args:
            message: Message sent to the collaborator
        """
        return self.stub.RequestJob(message)

    def DownloadModel(self, message):
        """Download Model

        Args:
            message: Message sent to the collaborator
        """
        stream = self.stub.DownloadModel(message)
        # turn datastream into global model update
        return datastream_to_proto(GlobalModelUpdate(), stream)

    def UploadLocalModelUpdate(self, message):
        """Upload Local Model Update

        Args:
            message: Message sent to the collaborator
        """
        # turn local model update into datastream
        stream = []
        stream += proto_to_datastream(message, self.logger)
        return self.stub.UploadLocalModelUpdate(iter(stream))

    def UploadLocalMetricsUpdate(self, message):
        """Upload Local Metrics Update

        Args:
            message: Message sent to the collaborator
        """
        return self.stub.UploadLocalMetricsUpdate(message)

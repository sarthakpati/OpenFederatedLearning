# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import openfl.proto.collaborator_aggregator_interface_pb2 as collaborator__aggregator__interface__pb2


class AggregatorStub(object):
    """we start with everything as "required" while developing / debugging. This forces correctness better.
    FIXME: move to "optional" once development is complete

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RequestJob = channel.unary_unary(
                '/openfl_proto.Aggregator/RequestJob',
                request_serializer=collaborator__aggregator__interface__pb2.JobRequest.SerializeToString,
                response_deserializer=collaborator__aggregator__interface__pb2.JobReply.FromString,
                )
        self.DownloadTensor = channel.unary_stream(
                '/openfl_proto.Aggregator/DownloadTensor',
                request_serializer=collaborator__aggregator__interface__pb2.TensorDownloadRequest.SerializeToString,
                response_deserializer=collaborator__aggregator__interface__pb2.DataStream.FromString,
                )
        self.UploadResults = channel.stream_unary(
                '/openfl_proto.Aggregator/UploadResults',
                request_serializer=collaborator__aggregator__interface__pb2.DataStream.SerializeToString,
                response_deserializer=collaborator__aggregator__interface__pb2.ResultsAck.FromString,
                )
        self.DownloadRoundSummary = channel.unary_unary(
                '/openfl_proto.Aggregator/DownloadRoundSummary',
                request_serializer=collaborator__aggregator__interface__pb2.RoundSummaryDownloadRequest.SerializeToString,
                response_deserializer=collaborator__aggregator__interface__pb2.RoundSummary.FromString,
                )


class AggregatorServicer(object):
    """we start with everything as "required" while developing / debugging. This forces correctness better.
    FIXME: move to "optional" once development is complete

    """

    def RequestJob(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DownloadTensor(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UploadResults(self, request_iterator, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DownloadRoundSummary(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AggregatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RequestJob': grpc.unary_unary_rpc_method_handler(
                    servicer.RequestJob,
                    request_deserializer=collaborator__aggregator__interface__pb2.JobRequest.FromString,
                    response_serializer=collaborator__aggregator__interface__pb2.JobReply.SerializeToString,
            ),
            'DownloadTensor': grpc.unary_stream_rpc_method_handler(
                    servicer.DownloadTensor,
                    request_deserializer=collaborator__aggregator__interface__pb2.TensorDownloadRequest.FromString,
                    response_serializer=collaborator__aggregator__interface__pb2.DataStream.SerializeToString,
            ),
            'UploadResults': grpc.stream_unary_rpc_method_handler(
                    servicer.UploadResults,
                    request_deserializer=collaborator__aggregator__interface__pb2.DataStream.FromString,
                    response_serializer=collaborator__aggregator__interface__pb2.ResultsAck.SerializeToString,
            ),
            'DownloadRoundSummary': grpc.unary_unary_rpc_method_handler(
                    servicer.DownloadRoundSummary,
                    request_deserializer=collaborator__aggregator__interface__pb2.RoundSummaryDownloadRequest.FromString,
                    response_serializer=collaborator__aggregator__interface__pb2.RoundSummary.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'openfl_proto.Aggregator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Aggregator(object):
    """we start with everything as "required" while developing / debugging. This forces correctness better.
    FIXME: move to "optional" once development is complete

    """

    @staticmethod
    def RequestJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl_proto.Aggregator/RequestJob',
            collaborator__aggregator__interface__pb2.JobRequest.SerializeToString,
            collaborator__aggregator__interface__pb2.JobReply.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DownloadTensor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/openfl_proto.Aggregator/DownloadTensor',
            collaborator__aggregator__interface__pb2.TensorDownloadRequest.SerializeToString,
            collaborator__aggregator__interface__pb2.DataStream.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UploadResults(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/openfl_proto.Aggregator/UploadResults',
            collaborator__aggregator__interface__pb2.DataStream.SerializeToString,
            collaborator__aggregator__interface__pb2.ResultsAck.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DownloadRoundSummary(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/openfl_proto.Aggregator/DownloadRoundSummary',
            collaborator__aggregator__interface__pb2.RoundSummaryDownloadRequest.SerializeToString,
            collaborator__aggregator__interface__pb2.RoundSummary.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

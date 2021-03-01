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

import numpy as np
from openfl.proto.collaborator_aggregator_interface_pb2 import TensorProto, DataStream

def tensor_proto_to_numpy_array(tensor_proto):
    return np.frombuffer(tensor_proto.data_bytes, dtype=np.float32).reshape(tuple(tensor_proto.shape))

def numpy_array_to_tensor_proto(array, name):
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    array_shape = list(array.shape)
    return TensorProto(name=name, data_bytes=array.tobytes(order='C'), shape=array_shape)

def load_proto(fpath, proto_type):
    """Load the protobuf

    Args:
        fpath: The filepath for the protobuf

    Returns:
        protobuf: A protobuf of the model
    """
    with open(fpath, "rb") as f:
        return proto_type().FromString(f.read())

def dump_proto(model_proto, fpath):
    """Dumps the protobuf to a file

    Args:
        model_proto: The protobuf of the model
        fpath: The filename to save the model protobuf

    """
    s = model_proto.SerializeToString()
    with open(fpath, "wb") as f:
        f.write(s)

def datastream_to_proto(proto, stream, logger=None):
    """Converts the datastream to the protobuf

    Args:
        model_proto: The protobuf of the model
        stream: The data stream from the remote connection
        logger: (Optional) The log object

    Returns:
        protobuf: A protobuf of the model
    """
    npbytes = b""
    for chunk in stream:
        npbytes += chunk.npbytes

    if len(npbytes) > 0:
        proto.ParseFromString(npbytes)
        if logger is not None:
            logger.debug("datastream_to_proto parsed a {}.".format(type(proto)))
        return proto
    else:
        raise RuntimeError("Received empty stream message of type {}".format(type(proto)))

def proto_to_datastream(proto, logger, max_buffer_size=(256 * 1024)):
    """Convert the protobuf to the datastream for the remote connection

    Args:
        model_proto: The protobuf of the model
        logger: The log object
        max_buffer_size: The buffer size (Default= 256 * 1024)
    Returns:
        reply: The message for the remote connection.
    """
    npbytes = proto.SerializeToString()
    data_size = len(npbytes)
    buffer_size = data_size if max_buffer_size > data_size else max_buffer_size
    logger.debug("Setting stream chunks with size {} for proto of type {}".format(buffer_size, type(proto)))

    for i in range(0, data_size, buffer_size):
        chunk = npbytes[i : i + buffer_size]
        reply = DataStream(npbytes=chunk, size=len(chunk))
        yield reply

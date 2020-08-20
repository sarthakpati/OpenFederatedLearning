from openfl.tensor_transformation_pipelines import TransformationPipeline, Transformer

import numpy as np

class Float32NumpyArrayToBytes(Transformer):
    """Converts float32 Numpy array to Bytes array
    """
    def __init__(self, **kwargs):
        """Initializer
        """
        pass

    def forward(self, data, **kwargs):
        """Forward pass

        Args:
            data:
            **kwargs: Additional arguments to pass to the function

        Returns:
            data_bytes:
            metadata:
        """
        # TODO: Warn when this casting is being performed.
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        array_shape = data.shape
        metadata = {'int_list': list(array_shape)}
        data_bytes = data.tobytes(order='C')
        return data_bytes, metadata

    def backward(self, data, metadata):
        """Backward pass

        Args:
            data:
            metadata:

        Returns:
            Numpy Array

        """
        array_shape = tuple(metadata['int_list'])
        flat_array = np.frombuffer(data, dtype=np.float32)
        return np.reshape(flat_array, newshape=array_shape, order='C')


class NoCompressionPipeline(TransformationPipeline):
    """The data pipeline without any compression
    """

    def __init__(self, **kwargs):
        """Initializer
        """
        super(NoCompressionPipeline, self).__init__(transformers=[Float32NumpyArrayToBytes()], **kwargs)

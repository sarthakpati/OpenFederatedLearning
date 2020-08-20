
class Transformer(object):
    """Data transformation class
    """

    def __init__(self):
        """Initializer
        """
        raise NotImplementedError


    def forward(self, data, **kwargs):
        """Forward pass data transformation

        Implement the data transformation.

        Args:
            data:
            **kwargs: Additional parameters to pass to the function

        Returns:
            transformed_data
            metadata
        """
        raise NotImplementedError

    def backward(self, data, metadata, **kwargs):
        """Backward pass data transformation

        Implement the data transformation needed when going the opposite
        direction to the forward method.

        Args:
            data:
            metadata:
            **kwargs: Additional parameters to pass to the function

        Returns:
            transformed_data
        """
        raise NotImplementedError


class TransformationPipeline(object):
    """Data Transformer Pipeline Class

    A sequential pipeline to transform (e.x. compress) data (e.x. layer of model_weights) as well as return
    metadata (if needed) for the reconstruction process carried out by the backward method.
    """

    def __init__(self, transformers, **kwargs):
        """Initializer

        Args:
            transformers:
            **kwargs: Additional parameters to pass to the function
        """
        self.transformers = transformers

    def forward(self, data, **kwargs):
        """Forward pass of pipeline data transformer

        Args:
            data: Data to transform
            **kwargs: Additional parameters to pass to the function

        Returns:
            data:
            transformer_metadata:

        """
        transformer_metadata = []

        # dataformat::numpy::float.32
        # model proto:: a collection of tensor_dict proto
        # protobuff::-> a layer of weights
        # input::tensor_dict:{"layer1":np.array(float32, [128,3,28,28]), "layer2": np.array()}
        # output::meta data::numpy::int array
        # (data, transformer_metadata)::(float32, dictionary of key+float32 vlues)
        # input:: numpy_data (float32)
        # input:: (data(bytes), transformer_metadata_list::a list of dictionary from int to float)

        metadata_list = []
        data = data.copy()
        for transformer in self.transformers:
            data, metadata = transformer.forward(data=data, **kwargs)
            transformer_metadata.append(metadata)
        return data, transformer_metadata

    def backward(self, data, transformer_metadata, **kwargs):
        """Backward pass of pipeline data transformer

        Args:
            data: Data to transform
            transformer_metadata:
            **kwargs: Additional parameters to pass to the function

        Returns:
            data:

        """

        for transformer in self.transformers[::-1]:
            data = transformer.backward(data=data, metadata=transformer_metadata.pop(), **kwargs)
        return data

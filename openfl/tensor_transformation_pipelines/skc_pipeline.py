import numpy as np
import gzip
import copy
# FIXME: We have removed sklearn temporarily due to a CVE issue 
#from sklearn import cluster

from openfl.tensor_transformation_pipelines import TransformationPipeline, Transformer

class SparsityTransformer(Transformer):
    """ A transformer class to sparsify input data.
    """

    def __init__(self, p=0.01):
        """Initializer

        Args:
            p (float): sparsity ratio (Default=0.01)
        """
        self.p = p
        return

    def forward(self, data, **kwargs):
        """ Sparsify data and pass over only non-sparsified elements by reducing the array size.

        Args:
            data: an numpy array from the model tensor_dict.

        Returns:
            condensed_data: an numpy array being sparsified.
            metadata: dictionary to store a list of meta information.
        """
        self.p = 1
        metadata = {}
        metadata['int_list'] = list(data.shape)
        # sparsification
        data = data.astype(np.float32)
        flatten_data = data.flatten()
        n_elements = flatten_data.shape[0]
        k_op = int(np.ceil(n_elements*self.p))
        topk, topk_indices = self._topk_func(flatten_data, k_op)
        #
        condensed_data = topk
        sparse_data = np.zeros(flatten_data.shape)
        sparse_data[topk_indices] = topk
        nonzero_element_bool_indices = sparse_data != 0.0
        metadata['bool_list'] = list(nonzero_element_bool_indices)
        return condensed_data, metadata

    def backward(self, data, metadata, **kwargs):
        """ Recover data array with the right shape and numerical type.

        Args:
            data: an numpy array with non-zero values.
            metadata: dictionary to contain information for recovering back to original data array.

        Returns:
            recovered_data: an numpy array with original shape.
        """
        data = data.astype(np.float32)
        data_shape = metadata['int_list']
        nonzero_element_bool_indices = list(metadata['bool_list'])
        recovered_data = np.zeros(data_shape).reshape(-1).astype(np.float32)
        recovered_data[nonzero_element_bool_indices] = data
        recovered_data = recovered_data.reshape(data_shape)
        return recovered_data


    def _topk_func(self, x, k):
        """ Select top k values.

        Args:
            x: an numpy array to be sorted out for top-k components.
            k: k most maximum values.

        Returns:
            topk_mag: components with top-k values.
            indices: indices of the top-k components.
        """
        # quick sort as default on magnitude
        idx = np.argsort(np.abs(x))
        # sorted order, the right most is the largest magnitude
        length = x.shape[0]
        start_idx = length - k
        # get the top k magnitude
        topk_mag = np.asarray(x[idx[start_idx:]])
        indices = np.asarray(idx[start_idx:])
        if min(topk_mag)-0 < 10e-8:# avoid zeros
            topk_mag = topk_mag + 10e-8
        return topk_mag, indices

class KmeansTransformer(Transformer):
    """ A transformer class to quantize input data.
    """
    def __init__(self, n_cluster=6):
        self.n_cluster = n_cluster
        return

    def forward(self, data, **kwargs):
        """ Quantize data into n_cluster levels of values.

        Args:
            data: an flattened numpy array.

        Returns:
            int_data: an numpy array being quantized.
            metadata: dictionary to store a list of meta information.
        """
        # FIXME: remove this when sklearn CVE is resolved
        raise NotImplementedError("Clustering has been temporarily disabled due to an issue of SKLearn CVE. We will enable it when the CVE is fixed.") 
        """
        # clustering
        data = data.reshape((-1,1))
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        k_means.fit(data)
        quantized_values = k_means.cluster_centers_.squeeze()
        indices = k_means.labels_
        quant_array = np.choose(indices, quantized_values)
        int_array, int2float_map = self._float_to_int(quant_array)
        metadata = {}
        metadata['int_to_float']  = int2float_map
        int_array = int_array.reshape(-1)
        return int_array, metadata
        """

    def backward(self, data, metadata, **kwargs):
        """ Recover data array back to the original numerical type.

        Args:
            data: an numpy array with non-zero values
            metadata: dictionary to contain information for recovering back to original data array

        Returns:
            data: an numpy array with original numerical type
        """
        # convert back to float
        data = copy.deepcopy(data)
        int2float_map = metadata['int_to_float']
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        return data

    def _float_to_int(self, np_array):
        """ Creating look-up table for conversion between floating and integer types.

        Args:
            np_array

        Returns:
            int_array, int_to_float_map
        """
        flatten_array = np_array.reshape(-1)
        unique_value_array = np.unique(flatten_array)
        int_array = np.zeros(flatten_array.shape, dtype=np.int)
        int_to_float_map = {}
        float_to_int_map = {}
        # create table
        for idx, u_value in enumerate(unique_value_array):
            int_to_float_map.update({idx: u_value})
            float_to_int_map.update({u_value: idx})
            # assign to the integer array
            indices = np.where(flatten_array==u_value)
            int_array[indices] = idx
        int_array = int_array.reshape(np_array.shape)
        return int_array, int_to_float_map

class GZIPTransformer(Transformer):
    """ A transformer class to losslessly compress data.
    """
    def __init__(self):
        return

    def forward(self, data, **kwargs):
        """ Compress data into bytes.

        Args:
            data: an numpy array with non-zero values
        """
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes_ = gzip.compress(bytes_)
        metadata = {}
        return compressed_bytes_, metadata

    def backward(self, data, metadata, **kwargs):
        """ Decompress data into numpy of float32.

        Args:
            data: an numpy array with non-zero values
            metadata: dictionary to contain information for recovering back to original data array

        Returns:
            data:
        """
        decompressed_bytes_ = gzip.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data

class SKCPipeline(TransformationPipeline):
    """ A pipeline class to compress data lossly using sparsity and k-means methods.
    """

    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        """ Initializing a pipeline of transformers.

        Args:
            p_sparsity (float): Sparsity factor (Default=0.01)
            n_cluster (int): Number of K-Means clusters (Default=6)

        Returns:
            Data compression transformer pipeline object
        """
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [SparsityTransformer(self.p), KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super(SKCPipeline, self).__init__(transformers=transformers, **kwargs)

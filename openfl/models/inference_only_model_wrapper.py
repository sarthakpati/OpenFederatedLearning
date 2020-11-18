# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


from openfl.models import FLModel
from openfl.flplan import init_object

class InferenceOnlyModelWrapper(FLModel):
    """Model class wrapper for Federated Learning to enable inferencing on the model using 
    run_inference_from_flplan with minimal requirements on the base_model. Those requirements
    on the base model are as follows:

    A python class providing access to your model, with two required instance methods. 

    Please populate argument defaults where appropriate to allow for the optimal 
    configuration, and document to allow users to understand how to customize.

    Particularly relevant, will be that your model infer_volume method processes images according 
    to our assumption on input and output shapes (see below).

    The required instance methods are as follows:


    def __init__(self, *args, **kwargs):
        '''
        Instantiates a properly configured model object, including population
        of all model weights from a model serialization file.
        
        Args: ...
            
        Kwargs: ...
        
        Returns:
            None
        '''
        raise NotImplementedError()    
        

        
    def infer_volume(self, X):
        '''
        Perform model inference on a volume of data.
        
        Args:
            X (numpy array): Input volume to perform inference on, containing channels for
                            all scan modalities required for the model task. 
                            The expected input shape can be either:
                            (num_channels is the number of scanning modalities used by the model)
                            - channels first -
                            (num_samples, num_channels,coronal_axis_dim, sagittal_axis_dim, transversal_axis_dim)
                            - or channels last -
                            (num_samples,coronal_axis_dim, sagittal_axis_dim, transversal_axis_dim, num_channels)
        
        Returns:
            (numpy array): Model output for the input volume.
                        The output shape is as follows:
                        (num_samples, coronal_axis_dim, sagittal_axis_dim, transversal_axis_dim)
        '''
        
        raise NotImplementedError()
            
        
        """

    def __init__(self, data, base_model):
        """Initializer

        Args:
            data (child class of fldata): Object to provide inference data loader.
            base_model: Base model satisfying requirements above
        """

        super(InferenceOnlyFLModelWrapper, self).__init__(data=data)

        self.infer_batch = base_model.infer_volume





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

# from openfl.data import load_from_NIfTY
from openfl.data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory


class PyTorchBratsInMemory(PyTorchFLDataInMemory):
    """Loads the BraTS dataset into memory for the PyTorch examples
    """

    def __init__(self,
                 data_path,
                 batch_size,
                 percent_train=0.8,
                 pre_split_shuffle=True,
                 channels_last=False,
                 **kwargs):

        """Initializer

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before performing the train/validate split (Default=True)
            channels_last (bool): True= data is channels last (NHWDC) (Default=False)
            **kwargs: Additional arguments, passed to super init and load_from_NIfTI to function

        Returns:
            Data loader with BraTS data
        """

        super().__init__(batch_size, **kwargs)

        # X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path,
        #                                                 percent_train=percent_train,
        #                                                 shuffle=pre_split_shuffle,
        #                                                 channels_last=channels_last,
        #                                                 **kwargs)
        # self.training_data_size = len(X_train)
        # self.validation_data_size = len(X_val)
        # self.train_loader = self.create_loader(X=X_train, y=y_train, shuffle=True)
        # self.val_loader = self.create_loader(X=X_val, y=y_val, shuffle=False)

        



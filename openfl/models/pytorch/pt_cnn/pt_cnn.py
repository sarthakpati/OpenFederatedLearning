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


"""
You may copy this file as the starting point of your own model.
"""
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from openfl.models.pytorch import PyTorchFLModel

def cross_entropy(output, target):
    """Binary cross-entropy metric

    Args:
        output: The mode prediction
        target: The target (ground truth label)

    Returns:
        Binary cross-entropy with logits

    """
    return F.binary_cross_entropy_with_logits(input=output, target=target)



class PyTorchCNN(PyTorchFLModel):
    """Simple CNN for classification.
    """

    def __init__(self, data, device='cpu', **kwargs):
        """Initializer

        Args:
            data: The data loader class
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(data=data, device=device, **kwargs)

        self.num_classes = self.data.num_classes
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.loss_fn = cross_entropy

    def _init_optimizer(self):
        """Initializer the optimizer
        """
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def init_network(self,
                     device,
                     print_model=True,
                     pool_sqrkernel_size=2,
                     conv_sqrkernel_size=5,
                     conv1_channels_out=20,
                     conv2_channels_out=50,
                     fc2_insize = 500,
                     **kwargs):
        """Create the network (model)

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            pool_sqrkernel_size (int): Max pooling kernel size (Default=2), assumes square 2x2
            conv_sqrkernel_size (int): Convolutional filter size (Default=5), assumes square 5x5
            conv1_channels_out (int): Number of filters in first convolutional layer (Default=20)
            conv2_channels_out: Number of filters in second convolutional layer (Default=50)
            fc2_insize (int): Number of neurons in the fully-connected layer (Default = 500)
            **kwargs: Additional arguments to pass to the function

        """
        """
        FIXME: We are tracking only side lengths (rather than length and width) as we are assuming square
        shapes for feature and kernels.
        In order that all of the input and activation components are used (not cut off), we rely on a criterion:
        appropriate integers are divisible so that all casting to int perfomed below does no rounding
        (i.e. all int casting simply converts a float with '0' in the decimal part to an int.)

        (Note this criterion held for the original input sizes considered for this model: 28x28 and 32x32
        when used with the default values above)

        """
        self.pool_sqrkernel_size = pool_sqrkernel_size
        channel = self.data.get_feature_shape()[0]# (channel, dim1, dim2)
        self.conv1 = nn.Conv2d(channel, conv1_channels_out, conv_sqrkernel_size, 1)

        # perform some calculations to track the size of the single channel activations
        # channels are first for pytorch
        conv1_sqrsize_in = self.feature_shape[-1]
        conv1_sqrsize_out = conv1_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv1 out
        # (note dependence on 'forward' function below)
        conv2_sqrsize_in = int(conv1_sqrsize_out/pool_sqrkernel_size)

        self.conv2 = nn.Conv2d(conv1_channels_out, conv2_channels_out, conv_sqrkernel_size, 1)

        # more tracking of single channel activation size
        conv2_sqrsize_out = conv2_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv2 out
        # (note dependence on 'forward' function below)
        l = int(conv2_sqrsize_out/pool_sqrkernel_size)
        self.fc1_insize = l*l*conv2_channels_out
        self.fc1 = nn.Linear(self.fc1_insize, fc2_insize)
        self.fc2 = nn.Linear(fc2_insize, self.num_classes)
        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        """Forward pass of the model

        Args:
            x: Data input to the model for the forward pass
        """

        x = F.relu(self.conv1(x))
        pl = self.pool_sqrkernel_size
        x = F.max_pool2d(x, pl, pl)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, pl, pl)
        x = x.view(-1, self.fc1_insize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def validate(self, use_tqdm=False):
        """Validate

        Run validation of the model on the local data.

        Args:
            use_tqdm (bool): Use tqdm to print a progress bar (Default=True)

        """
        self.eval()
        val_score = 0
        total_samples = 0

        loader = self.data.get_val_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                output = self(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                target_categorical = target.argmax(dim=1, keepdim=True)
                val_score += pred.eq(target_categorical).sum().cpu().numpy()

        return val_score / total_samples

    def train_batches(self, num_batches, use_tqdm=False):
        """Train batches

        Train the model on the requested number of batches.

        Args:
            num_batches: The number of batches to train on before returning
            use_tqdm (bool): Use tqdm to print a progress bar (Default=True)

        Returns:
            loss metric
        """

        # set to "training" mode
        self.train()

        losses = []

        loader = self.data.get_train_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="train batches")

        batch_num = 0

        while batch_num < num_batches:
            # shuffling occurs every time this loader is used as an interator
            for data, target in loader:
                if batch_num >= num_batches:
                    break
                else:
                    data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                    self.optimizer.zero_grad()
                    output = self(data)
                    loss = self.loss_fn(output=output, target=target)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.detach().cpu().numpy())

                    batch_num += 1

        return np.mean(losses)

    def reset_opt_vars(self):
        """Reset optimizer variables

        Resets the optimizer state variables.

        """
        self._init_optimizer()

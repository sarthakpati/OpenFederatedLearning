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

from functools import partial
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from openfl.models.pytorch import PyTorchFLModel

# FIXME: move to some custom losses.py file?
def dice_coef(pred, target, smoothing=1.0):
    """Dice Coefficient

    Calculates the Soresen Dice cofficient

    Args:
        pred: Array for the model predictions
        target: Array for the model target (ground truth labels)
        smoothing (float): Laplace smoothing factor (Default = 1.0)

    """
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3))

    return ((2 * intersection + smoothing) / (union + smoothing)).mean()


def dice_coef_loss(pred, target, smoothing=1.0):
    """Dice coefficient loss

    This is actually -log Dice

    Args:
        pred: Array for the model predictions
        target: Array for the model target (ground truth labels)
        smoothing (float): Laplace smoothing factor (Default = 1.0)
    """
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3))

    term1 = -torch.log(2 * intersection + smoothing)
    term2 = torch.log(union + smoothing)

    return term1.mean() + term2.mean()


class PyTorch2DUNet(PyTorchFLModel):
    """PyTorch 2D U-Net model class for Federated Learning
    """

    def __init__(self, data, device='cpu', optimizer='SGD', batch_norm=True, **kwargs):
        """Initializer

        Args:
            data: The data loader class
            device: The hardware device to use for training (Default = "cpu")
            optimizer: The deep learning optimizer (Default="SGD", stochastic gradient descent)
            batch_norm (bool): True uses the batch normalization layer (Default=True)
            **kwargs: Additional arguments to pass to the function

        """

        super().__init__(data=data, device=device, **kwargs)

        self.batch_norm = batch_norm
        self.init_network(device=self.device, **kwargs)
        self.init_optimizer(optimizer)
        self.loss_fn = partial(dice_coef_loss, smoothing=1.0)

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

        gen = self.data.get_train_loader()
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="training for this round")

        batch_num = 0

        while batch_num < num_batches:
            # shuffling happens every time gen is used as an iterator
            for (data, target) in gen:
                if batch_num >= num_batches:
                    break
                else:
                    if isinstance(data, np.ndarray):
                            data = torch.Tensor(data)
                    if isinstance(target, np.ndarray):
                        target = torch.Tensor(data)
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self(data)
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.detach().cpu().numpy())

                    batch_num += 1

        return np.mean(losses)

    def validate(self, use_tqdm=False):
        """Validate

        Run validation of the model on the local data.

        Args:
            use_tqdm (bool): Use tqdm to print a progress bar (Default=True)

        """

        self.eval()
        val_score = 0
        total_samples = 0

        gen = self.data.get_val_loader()
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="validate")

        with torch.no_grad():
            for data, target in gen:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                val_score += dice_coef(output, target).cpu().numpy() * samples
        return val_score / total_samples

    def reset_opt_vars(self):
        """Reset optimizer variables

        Resets the optimizer state variables.

        """
        self.init_optimizer(self.optimizer.__class__.__name__)

    def init_network(self,
                     device,
                     print_model=True,
                     dropout_layers=[2, 3],
                     initial_channels=1,
                     depth_per_side=5,
                     initial_filters=32,
                     **kwargs):
        """Initialize the Model (Network)

        Creates the 2D U-Net model in PyTorch.

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            dropout_layers (list): (Default=[2, 3])
            initial_channels (int): Number of channels in the input layer (Default=1)
            depth_per_side (int): Number of max pooling layers in the encoder/decoder (Default=5)
            initial_filters (int): Number of filters in the initial convolutional layer (Default=32)
            **kwargs: Additional arguments to pass to the function
        """


        f = initial_filters
        if dropout_layers is None:
            self.dropout_layers = []
        else:
            self.dropout_layers = dropout_layers

        # store our depth for our forward function
        self.depth_per_side = 5

        # parameter-less layers
        self.dropout = nn.Dropout(p=0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # initial down layers
        conv_down_a = [nn.Conv2d(initial_channels, f, 3, padding=1)]
        conv_down_b = [nn.Conv2d(f, f, 3, padding=1)]
        if self.batch_norm:
            batch_norms = [nn.BatchNorm2d(f)]

        # rest of the layers going down
        for i in range(1, depth_per_side):
            f *= 2
            conv_down_a.append(nn.Conv2d(f // 2, f, 3, padding=1))
            conv_down_b.append(nn.Conv2d(f, f, 3, padding=1))
            if self.batch_norm:
                batch_norms.append(nn.BatchNorm2d(f))

        # going up, do all but the last layer
        conv_up_a = []
        conv_up_b = []
        for _ in range(depth_per_side-1):
            f //= 2
            # triple input channels due to skip connections
            conv_up_a.append(nn.Conv2d(f*3, f, 3, padding=1))
            conv_up_b.append(nn.Conv2d(f, f, 3, padding=1))

        # do the last layer
        self.conv_out = nn.Conv2d(f, 1, 1, padding=0)

        # all up/down layers need to to become fields of this object
        for i, (a, b) in enumerate(zip(conv_down_a, conv_down_b)):
            setattr(self, 'conv_down_{}a'.format(i+1), a)
            setattr(self, 'conv_down_{}b'.format(i+1), b)

        # all up/down layers need to to become fields of this object
        for i, (a, b) in enumerate(zip(conv_up_a, conv_up_b)):
            setattr(self, 'conv_up_{}a'.format(i+1), a)
            setattr(self, 'conv_up_{}b'.format(i+1), b)

        if self.batch_norm:
            # all the batch_norm layers need to become fields of this object
            for i, bn in enumerate(batch_norms):
                setattr(self, 'batch_norm_{}'.format(i+1), bn)

        if print_model:
            print(self)

        # send this to the device
        self.to(device)

    def forward(self, x):
        """Forward pass of the model

        Args:
            x: Data input to the model for the forward pass
        """

        # gather up our up and down layer members for easier processing
        conv_down_a = [getattr(self, 'conv_down_{}a'.format(i+1)) for i in range(self.depth_per_side)]
        conv_down_b = [getattr(self, 'conv_down_{}b'.format(i+1)) for i in range(self.depth_per_side)]
        conv_up_a = [getattr(self, 'conv_up_{}a'.format(i+1)) for i in range(self.depth_per_side - 1)]
        conv_up_b = [getattr(self, 'conv_up_{}b'.format(i+1)) for i in range(self.depth_per_side - 1)]

        # if batch_norm, gather up our batch norm layers
        if self.batch_norm:
            batch_norms = [getattr(self, 'batch_norm_{}'.format(i+1)) for i in range(self.depth_per_side)]

        # we concatenate the outputs from the b layers (or the batch norm layers)
        concat_me = []
        pool = x

        # going down, wire each pair and then pool except the last
        for i, (a, b) in enumerate(zip(conv_down_a, conv_down_b)):
            out_down = F.relu(a(pool))
            if i in self.dropout_layers:
                out_down = self.dropout(out_down)
            out_down = F.relu(b(out_down))
            if self.batch_norm:
                out_down = batch_norms[i](out_down)
            # if not the last down b layer, pool it and add it to the concat list
            if b != conv_down_b[-1]:
                concat_me.append(out_down)
                pool = self.maxpool(out_down) # feed the pool into the next layer

        # reverse the concat_me layers
        concat_me = concat_me[::-1]

        # we start going up with the b (not-pooled) from previous layer
        in_up = out_down

        # going up, we need to zip a, b and concat_me
        for a, b, c in zip(conv_up_a, conv_up_b, concat_me):
            up = torch.cat([self.upsample(in_up), c], dim=1)
            up = F.relu(a(up))
            in_up = F.relu(b(up))

        # finally, return the output
        return torch.sigmoid(self.conv_out(in_up))

    def init_optimizer(self, optimizer='SGD'):
        """Initialize the optimizer

        Args:
            optimizer: Type of optimizer (Default="SGD")
        """

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=1e-5, momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=1e-5)
        else:
            raise ValueError('Optimizer: {} is not curently supported'.format(optimizer))

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
A sample CNN using representation matching during training
"""
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from openfl.models.pytorch import PyTorchFLModel
from openfl.models.pytorch import RepresentationMatchingWrapper

class ReshapeBatch(nn.Module):
    """ A simple layer that reshapes its input and outputs the reshaped tensor """
    def __init__(self, *args):
        super(ReshapeBatch, self).__init__()
        self.args = args

    def forward(self, x):
        return x.view(x.size(0), *self.args)

    def __repr__(self):
        s = '{}({})'
        return s.format(self.__class__.__name__, self.args)

        

class PyTorchCNNRepMatching(PyTorchFLModel):
    """
    Simple CNN for classification.
    """

    def __init__(self, data, device='cpu', num_classes=10,RM_loss_coeff = 0.0001, **kwargs):
        super().__init__(data=data, device=device)

        self.num_classes = num_classes
        self.init_network(device, **kwargs)
        self._init_optimizer()        

        self.RM_loss_coeff = RM_loss_coeff
        self.loss_fn = nn.CrossEntropyLoss()

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        #Keep a separate optimizer for the matching network becuase we do not want their optimizer state out
        self.matching_optimizer = optim.Adam(self.rep_matching_wrapper[0].parameters(), lr=1e-4)
    
    def init_network(self, 
                     device,
                     print_model=True, 
                     pool_sqrkernel_size=2,
                     conv_sqrkernel_size=5, 
                     conv1_channels_out=20, 
                     conv2_channels_out=50, 
                     fc2_insize = 500):

        pooling_contraction = pool_sqrkernel_size * 2
        fc1_insize = (self.feature_shape[-1] // pooling_contraction ) * (self.feature_shape[-1] // pooling_contraction) * conv2_channels_out

        self.stack = nn.Sequential(nn.Conv2d(1, conv1_channels_out, conv_sqrkernel_size, padding = conv_sqrkernel_size // 2 ),
                                   nn.MaxPool2d(pool_sqrkernel_size),
                                   nn.ReLU(),
                                   nn.Conv2d(conv1_channels_out, conv2_channels_out, conv_sqrkernel_size, padding = conv_sqrkernel_size // 2),
                                   nn.MaxPool2d(pool_sqrkernel_size),
                                   nn.ReLU(),
                                   ReshapeBatch(-1),
                                   nn.Linear(fc1_insize, fc2_insize),
                                   nn.ReLU(),
                                   nn.Linear(fc2_insize, self.num_classes)
                                   )

        #Keep the wrapper in a list so that it does not register as a submodule of the main model
        self.rep_matching_wrapper = [RepresentationMatchingWrapper(self.stack,self.feature_shape,matching_kernel_size = conv_sqrkernel_size)]
        
        if print_model:
            print(self)
        self.to(device)

        self.rep_matching_wrapper[0].to(device)
        self.rep_matching_wrapper[0].update_aggregator_model()
        
    def forward(self, x):
        return self.rep_matching_wrapper[0](x)

    def set_tensor_dict(self, *args,**kwargs):
        print('updating aggregator model')

        super(PyTorchCNNRepMatching,self).set_tensor_dict(*args,**kwargs)
        self.rep_matching_wrapper[0].update_aggregator_model()
        
    def validate(self, use_tqdm=False):
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

        #print('validation val :', repr(val_score / total_samples))
        return val_score / total_samples

    def train_batches(self, num_batches, use_tqdm=False): 
        # set to "training" mode
        self.train()
        
        losses = []

        loader = self.data.get_train_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="train epoch")

        batch_num = 0

        while batch_num < num_batches:
            # shuffling occers every time loader is used as an iterator
            for data, target in loader:
                if batch_num >= num_batches:
                    break
                else:
                    data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                    self.optimizer.zero_grad()
                    self.matching_optimizer.zero_grad()            
                    output = self(data)
                    model_loss = self.loss_fn(output, target.argmax(1))
                    matching_loss = self.rep_matching_wrapper[0].get_matching_loss()
                    (model_loss + self.RM_loss_coeff * matching_loss).backward() 
                    self.optimizer.step()
                    self.matching_optimizer.step()            
                    losses.append(model_loss.detach().cpu().numpy())
                    
                    batch_num += 1

        return np.mean(losses)

    def reset_opt_vars(self):
        self._init_optimizer()

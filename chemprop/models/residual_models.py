import torch
from torch import nn, optim, from_numpy, FloatTensor
from torch.autograd import Variable

import numpy as np


class ResidualModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualModel, self).__init__()

        self.first_l = nn.Linear(input_size, 10)
        self.last_l = nn.Linear(10, output_size)
        self.activation = nn.Tanh()

    def forward(self, x, p):
        x = from_numpy(x).float()
        p = FloatTensor(p)
        x = self.first_l(x)
        x = self.last_l(x)
        x = self.activation(x)
        # x = self.activation(self.last_l(self.first_l(x)))  # a number between -1 and 1 due to tanh
        p_prime = x
        p_prime = torch.clamp(p_prime, 0, 1)  # a number between 0 and 1

        return p_prime

def train_residual_model(train_inputs, old_preds, train_targets, epochs):
    model = ResidualModel(len(train_inputs[0]), len(train_targets[0]))
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for i in range(len(train_inputs)):
            model.zero_grad()
            preds = model(train_inputs[i:i+1], old_preds[i:i+1])
            loss = loss_fn(preds,
                           from_numpy((np.abs(np.array(train_targets[i:i+1]) - 
                                       np.array(old_preds[i:i+1])) < 0.5) * 1).float())
            loss.backward()
            optimizer.step()
        
    return model
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 07:09:01 2020

@author: Banani Fatima-Zahra
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class JANETCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(JANETCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        
    def forward(self, x, hidden):
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        forgetgate, cellgate = gates.chunk(2, 1)
        
        beta = 1
        cellgate = F.tanh(cellgate)  
        cy = (F.sigmoid(forgetgate) * cx) + (1 - F.sigmoid(forgetgate-beta)) * cellgate
        hy = cy

        return hy, cy
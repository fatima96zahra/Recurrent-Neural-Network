# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 07:10:52 2020

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

class CIFGCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CIFGCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
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
        
        ingate, gategate, outgate = gates.chunk(3, 1)
        
        ingate = F.sigmoid(ingate)
        gategate = F.tanh(gategate)
        outgate = F.sigmoid(outgate)
        

        cy = (1-ingate)*cx +  ingate*gategate        

        hy = outgate*F.tanh(cy)
        
        return (hy, cy)
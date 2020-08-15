# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:11:29 2020

@author: Banani Fatima-Zahra
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math

from cell_model import JANETCell, CIFGCell, NRUCell


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, cell_type ="gru", bias=True):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        #define the model's cell
        if cell_type == "gru":
            self.cell = nn.GRUCell(input_dim, hidden_dim)
        elif cell_type == "lstm":
            self.cell = nn.LSTMCell(input_dim, hidden_dim)
        elif cell_type == "janet":
            self.cell = JANETCell(input_dim, hidden_dim)
        elif cell_type == "nru":
            self.cell = NRUCell(device, input_dim, hidden_dim)
        else:   
            self.cell = CIFGCell(input_dim, hidden_dim)
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
         
       
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.cell(x[:,seq,:], hn) 
            outs.append(hn)
            

        out = outs[-1].squeeze()
        out = self.fc(out) 
        
        return out
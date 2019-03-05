#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch

class CNN(nn.Module):
    """ CNN
    """
    def __init__(self, char_embed_size, word_embed_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size, kernel_size=5)
        self.maxpool = nn.MaxPool1d(17)

    def forward(self, input):
        x_conv = self.conv(input)
        x_relu = nn.functional.relu(x_conv)
        x_convout = self.maxpool(x_relu)

        return torch.squeeze(x_convout, dim=2)


### END YOUR CODE

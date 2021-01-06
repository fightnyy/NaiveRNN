#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class Embedding :
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, =self.params #이렇게 쓰는 이유는 지금 params를 보면 [W]형태로 되어있는데 이러면 [[1,2,3]] 이런식으로 되어있어서 이걸 [1,2,3] 이렇게 해주는역할
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dw = self.grads
        dw[...] = 0
        np.add.at(dw, self.idx, dout)
        return None


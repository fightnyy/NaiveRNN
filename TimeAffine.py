#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TimeAffine:
    def __init__(self, W, b):
        self.params[W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self ,x):
        N, T, D = x.shape
        W, b = self.params

        rx = 

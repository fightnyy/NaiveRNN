#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Embedding
class TimeEmbedding :
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W


    def forward(self, xs):
        N, T = xs.shape # token 의 갯수 , 이런 token이 몇개 더 들어오는거 왜? time 이니까
        V, D = self.W.shape# 단어를 몇개의 dimension으로 할 것이냐가 여기서 나오는것 

        out = np.empty((N, T, D), dtype = 'f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out = [:,t,:] = layer.foward(xs[:,t])
            self.layers.append(layer)

        return out


    def backward(self, dout):
        N, T, D = dout.shape #N : 하나의 문장에서 얼마나 많은 token 이 있는지  , T : Time , D : 한개의 단어가 얼마의 사이즈로 표현되는지

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None

        

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import ptb
import numpy as np
import TRNN
import TimeEmbedding
import TimeAffine
train_data , word_to_id, id_to_word = ptb.load_data('train') #ptb dataset 받기

class SimpleRnnln :
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn


        embed_W = (rn(V,D)/100).astype('f')
        rnn_Wx = (rn(D, H)/ np.sqrt(D)).astype('f')
        rnn_Wh = (rn(D,H)/np.sqrt(D)).astype('f')
        rnn_b = np.zeros_like(H).astype('f')
        affine_W = (rn(H, V))/np.sqrt(H)).astype('f')
        affine_B = np.zeros_like(V).astype('f')

        self.layers = [
        TimeEmbedding(embed_W),
        TRNN(rnn_Wx, rnn_Wh, rnn_b),
        TimeAffine(affine_W, affine_b)
        ]
        
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer =self.layers[1]
        
        self.params. self.grads = [], []

        for layer in self.layers :
            self.params += layer.params
            self.grads += layer.grads


    def forward(self, xs, ts):
        for layer in self.layers :
            xs =layer.forward(xs)
        loss = self.loss_layer(xs,ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()

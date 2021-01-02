#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pdb
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layer = None #RNN 계층들을 list로 저장할 용도

        self.h , self.dh = None, None
        self.stateful = stateful #True = 은닉상태 유지
                                 #False = 은닉상태 유지 안함 즉, 0행렬 

    def set_state(self , h):
        self.h = h #마지막 RNN계층의 은닉상 태 저장

    def reset_state(self):
        self.h = None #

    def forward(self, xs):
        Wx , Wh, b = self.params
        N, T, D = xs.shape #N은 batch_size T는 Time 즉 몇개의 RNN 셀이 있는지\
                           #D는 token이 몇개의 벡터로 이루어져있는지
        D, H = Wx.shape #D는 한개의 token 이 몇개의 분산 원소로 이루어져있는지

        self.layers = []
        hs = np.empty((N, T, H), dtype = 'f')#hs는 hidden_state 즉
                                             #xs와 Wx의 곱으로 이루어진 
                                             #hidden 값
        if not self.stateful or self.h is None :
            self.h = np.zeros((N, H) , dtype = 'f')

        for t in range(T):
            layer = RNN(*self.params)#*으로 변수를 넣을경우 list의 원소값이
                                     #추출되어 메소드 인수로 전달
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs


    def backward(self, dhs) :
        Wx, Wh, b = self.params
        N, T, H =dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype = 'f')
        dh = 0
        grads = [0, 0, 0]# dWx, dWh, db 
        for t in reverse(range(T)):
            layer = self.layers(t)
            dx, dh =layer.backward(dhs[:,t,:]+dhs)
            dxs[:,t,:] = dx

            for i , grad in(layer.grads):
                pdb.set_trace()
                grads[i] += grad
                
        for i, grad in enumerate(grads):
            self.grads[i][...] = grads
        self.dh = dh
        return dxs











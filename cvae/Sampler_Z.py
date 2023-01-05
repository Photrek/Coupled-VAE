# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:42:17 2022

@author: jkcle
"""
from tensorflow import exp
from tensorflow.keras import layers as tfkl
from tensorflow.random import normal


class Sampler_Z(tfkl.Layer):   

    def call(self, inputs):
        mean, logvar = inputs
        eps = normal(shape=mean.shape)
        z_sample = eps * exp(logvar * .5) + mean

        return z_sample
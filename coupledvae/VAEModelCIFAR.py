# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:01:47 2022

@author: jkcle

Holds the base model for CIFAR.
"""
import tensorflow as tf

from .Sampler_Z import Sampler_Z
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl


class EncoderZ(tfkl.Layer):
    # n_filter_base is a bogus argument for now
    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(EncoderZ, self).__init__(name=name, **kwargs)

        self.conv_layers = tfk.Sequential(
          [
              tfkl.InputLayer(input_shape=(32, 32, 3)),
              tfkl.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                     activation='relu', dtype=dtype),
              tfkl.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                     activation='relu', dtype=dtype),
              tfkl.Conv2D(filters=128, kernel_size=3, 
                                     strides=(2, 2), activation='relu',
                                     dtype=dtype),
              tfkl.Flatten()
          ]
        )
        
        self.dense_mean = tfkl.Dense(
              z_dim, 
              activation=None, 
              name='z_mean', 
              dtype=dtype
              )
        
        self.dense_raw_stddev = tfkl.Dense(
            z_dim, 
            activation=None,
            name='z_raw_stddev', 
            dtype=dtype
            )
        
        self.sampler_z = Sampler_Z()
      
    def call(self, x_input):
      z = self.conv_layers(x_input)
      mean = self.dense_mean(z)
      logvar = self.dense_raw_stddev(z)
      z_sample = self.sampler_z((mean, logvar))
      return z_sample, mean, logvar


class DecoderX(tfkl.Layer):

    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(DecoderX, self).__init__(name=name, **kwargs)

        self.deconv_layers = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=(z_dim,)),
                tfkl.Dense(units=4*4*128, activation=tf.nn.relu,
                                      dtype=dtype),
                tfkl.Reshape(target_shape=(4, 4, 128)),
                tfkl.Conv2DTranspose(filters=128, kernel_size=3,
                                                strides=2, padding='same',
                                                activation='relu',
                                                dtype=dtype),
                tfkl.Conv2DTranspose(filters=64, kernel_size=3,
                                                strides=2, padding='same',
                                                activation='relu',
                                                dtype=dtype),
                tfkl.Conv2DTranspose(filters=32, kernel_size=3,
                                                strides=2, padding='same',
                                                activation='relu',
                                                dtype=dtype),
                tfkl.Conv2DTranspose(filters=3, kernel_size=3,
                                                strides=1, padding='same',
                                                dtype=dtype),
            ]
        )
    

    def call(self, z):
      return self.deconv_layers(z)


class VAEModelCIFAR(tfk.Model):
    '''Convolutional variational autoencoder base model.
    '''
    
    def __init__(self, z_dim, n_filter_base, seed, dtype):
        super(VAEModelCIFAR, self).__init__()
        self.encoder = EncoderZ(z_dim, n_filter_base, dtype)
        self.decoder = DecoderX(z_dim, n_filter_base, dtype)
        return
    
    def sample(self, z_sample):
        x_recons_logits = self.decoder(z_sample)
        sample_images = tf.sigmoid(x_recons_logits)
        return sample_images
    
    def call(self, x_input):
        z_sample, mean, logvar = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)
        return x_recons_logits, z_sample, mean, logvar

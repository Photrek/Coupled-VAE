# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:44:54 2022

@author: jkcle
"""
import tensorflow as tf

from Sampler_Z import Sampler_Z
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl


# Encoder/Decoder layers 1 (for MNIST images)
class EncoderZ(tfkl.Layer):

    def __init__(self, 
                 z_dim, 
                 n_filter_base, 
                 seed, 
                 dtype, 
                 name='encoder', 
                 **kwargs
                 ):
        super(EncoderZ, self).__init__(name=name, **kwargs)
        # Block-1
        self.conv_layer_1 = tfkl.Conv2D(
            filters=n_filter_base, 
            kernel_size=3,
            strides=1, 
            padding='same', 
            name='conv_1',
            dtype=dtype
            )

        self.batch_layer_1 = tfkl.BatchNormalization(
            name='bn_1', 
            dtype=dtype
            )
        self.activation_layer_1 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_1', 
            dtype=dtype
            )
        # Block-2
        self.conv_layer_2 = tfkl.Conv2D(
            filters=n_filter_base*2, 
            kernel_size=3,
            strides=2, 
            padding='same', 
            name='conv_2', 
            dtype=dtype
            )
        self.batch_layer_2 = tfkl.BatchNormalization(name='bn_2', dtype=dtype)
        self.activation_layer_2 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_2', 
            dtype=dtype
            )
        # Block-3
        self.conv_layer_3 = tfkl.Conv2D(
            filters=n_filter_base*2, 
            kernel_size=3,
            strides=2, 
            padding='same', 
            name='conv_3', 
            dtype=dtype
            )
        self.batch_layer_3 = tfkl.BatchNormalization(name='bn_3', dtype=dtype)
        self.activation_layer_3 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_3', 
            dtype=dtype
            )
        # Block-4
        self.conv_layer_4 = tfkl.Conv2D(
            filters=n_filter_base*2, 
            kernel_size=3,
            strides=1, 
            padding='same', 
            name='conv_4', 
            dtype=dtype
            )
        self.batch_layer_4 = tfkl.BatchNormalization(name='bn_4', dtype=dtype)
        self.activation_layer_4 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_4', 
            dtype=dtype
            )
        # Final Block
        self.flatten_layer = tfkl.Flatten()
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

    # Functional
    def call(self, x_input):
        z = self.conv_layer_1(x_input)
        z = self.batch_layer_1(z)
        z = self.activation_layer_1(z)
        z = self.conv_layer_2(z)
        z = self.batch_layer_2(z)
        z = self.activation_layer_2(z)
        z = self.conv_layer_3(z)
        z = self.batch_layer_3(z)
        z = self.activation_layer_3(z)
        z = self.conv_layer_4(z)
        z = self.batch_layer_4(z)
        z = self.activation_layer_4(z)
        z = self.flatten_layer(z)
        mean = self.dense_mean(z)
        logvar = self.dense_raw_stddev(z)
        z_sample = self.sampler_z((mean, logvar))
        return z_sample, mean, logvar


class DecoderX(tfkl.Layer):

    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(DecoderX, self).__init__(name=name, **kwargs)
        # For MNIST images
        self.dense_z_input = tfkl.Dense(
            units=7*7*n_filter_base*2,
            activation=tf.nn.relu, 
            dtype=dtype
            )
        self.reshape_layer = tfkl.Reshape(target_shape=(7, 7, n_filter_base*2))
        # Block-1
        self.conv_transpose_layer_1 = tfkl.Conv2DTranspose(
            filters=n_filter_base*2,
            kernel_size=3,
            strides=1, 
            padding='same',
            name='conv_transpose_1', 
            dtype=dtype
            )
        self.batch_layer_1 = tfkl.BatchNormalization(name='bn_1', dtype=dtype)
        self.activation_layer_1 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_1', 
            dtype=dtype
            )
        # Block-2
        self.conv_transpose_layer_2 = tfkl.Conv2DTranspose(
            filters=n_filter_base*2,
            kernel_size=3,
            strides=2, 
            padding='same',
            name='conv_transpose_2', 
            dtype=dtype
            )
        self.batch_layer_2 = tfkl.BatchNormalization(name='bn_2', dtype=dtype)
        self.activation_layer_2 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_2', 
            dtype=dtype
            )
        # Block-3
        self.conv_transpose_layer_3 = tfkl.Conv2DTranspose(
            filters=n_filter_base,
            kernel_size=3,
            strides=2, 
            padding='same',
            name='conv_transpose_3', 
            dtype=dtype
            )
        self.batch_layer_3 = tfkl.BatchNormalization(name='bn_3', dtype=dtype)
        self.activation_layer_3 = tfkl.Activation(
            tf.nn.leaky_relu, 
            name='lrelu_3', 
            dtype=dtype
            )
        # Block-4
        # Filters=1 for gray-scaled images
        self.conv_transpose_layer_4 = tfkl.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1, 
            padding='same',
            name='conv_transpose_4',
            dtype=dtype
            )

    # Functional
    def call(self, z):
        x_output = self.dense_z_input(z)
        x_output = self.reshape_layer(x_output)
        x_output = self.conv_transpose_layer_1(x_output)
        x_output = self.batch_layer_1(x_output)
        x_output = self.activation_layer_1(x_output)
        x_output = self.conv_transpose_layer_2(x_output)
        x_output = self.batch_layer_2(x_output)
        x_output = self.activation_layer_2(x_output)
        x_output = self.conv_transpose_layer_3(x_output)
        x_output = self.batch_layer_3(x_output)
        x_output = self.activation_layer_3(x_output)
        x_output = self.conv_transpose_layer_4(x_output)
        return x_output


class VAEModelMNIST(tfk.Model):
    '''Convolutional variational autoencoder base model.
    '''
    
    def __init__(self, z_dim, n_filter_base, seed, dtype):
        super(VAEModelMNIST, self).__init__()
        self.encoder = EncoderZ(z_dim, n_filter_base, seed, dtype)
        self.decoder = DecoderX(z_dim, n_filter_base, dtype)
        return
    
    def sample(self, z_sample):
        x_recons_logits = self.decoder(z_sample)
        sample_images = tf.sigmoid(x_recons_logits)  # predictions
        return sample_images
    
    def call(self, x_input):
        z_sample, mean, logvar = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)
        return x_recons_logits, z_sample, mean, logvar
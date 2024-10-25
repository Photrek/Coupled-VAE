# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:21:01 2022

@author: jkcle
"""
import math
import numpy as np
import nsc_tf
import pandas as pd
import random
import tensorflow as tf
import time
from tqdm import tqdm

from .GeneralizedMean import GeneralizedMean
from IPython import display
from tensorflow import keras as tfk
from .VAEModelCIFAR import VAEModelCIFAR as VAEModel
from .Visualize import Visualize


class VAE:
    """Variational Autoencoder wrapper."""
    def __init__(self, z_dim, n_filter_base=32, learning_rate=0.0005,
                 beta=1., p_std=1., seed=0, loss_coupling = 0.0,
                 analytic_kl=False, dtype='float64', display_path='.', 
                 checkpoint_path='.', input_type='mnist',
                 ):
        self.optimizer = tfk.optimizers.Adam(learning_rate)
        self.model = VAEModel(z_dim, n_filter_base, seed, dtype)
        metrics_col = ['epoch', 'train_neg_elbo', 'train_recon_loss', 'train_coupled_div'] + \
                      [f'{x}_{y}' for x in ['elbo', 'recon', 'coupled_div'] 
                                  for y in [
                                          'decisiveness', 
                                          'accuracy', 
                                          'robustness'
                                          ]
                      ]
        self.metrics_df = pd.DataFrame(columns=metrics_col)
        self.val_metrics_df = pd.DataFrame(columns=[
                                                    'val_neg_elbo', 
                                                    'val_recon_loss', 
                                                    'val_coupled_div'
                                                    ]
                                            )
        self.beta = beta
        self.p_std = p_std
        self.analytic_kl = analytic_kl
        self.display_path = display_path
        self._set_random_seeds(seed)
        self.loss_coupling = loss_coupling
        self.dtype=dtype
        self.z_dim = z_dim
        self.input_type = input_type
        
        print('Model initialized')
        return

    def train(
            self, 
            train_dataset, 
            val_dataset, 
            val_label,
            n_epoch=10, 
            n_epoch_display=10, 
            model_path = '.',
            show_display=False,
            early_stop=3
            ):
        epochs_since_last_improvement = 0
          
        print('Starting training')
        # declare an arbitrarly large loss for initialization
        best_val_score = float("inf") 
          
        # Pick a sample of the val set for generating output images
        for val_batch, val_batch_label in zip(
                val_dataset.take(1),
                val_label.take(1)
                ):
          val_sample = val_batch
          val_sample_label = val_batch_label
        
        
        for epoch in range(1, n_epoch + 1):
          if epochs_since_last_improvement < (early_stop + 1):
            # Training loop
            # Create empty lists to hold the training losses
            loss_lst, neg_ll_lst, kl_div_lst, ll_values_lst, kl_values_lst = [], [], [], [], []
            start_time = time.time()
            for train_x in tqdm(train_dataset):
        
              loss, neg_ll, kl_div, ll_values, kl_values = self.train_step(train_x)
              loss_lst.append(loss)
              neg_ll_lst.append(neg_ll)
              kl_div_lst.append(kl_div)
              ll_values_lst.append(ll_values)
              kl_values_lst.append(kl_values)
            end_time = time.time()
            # Concatenate all the loss metric components into their own tensors.
            loss = tf.reduce_mean(tf.stack(loss_lst, axis=0))
            neg_ll = tf.reduce_mean(tf.stack(neg_ll_lst, axis=0))
            kl_div = tf.reduce_mean(tf.stack(kl_div_lst, axis=0))
            ll_values = tf.concat(ll_values_lst, axis=0)
            kl_values = tf.concat(kl_values_lst, axis=0)
        
            # Get Validation Metrics
            # Only has one iteration, so not sure why loop is needed?
            val_loss_lst, val_neg_ll_lst, val_kl_div_lst, val_ll_values_lst, val_kl_values_lst = [], [], [], [], []
            for val_x in val_dataset:
              val_loss, val_neg_ll, val_kl_div, val_ll_values, val_kl_values = self.compute_loss(val_x, loss_coupling=self.loss_coupling)
              val_loss_lst.append(val_loss)
              val_neg_ll_lst.append(val_neg_ll)
              val_kl_div_lst.append(val_kl_div)
              val_ll_values_lst.append(val_ll_values)
              val_kl_values_lst.append(val_kl_values)
            
            val_loss = tf.reduce_mean(tf.concat(val_loss_lst, axis=0))
            val_neg_ll = tf.reduce_mean(tf.concat(val_neg_ll_lst, axis=0))
            val_kl_div = tf.reduce_mean(tf.concat(val_kl_div_lst, axis=0))
            val_ll_values = tf.concat(val_ll_values_lst, axis=0)
            val_kl_values = tf.concat(val_kl_values_lst, axis=0)
        
            display.clear_output(wait=False)
            print(
                f"Epoch: {epoch}, Train set Loss: {loss},\n " + \
                f"Train set Recon: {neg_ll}, Train set KL: {kl_div}, \n" + \
                f"Val set Loss: {val_loss},\n " + \
                f"Val set Recon: {val_neg_ll}, Val set KL: {val_kl_div}, \n" + \
                f"time elapse for current epoch: {end_time - start_time}"
                )
                
            if val_loss < best_val_score:
              best_val_score = val_loss
              print('Saving model checkpoint at epoch ' + str(epoch))
              print(model_path)
              self.model.save_weights(str(model_path / 'cp.ckpt'))
              epochs_since_last_improvement = 0
            else:
              epochs_since_last_improvement += 1
              
            # Generalized Mean
            gmean = GeneralizedMean(ll_values, kl_values, self.loss_coupling, self.z_dim)
        
            # Visualize / Display
            display_list = [n_epoch_display * x for x in range(1, n_epoch + 1)]
            if epoch in display_list:
              z_sample, _, _ = self.model.encoder(val_sample)
              viz = Visualize(self.z_dim,
                              self.loss_coupling,
                              self.input_type,
                              self.model.sample,
                              z_sample,
                              val_sample,
                              val_sample_label,
                              gmean.get_metrics(),
                              gmean.get_log_prob_values(),
                              self.display_path
                              )
              viz.display(show=show_display,
                          cd='X',
                          cl='X',
                          epoch=epoch
                          )
            metrics_row = [
                int(epoch), loss.numpy(), neg_ll.numpy(), kl_div.numpy()
                ]
            metrics_row = pd.Series(
                metrics_row, 
                index=self.metrics_df.columns[:4]
                )
            metrics_row = metrics_row.append(gmean.get_metrics())
            val_metrics_row = pd.Series(
                [
                val_loss.numpy(), val_neg_ll.numpy(), val_kl_div.numpy()
                ],
                index=self.val_metrics_df.columns
            )
            self.metrics_df = self.metrics_df.append(
                metrics_row, 
                ignore_index=True
                )
            self.val_metrics_df = self.val_metrics_df.append(
                val_metrics_row,
                ignore_index=True
                )
          else:
            print(f'{epochs_since_last_improvement} epochs since last improvement. Stopping Training.')
            break
        return


    def test(
            self, 
            test_corrupted,
            test_clean,
            test_label,
            save_path,
            loss_coupling=0.0,
            show_display=False
            ):
      
        print('Starting testing')
        
        # Pick a sample of the corrupted test set for generating output images
        for test_batch, test_batch_label in zip(
                test_corrupted.take(1),
                test_label.take(1)
                ):
          test_sample = test_batch
          test_sample_label = test_batch_label
          
        # Testing loop
        # Create empty lists to hold the training losses
        loss_lst, neg_ll_lst, kl_div_lst, ll_values_lst, kl_values_lst = [], [], [], [], []
        
        for test_x in tqdm(zip(test_corrupted, test_clean)):
            loss, neg_ll, kl_div, ll_values, kl_values = self.compute_loss_test(test_x[0], test_x[1], loss_only=False, loss_coupling=loss_coupling)
            loss_lst.append(loss)
            neg_ll_lst.append(neg_ll)
            kl_div_lst.append(kl_div)
            ll_values_lst.append(ll_values)
            kl_values_lst.append(kl_values)
        
        # Concatenate all the loss metric components into their own tensors.
        loss = tf.reduce_mean(tf.stack(loss_lst, axis=0))
        neg_ll = tf.reduce_mean(tf.stack(neg_ll_lst, axis=0))
        kl_div = tf.reduce_mean(tf.stack(kl_div_lst, axis=0))
        ll_values = tf.concat(ll_values_lst, axis=0)
        kl_values = tf.concat(kl_values_lst, axis=0)
        
        display.clear_output(wait=False)
        print(
            f"Test set Loss: {loss}, " + \
            f"Test set Recon: {neg_ll}, Test set KL: {kl_div}"
            )
        
        # Generalized Mean
        gmean = GeneralizedMean(ll_values, kl_values, self.loss_coupling, self.z_dim)
        
        z_sample, _, _ = self.model.encoder(test_sample)
        viz = Visualize(self.z_dim,
                        self.loss_coupling,
                        self.input_type,
                        self.model.sample,
                        z_sample,
                        test_sample,
                        test_sample_label,
                        gmean.get_metrics(),
                        gmean.get_log_prob_values(),
                        save_path
                        )
        viz.display_test(show=show_display,
                         cd='X',
                         cl='X',
                         epoch=''
                         )
        
        metrics_row = [
            loss.numpy(), neg_ll.numpy(), kl_div.numpy()
            ]
        metrics_row = pd.Series(
            metrics_row, 
            index=['neg_elbo', 'recon_loss', 'coupled_div']
            ) 
        
        metrics_row = metrics_row.append(gmean.get_metrics())
        
        return metrics_row


    @tf.function
    def train_step(self, x_true, return_loss_components=True):
        """Executes one training step and returns the loss.
        
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss, neg_ll, kl_div, ll_values, kl_values = self.compute_loss(
                x_true, loss_only=False, loss_coupling=self.loss_coupling
                )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
            )
        if return_loss_components:
            return loss, neg_ll, kl_div, ll_values, kl_values
        return

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    

    def _coupled_div(self, mean_vector, std_vector, coupling_loss):
        d = self.z_dim
        d1 = 1 + d*coupling_loss + 2*coupling_loss
        KL_d1 = tf.reduce_prod(
            tf.pow(2 * tf.constant(math.pi, dtype=tf.float64), 
                   coupling_loss/(1 + d*coupling_loss)) \
            * tf.sqrt(d1 / (d1 - 2*coupling_loss*tf.square(std_vector))) \
            * tf.exp(tf.square(mean_vector)*d1*coupling_loss 
                     / (1 + d*coupling_loss) 
                     / (d1 - 2*coupling_loss*tf.square(std_vector))), 
                     1)
        KL_d2 = tf.reduce_prod(
            tf.pow(2 * tf.constant(math.pi, dtype=tf.float64)*tf.square(std_vector),
                   coupling_loss / (1 + coupling_loss*d)) 
            * tf.sqrt(d1 / (1 + d*coupling_loss)), 
            1)
        
        KL_divergence = (KL_d1 - KL_d2) / coupling_loss / 2
        return KL_divergence

    def _kl_div(self, mean, logvar):
        # Analytical KL-Devergence that I want to put in the code.
        m = logvar.shape[1]
        
        # Calculate the determinant of the sample and reference covariance (which are 
        # vectors because the covariance matrices are diagonal).
        det_logvar = tf.reduce_sum(logvar, axis=1)
        det_logvar_ref = 0  # Determinant of the Identity matrix is 1 and log(1) = 0
        # Calculate the log ratio of the determinant of the sample covariance matrix
        # to the determinante of the reference covariance matrix.
        log_det_sigma_div_det_sigma_ref = det_logvar - det_logvar_ref
        # Calculate the trace of the inverse sample covariance matrix times the 
        # reference covariance matrix. For this special case where the new
        # covariance matrix is diagonal and the reference is the identity matrix,
        # This equals the sum of the diagonal of the inverse covariance matrix.
        trace_sigma_inv_sigma_ref = tf.reduce_sum(1 / tf.exp(logvar), axis=1)
        # Calculate the mean difference times the inverse covariance matrix
        # times the mean difference.
        mu_diff_sigma_inv_mu_diff = tf.reduce_sum(
            tf.multiply(tf.pow(mean, 2), tf.math.exp(-1*logvar)),
            axis=1)
        # Calculate the KL-Divergence
        kl_div = 0.5*(log_det_sigma_div_det_sigma_ref - m  + trace_sigma_inv_sigma_ref + mu_diff_sigma_inv_mu_diff)

        return kl_div


    def neg_elbo_loss(self, x_recons_logits, x_true, mean, logvar, beta=1.):
        
        # Sigmoid Cross Entropy Loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_recons_logits,
            labels=x_true
            )
        
        # Negative Log-Likelihood
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3]) # log-likelihood
        neg_ll = -logpx_z  # negative log-likelihood
        
        # KL-Divergence
        if self.analytic_kl is True:
            kl_div = self._kl_div(mean, logvar)
        # TODO Analytic is always used, so no z_sample is passed. Should remove
        else:
            logpz = self.log_normal_pdf(z_sample, 0., self.p_std)
            logqz_x = self.log_normal_pdf(z_sample, mean, logvar)
            kl_div = logqz_x - logpz
        
        # ELBO
        neg_ll_mean = tf.math.reduce_mean(neg_ll)
        kl_div_mean = tf.math.reduce_mean(kl_div)
        loss = neg_ll_mean + beta*kl_div_mean
        
        return_tuple = (loss, neg_ll_mean, kl_div_mean,
                        tf.cast(logpx_z, tf.float64),tf.cast(kl_div, tf.float64))
        
        return return_tuple


    def coupled_neg_elbo(self, x_recons_logits, x_true, mean, logvar, loss_coupling, beta=1.):
        ##NSC ELBO
        
        #Conversion from logits to probs
        p = x_true
        q = tf.math.sigmoid(x_recons_logits)
        
        #Calculation of binary log_loss
        cross_ent_2 = p*nsc_tf.math.function.coupled_logarithm(q, 
                                                              kappa=self.loss_coupling
            ) + (1-p)*nsc_tf.math.function.coupled_logarithm(
                    1-q, 
                    kappa=self.loss_coupling
                    )
        
        logpx_z= tf.reduce_sum(cross_ent_2, axis=[1, 2, 3])
        neg_ll = -logpx_z
        
        kl_div = self._coupled_div(mean, tf.exp(logvar/2), self.loss_coupling)
        
        neg_ll_mean = tf.math.reduce_mean(neg_ll)
        kl_div_mean = tf.math.reduce_mean(kl_div)
        
        loss = neg_ll_mean + beta*kl_div_mean
        
        return_tuple = (loss, neg_ll_mean, kl_div_mean, tf.cast(logpx_z, tf.float64),
                        tf.cast(kl_div, tf.float64))
        
        return return_tuple


    def compute_loss(self,
                    x_true, 
                    loss_only=False, 
                    loss_coupling=0.0
                    ):
        x_recons_logits, z_sample, mean, logvar = self.model(
            x_true
            )
        
        if loss_coupling == 0.0:
          loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = self.neg_elbo_loss(
              x_recons_logits, x_true, mean, logvar, beta=self.beta
              )
        else:
          ##NSC ELBO
          loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = self.coupled_neg_elbo(
              x_recons_logits, x_true, mean, logvar, loss_coupling, beta=self.beta
              )
        
        return_obj = (loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div)
        
        if loss_only:
            return_obj = loss
        
        return return_obj


    def compute_loss_test(self, x_corrupt, x_true, loss_only=False, loss_coupling=0):

        if loss_coupling == 0:
          x_recons_logits, z_sample, mean, logvar = self.model(x_corrupt)
          loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = self.neg_elbo_loss(
                x_recons_logits, x_true, mean, logvar, beta=1.
                )
        else:
          x_recons_logits, z_sample, mean, logvar = self.model(x_corrupt)
          loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = self.coupled_neg_elbo(
              x_recons_logits, x_true, mean, logvar, loss_coupling=loss_coupling, beta=1.
              )
        
        return_obj = (loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div)
        if loss_only:
            return_obj = loss
        
        return return_obj

    def _set_random_seeds(self, seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return


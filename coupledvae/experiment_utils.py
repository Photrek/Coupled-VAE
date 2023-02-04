# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:09:43 2022

@author: jkcle
Holds training and testing utils.
"""
import colorsys
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from matplotlib import colors as mc
from matplotlib import pyplot as plt
from tensorflow import cast, float64, reshape


def train_VAE(loss_coupling, 
              z_dim,
              n_filter_base,
              beta, 
              p_std, 
              analytic_kl, 
              n_epoch, 
              n_epoch_display, 
              datasets,
              dataset_type,
              random_seed, 
              model_path, 
              test_name,
              show_display,
              early_stop,
              cvae_type
              ):
  '''This function runs an experiment with the passed in parameters.

Inputs
  ------
  loss_coupling : float
    The coupling loss parameter.
  z_dim : int
    A positive integer for the dimensionality of the latent space.
  n_filter_base: int
    The number of filters to use in the CNN.
  beta : float
    The weight to place on the coupled divergence in the coupled ELBO.
  p_std : float
    The prior distribution's scale parameter.
  analytic_kl : bool
    Whether or not to use the analytical coupled divergence.
  n_epoch : int
    The number of epochs to train for.
  n_epoch_display : int
    The number of epochs to wait before displaying plots after an epoch.
  datasets : collections defaultdict
    A collection of datasets.
  dataset_type: str
    A string specifying the dataset type e.g. mnist or cifar10.
  random_seed : int
    A random seed.
  model_path : pathlib Path
    The root path of the output.
  test_name : str
    Name of the dataset.
  show_display : bool
    Whether or not to display plots while training.
  early_stop : int
    Max number of epochs to run since the last improvement.
  cvae_type : str
      Either MNIST or CIFAR for one of the two networks.

  Returns
  -------
  tuple: 
    display_name : str
      A name associated with the experiment.
    vae : VAE
      A trained VAE object.

  '''
  if cvae_type.lower() == 'cifar':
      print('Importing CIFAR CVAE')
      from .VAECIFAR import VAE
  else:
      print('Importing MNIST CVAE')
      from .VAEMNIST import VAE
  
  print(test_name)
  #Setting parameter string for files to be named    
  parameter_str = '_beta_' + str(beta) + '_zdim_' + str(z_dim) + \
                    '_p_std_' + str(p_std) + '_coupling_' + str(loss_coupling) +\
                    '_seed_'+str(random_seed)
  print('parameter_str:\t', parameter_str)

  # display_name = 'original' if test_name == 'mnist' else test_name.split('/')[-1]
  display_name = test_name.split('/')[-1]
  
  display_path = model_path / display_name
  my_vae = VAE(
      z_dim=z_dim,
      n_filter_base=n_filter_base,
      beta=beta,
      p_std=p_std,
      seed=random_seed,
      loss_coupling=cast(loss_coupling, float64),
      analytic_kl=analytic_kl,
      dtype='float64',
      display_path=display_path,
      input_type = dataset_type
      )
  my_vae.train(
      train_dataset=datasets[dataset_type]['train'],
      val_dataset=datasets[dataset_type]['val'],
      val_label=datasets[dataset_type]['val_label'],
      n_epoch=n_epoch,
      n_epoch_display=n_epoch_display,
      model_path=model_path,
      show_display=show_display,
      early_stop=early_stop
      )
  #Save to metrics tables.csv
  full_metrics_df = pd.concat([my_vae.metrics_df, my_vae.val_metrics_df], axis=1)
  full_metrics_df.to_csv(
      f"{display_path}/metrics/table_{parameter_str}.csv", 
      index=False
      )
  return display_name, my_vae

def train_VAEs(loss_coupling_vals, 
               z_dim_vals,
               n_filter_base,
               beta, 
               p_std, 
               analytic_kl, 
               n_epoch, 
               n_epoch_display, 
               datasets,
               dataset_type,
               datasets_names,
               random_seed, 
               model_path,
               show_display,
               early_stop,
               cvae_type
               ):
  '''This function runs experiments with the passed in parameters and 
  parameter lists.

  Inputs
  ------
  loss_coupling_vals : list
    A list of floats for the coupling loss parameter.
  z_dim_vals : list
    A list of positive integers for the dimensionality of the latent space.
  n_filter_base: int
    The number of filters to use in the CNN.
  beta : float
    The weight to place on the coupled divergence in the coupled ELBO.
  p_std : float
    The prior distribution's scale parameter.
  analytic_kl : bool
    Whether or not to use the analytical coupled divergence.
  n_epoch : int
    The number of epochs to train for.
  n_epoch_display : int
    The number of epochs to wait before displaying plots after an epoch.
  datasets : collections defaultdict
    A collection of datasets.
  dataset_type: str
    A string specifying the dataset type e.g. mnist or cifar10.
  datsets_names : list
    A list of the dataset names.
  random_seed : int
    A random seed.
  model_path : pathlib Path
    The root path of the output.
  show_display : bool
    Whether or not to display plots while training.
  early_stop : int
    Max number of epochs to run since the last improvement.
  cvae_type : str
      Either MNIST or CIFAR for one of the two networks.

  Returns
  -------
  vae_dict : dict
    A dict of the VAEs for each data set type.

  '''
  vae_dict = {}
  for loss_coupling in loss_coupling_vals:
    for z_dim in z_dim_vals:
      for test_name in datasets_names:

        display_name, trained_vae = train_VAE(
            loss_coupling=loss_coupling, 
            z_dim=z_dim,
            n_filter_base=n_filter_base,
            beta=beta, 
            p_std=p_std, 
            analytic_kl=analytic_kl, 
            n_epoch=n_epoch, 
            n_epoch_display=n_epoch_display, 
            datasets=datasets,
            dataset_type=dataset_type,
            random_seed=random_seed, 
            model_path=model_path, 
            test_name=test_name,
            show_display=show_display,
            early_stop=early_stop,
            cvae_type=cvae_type
            )

        vae_dict[display_name] = trained_vae
  return vae_dict


def test_VAE(datasets,
             my_vae,
             dataset_type,
             test_data,
             test_labels, 
             test_path, 
             test_name,
             show_display,
             random_seed,
             test_coupling=0
             ):
  '''This function runs an experiment with the passed in parameters.

  Inputs
  ------
  my_vae : VAE
    A trained VAE.
  dataset_type: str
    A string specifying the dataset type e.g. mnist or cifar10.
  test_path : pathlib Path
    The root path of the output.
  test_name : str
    Name of the dataset.
  show_display : bool
    Whether or not to display plots while training.

  Returns
  -------
  tuple: 
    display_name : str
      A name associated with the experiment.
    vae : VAE
      A trained VAE object.

  '''
  
  print(test_name)
  #Setting parameter string for files to be named    
  parameter_str = '_beta_' + str(my_vae.beta) + '_zdim_' + str(my_vae.z_dim) + \
                    '_p_std_' + str(my_vae.p_std) + '_coupling_' + str(round(my_vae.loss_coupling.numpy(), 10)) +\
                    '_seed'+str(random_seed)
  print('parameter_str:\t', parameter_str)

  # display_name = 'original' if test_name == 'mnist' else test_name.split('/')[-1]
  display_name = test_name.split('/')[-1]
  print(display_name)
  
  display_path = test_path / display_name
  print(display_path)
  #Save to metrics tables.csv
  full_metrics_df =   my_vae.test(
      test_corrupted=test_data,
      test_clean=datasets[f'{dataset_type}_corrupted/identity']['test'],
      test_label=test_labels,
      save_path=display_path,
      show_display=show_display,
      loss_coupling=test_coupling
  )
  full_metrics_df = pd.DataFrame(full_metrics_df).T
  print(f"Writing {display_path / 'metrics' / f'table_{parameter_str}.csv'}")
  full_metrics_df.to_csv(
      display_path / 'metrics' / f"table_{parameter_str}.csv", 
      index=False
      )
  # Calculate the KL probabilities.
  #kl_probs =tf.exp(-tf.concat(kl_values, axis=0))

  return #kl_probs


def test_VAE_loop(my_vae,
                  datasets,
                  dataset_type,
                  test_path, 
                  show_display,
                  random_seed,
                  test_coupling=1e-6
                  ):
  '''
  datasets : collections defaultdict
    A collection of datasets.
  dataset_type: str
    A string specifying the dataset type e.g. mnist or cifar10.
  '''
  for key in datasets.keys():
    test_data = datasets[key]['test']
    test_labels = datasets[key]['test_label']

    test_VAE(datasets,
             my_vae,
             dataset_type,
             test_data,
             test_labels, 
             test_path=test_path, 
             test_name=key,
             show_display=show_display,
             random_seed=random_seed,
             test_coupling=test_coupling
             )

  return


def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show();
    return

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_training(vae_dict, metric='neg_elbo'):
  for key, vae in vae_dict.items():
      metric_df = pd.concat([vae.metrics_df, vae.val_metrics_df], axis=1)
      x = metric_df['epoch']
      y = metric_df[f'train_{metric}']
      plt.plot(x, y, label=f'Training {metric}')
      y = metric_df[f'val_{metric}']
      plt.plot(x, y, label=f'Validation {metric}')
      plt.xticks(x)
      plt.legend()
      plt.xlabel('Epoch')
      plt.ylabel(f'{metric}')
      plt.title(f'Training and Validation {metric} vs. Epochs')
      plt.show()
  return
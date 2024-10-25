# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:02:13 2022

@author: jkcle

Holds set-up functions for the experiments.
"""
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from matplotlib import pyplot as plt
from numpy import squeeze
from pathlib import Path
from tensorflow_datasets import load


def update_experiments(experiment_params_dict, file_path):
  '''Update the CSV file that is tracking experimental parameters or make it
  if it does not exist.

  Inputs
  ------
  experiment_params_dict : dict
    A dictionary containing experimental parameters.
  file_path : str
    A path to the file tracking experimental parameters.

  Returns
  -------
  None

  '''

  # Covert the experiment_params_dict to a dataframe.
  experiment_params_df = pd.DataFrame(
      experiment_params_dict.items()
      ).set_index(0).T
  # If the file exists.
  if Path(file_path).exists():
    # Let the know the user know the file exists.
    print(f'Reading [{file_path}]')
    # Read in the file.
    old_experiments = pd.read_csv(file_path)
    # Let the user know the file is being updated.
    print(f'Updating [{file_path}]')
    # Add the new experimantal data to the old.
    experiment_params_df = pd.concat([old_experiments, experiment_params_df])
  # Let the user know the file is being written.
  print(f'Writing [{file_path}]')
  # Write the experimental params to the CSV.
  experiment_params_df.to_csv(file_path, index=False)

  return


def create_gdrive_output_folders(save_path, 
                                 img_folders=['identity', # TRAINING SET
                                              'motion_blur', 
                                              'rotate', 
                                              'translate'], 
                                 viz_folders=['generated_images', 
                                              'latent_spaces', 
                                              'manifolds', 
                                              'histograms',
                                              'metrics']
                                 ):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    train_img_path = save_path / 'train'
    train_img_path.mkdir(parents=True, exist_ok=True)

    for viz_folder in viz_folders:
      viz_path = train_img_path / f'{viz_folder}'
      viz_path.mkdir(parents=True, exist_ok=True)

    # Changed 8_26_2021: added explicit testing on each directory
    for img_folder in img_folders:
      img_path = save_path / 'test' / f'{img_folder}'
      img_path.mkdir(parents=True, exist_ok=True)
      for viz_folder in viz_folders:
        viz_path = img_path / f'{viz_folder}'
        viz_path.mkdir(parents=True, exist_ok=True)

    return


def check_gpu_availibility():
  '''This function checks if a GPU is present in a Google Colab notebook.
  '''

  # Import gpu_device_name from tensorflow.test.
  from tensorflow.test import gpu_device_name
  # If a GPU device is not detected, notify the user.
  if gpu_device_name() != '/device:GPU:0':
      print('WARNING: GPU device not found.')
  # If a GPU is detected, notify the user.
  else:
      print(f'SUCCESS: Found GPU: {gpu_device_name()}')
  return


def _preprocess(sample):
    '''Cast all the pixel values in a sample to 64-bit float and scale them
    between 0 and 1.
    '''
    image = tf.cast(sample['image'], tf.float64) / 255.
    return image


def _preprocess_label(sample):
    '''Cast the label of a sample to 64-bit integer.
    '''
    label = tf.cast(sample['label'], tf.int64)
    return label


def get_datasets(datasets_names, 
                 batch_size_train, 
                 batch_size_test, 
                 mnist_split,
                 random_seed
                 ):
    datasets = defaultdict(dict)
    for datasets_name in datasets_names:
        print('============================================================')
        print(f'\nExtracting {datasets_name.upper()} dataset...\n')
        # # CW - *** modified code such that when MNIST is loaded it gets 
        # # automatically split into train and validation sets
        if datasets_name == 'cifar10':
          (mtrain, mvalidation, mtest), datasets_raw_info = load(
              name=datasets_name,
              with_info=True,
              as_supervised=False,
              split=[
                  'train[:'+mnist_split+']', 
                  'train['+mnist_split+':]', 
                  'test']
              )
          
          train_size = int(mnist_split)
          datasets[datasets_name]['train'] = (
              mtrain.map(_preprocess)
              .batch(batch_size_train)
              .prefetch(tf.data.experimental.AUTOTUNE)
              .shuffle(train_size, seed=random_seed)
              )
          
          datasets[datasets_name]['val'] = (
              mvalidation
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['val_label'] = (
              mvalidation
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test'] = (
              mtest
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test_label'] = (
              mtest
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
        else:
          datasets_raw, datasets_raw_info = load(
              name=datasets_name,
              with_info=True,
              as_supervised=False
              )
          datasets[datasets_name]['test'] = (
              datasets_raw['test']
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test_label'] = (
              datasets_raw['test']
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
        print(datasets_raw_info)

        print('Here')
        if 'corrupted' in datasets_name:
          # View some examples from the dataset
          fig, axes = plt.subplots(3, 3, figsize=(8, 8))
          fig.subplots_adjust(hspace=0.2, wspace=0.1)
          for i, (elem, ax) in enumerate(
                  zip(datasets_raw['train'], axes.flat)
                  ):
              image = tf.squeeze(elem['image'])
              # print(image)
              label = elem['label']

              ax.imshow(image, cmap='gray')
              ax.text(0.7, -0.12, f'Digit = {label}', ha='right',
                      transform=ax.transAxes, color='black')
              ax.set_xticks([])
              ax.set_yticks([])
              # plt.show()
        else:
          # View some examples from the dataset
          fig, axes = plt.subplots(3, 3, figsize=(8, 8))
          fig.subplots_adjust(hspace=0.2, wspace=0.1)
          for i, (elem, ax) in enumerate(zip(mtrain, axes.flat)):
              image = tf.squeeze(elem['image'])
              # print(image)
              label = elem['label']

              ax.imshow(image, cmap='gray')
              ax.text(0.7, -0.12, f'Digit = {label}', ha='right',
                      transform=ax.transAxes, color='black')
              ax.set_xticks([])
              ax.set_yticks([])
              # plt.show()

        if 'corrupted' not in datasets_name:
            #
            print(' - Print one train set image:')
            for train_batch in datasets[datasets_name]['train'].take(1):
                image = train_batch[0].numpy()
            image = squeeze(image)
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.colorbar()
            plt.grid(False)
            plt.axis('off')
            plt.show();

        #
        print(' - Print one test set image:')
        for test_batch in datasets[datasets_name]['test'].take(1):
            image = test_batch[0].numpy()
        image = squeeze(image)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.grid(False)
        plt.axis('off')
        plt.show();

        #
        print(' - Print one test set label:')
        for test_label in datasets[datasets_name]['test_label'].take(1):
            label = test_label[0]
        print(label.numpy())
        print('\n')

    return datasets

def get_datasets_(datasets_names, 
                 batch_size_train, 
                 batch_size_test, 
                 training_split,
                 random_seed
                 ):
    datasets = defaultdict(dict)
    for datasets_name in datasets_names:
        print('============================================================')
        print(f'\nExtracting {datasets_name.upper()} dataset...\n')
        # # CW - *** modified code such that when MNIST is loaded it gets 
        # # automatically split into train and validation sets
        if 'corrupted' not in datasets_name:
          (mtrain, mvalidation, mtest), datasets_raw_info = load(
              name=datasets_name,
              with_info=True,
              as_supervised=False,
              split=[
                  'train[:'+training_split+']', 
                  'train['+training_split+':]', 
                  'test']
              )
          
          train_size = int(training_split)
          datasets[datasets_name]['train'] = (
              mtrain.map(_preprocess)
              .batch(batch_size_train)
              .prefetch(tf.data.experimental.AUTOTUNE)
              .shuffle(train_size, seed=random_seed)
              )
          
          datasets[datasets_name]['val'] = (
              mvalidation
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['val_label'] = (
              mvalidation
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test'] = (
              mvalidation
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test_label'] = (
              mvalidation
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          
          # Storing the identity dataset using the test set of the original data
          if 'corrupted' not in datasets_name:
            identity_datasets_name = f"{datasets_name}_corrupted/identity"

          datasets[identity_datasets_name]['test'] = (
              mtest
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[identity_datasets_name]['test_label'] = (
              mtest
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
        else:
          # No need to redownload the identity data
          if 'identity' in datasets_name:
            pass
          datasets_raw, datasets_raw_info = load(
              name=datasets_name,
              with_info=True,
              as_supervised=False
              )
          datasets[datasets_name]['test'] = (
              datasets_raw['test']
              .map(_preprocess)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
          datasets[datasets_name]['test_label'] = (
              datasets_raw['test']
              .map(_preprocess_label)
              .batch(batch_size_test)
              .prefetch(tf.data.experimental.AUTOTUNE)
              )
        print(datasets_raw_info)

        print('Here')
        if 'corrupted' in datasets_name:
          # View some examples from the dataset
          fig, axes = plt.subplots(3, 3, figsize=(8, 8))
          fig.subplots_adjust(hspace=0.2, wspace=0.1)
          for i, (elem, ax) in enumerate(
                  zip(datasets_raw['test'], axes.flat)
                  ):
              image = tf.squeeze(elem['image'])
              # print(image)
              label = elem['label']

              ax.imshow(image, cmap='gray')
              ax.text(0.7, -0.12, f'Digit = {label}', ha='right',
                      transform=ax.transAxes, color='black')
              ax.set_xticks([])
              ax.set_yticks([])
              # plt.show()
        else:
          # View some examples from the dataset
          fig, axes = plt.subplots(3, 3, figsize=(8, 8))
          fig.subplots_adjust(hspace=0.2, wspace=0.1)
          for i, (elem, ax) in enumerate(zip(mtrain, axes.flat)):
              image = tf.squeeze(elem['image'])
              # print(image)
              label = elem['label']

              ax.imshow(image, cmap='gray')
              ax.text(0.7, -0.12, f'Digit = {label}', ha='right',
                      transform=ax.transAxes, color='black')
              ax.set_xticks([])
              ax.set_yticks([])
              # plt.show()

        if 'corrupted' not in datasets_name:
            #
            print(' - Print one train set image:')
            for train_batch in datasets[datasets_name]['train'].take(1):
                image = train_batch[0].numpy()
            image = squeeze(image)
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.colorbar()
            plt.grid(False)
            plt.axis('off')
            plt.show();

        #
        print(' - Print one test set image:')
        print(datasets_name)
        for test_batch in datasets[datasets_name]['test'].take(1):
            image = test_batch[0].numpy()
        image = squeeze(image)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.grid(False)
        plt.axis('off')
        plt.show();

        #
        print(' - Print one test set label:')
        for test_label in datasets[datasets_name]['test_label'].take(1):
            label = test_label[0]
        print(label.numpy())
        print('\n')

    return datasets
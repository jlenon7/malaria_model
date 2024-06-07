import random
import numpy as np
from constants import MODEL_NAME
import tensorflow.keras as keras

from os import listdir
from os.path import exists
from typing import Optional

class Path:
  def plots(self, path: Optional[str]):
    return self.storage(f'plots/{path}')

  def board(self, path: Optional[str]):    
    return self.storage(f'board/{path}')

  def logs(self, path: Optional[str]):    
    return self.storage(f'logs/{path}')

  def storage(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'storage{path}'

  def resources(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'resources{path}'

  def clean_path(self, path: Optional[str]):
    if path is None:
      return ''

    if path.endswith('/') is True:
      path = path[:-1]

    if path.startswith('/') is True:
      return path 

    return f'/{path}'

path = Path()

def load_model():
  path = f'storage/{MODEL_NAME}'
  model_exists = exists(path)

  if (model_exists):
    return keras.models.load_model(path)

  model = keras.models.Sequential()

  model.add(
    keras.layers.Conv2D(
      filters=32,
      kernel_size=(3, 3),
      input_shape=(130, 130, 3),
      activation='relu'
    )
  )
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

  model.add(
    keras.layers.Conv2D(
      filters=32,
      kernel_size=(3, 3),
      input_shape=(130, 130, 3),
      activation='relu'
    )
  )
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

  model.add(
    keras.layers.Conv2D(
      filters=32,
      kernel_size=(3, 3),
      input_shape=(130, 130, 3),
      activation='relu'
    )
  )
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  return model

def get_data():
  test_path = path.resources('dataset/test')
  train_path = path.resources('dataset/train')

  image_gen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest'
  ) 

  return ( 
    image_gen.flow_from_directory(
      train_path,
      batch_size = 16,
      color_mode = 'rgb',
      class_mode = 'binary',
      target_size = (130, 130)
    ),
    image_gen.flow_from_directory(
      test_path,
      shuffle = False,
      batch_size = 16,
      color_mode = 'rgb',
      class_mode = 'binary',
      target_size = (130, 130)
    )
  )

def get_random_data():
  test_path = path.resources('dataset/test')

  classes = ['uninfected', 'parasitized']
  random_class = classes[random.randint(0, len(classes) - 1)]
  cells = listdir(f'{test_path}/{random_class}')
  random_cell = cells[random.randint(0, len(cells))]

  image = keras.preprocessing.image.load_img(f'{test_path}/{random_class}/{random_cell}', target_size = (130, 130))
  img_array = keras.preprocessing.image.img_to_array(image)

  return (image, np.expand_dims(img_array, axis = 0), random_class)

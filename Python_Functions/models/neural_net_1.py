# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:28:23 2023

@author: cfris
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.nn import relu
import kerastuner as kt
from kerastuner import HyperModel

tf.random.set_seed(21)


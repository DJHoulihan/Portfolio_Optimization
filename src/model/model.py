import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Model
import transformer as tm
import os
import sys

# Add project root to sys.path (two levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from config.config import get_config


class HouliPortfolio(Model):
    def __init__(self):
        super(HouliPortfolio, self).__init__()
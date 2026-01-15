import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Model
import transformer as tm
import os
import sys
import portfolio_generator as pg
import srem as sr
import caan as cn

# Add project root to sys.path (two levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from config.config import load_config, AppConfig

config= load_config()

class HouliPortfolio(Model):
    def __init__(self, config: AppConfig):
        super(HouliPortfolio, self).__init__()

        self.config = config
        self.caan = cn.CAAN(config.srem)
        self.srem = sr.SREM(config.srem)
        self.portfolio_generator = pg.Portfolio_Generator(config.gen)
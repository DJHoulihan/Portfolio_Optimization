import configparser
import os

class ConfigHandler:
    def __init__(self, config_file = 'config.ini'):
        # directory where this .py file lives
        project_root = os.path.dirname(os.path.abspath(__file__))

        self.config_file = os.path.join(project_root, config_file)
        self.config_parser = configparser.ConfigParser()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "lookback": self.lookback,
            "num_features": self.num_features,
            "embed_dim": self.embed_dim,
            })
        
        return self.config_parser
    

def get_config():
    return ConfigHandler().get_config()

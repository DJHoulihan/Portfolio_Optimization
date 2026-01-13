import configparser
import os

class ConfigHandler:
    def __init__(self, config_file = 'config.ini'):
        # directory where this .py file lives
        project_root = os.path.dirname(os.path.abspath(__file__))

        self.config_file = os.path.join(project_root, config_file)
        self.config_parser = configparser.ConfigParser()
    
    def get_config(self):
        return self.config_parser
    

def get_config():
    return ConfigHandler().get_config()

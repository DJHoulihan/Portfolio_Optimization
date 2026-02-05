import configparser
import os
from dataclasses import dataclass

class ConfigHandler:
    def __init__(self, config_file: str = "config.ini"):
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(project_root, config_file)
        self.parser = configparser.ConfigParser()

    def read(self):
        self.parser.read(self.config_file)
        return self.parser

@dataclass(frozen=True)
class SREMConfig:
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout_rate: float
    lookback: int
    num_features: int
    embed_dim: int

@dataclass(frozen=True)
class APIConfig:
    api_key: str

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    gamma: float
    sharpe_lambda: float
    lookback: int
    # reward_window: int
    # input_window: int


@dataclass(frozen=True)
class DataConfig:
    symbols: str
    data_path: str

@dataclass(frozen=True)
class GenConfig:
    fraction: float

@dataclass(frozen=True)
class EnvConfig:
    num_envs: int

@dataclass(frozen=True)
class AppConfig:
    srem: SREMConfig
    training: TrainingConfig
    api: APIConfig
    data: DataConfig
    gen: GenConfig
    env: EnvConfig

def load_config() -> AppConfig:
    
    parser = ConfigHandler().read()

    srem = SREMConfig(
        num_layers=parser.getint("SREM", "num_layers"),
        num_heads=parser.getint("SREM", "num_heads"),
        ff_dim=parser.getint("SREM", "ff_dim"),
        dropout_rate=parser.getfloat("SREM", "dropout_rate"),
        lookback=parser.getint("SREM", "lookback"),
        num_features=parser.getint("SREM", "num_features"),
        embed_dim=parser.getint("SREM", "embed_dim"),
    )

    training = TrainingConfig(
        batch_size=parser.getint("TRAINING", "batch_size"),
        epochs=parser.getint("TRAINING", "epochs"),
        learning_rate=parser.getfloat("TRAINING", "learning_rate"),
        gamma = parser.getfloat("TRAINING", "gamma"),
        sharpe_lambda = parser.getfloat("TRAINING", "sharpe_lambda"),
        lookback = parser.getint("TRAINING", "lookback")
        # reward_window = parser.getint("TRAINING", "reward_window"),
        # input_window = parser.getint("TRAINING", "input_window")
    )

    api = APIConfig(
        api_key = parser.get("API", "api_key")
    )

    data = DataConfig(
        symbols = parser.get("DATA",  "symbols"),
        data_path=parser.get("DATA", "data_path")
    )

    gen = GenConfig(
        fraction=parser.getfloat("GEN", "fraction")
    )

    env = EnvConfig(
        num_envs=parser.getint("ENV", "num_envs")
    )

    return AppConfig(
        srem=srem,
        training=training,
        api=api,        
        data=data,
        gen=gen,
        env=env
    )
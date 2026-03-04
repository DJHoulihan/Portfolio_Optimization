import configparser
import os
from dataclasses import dataclass

class ConfigHandler:
    def __init__(self, config_file: str = "config.ini"):
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(project_root, config_file)
        self.parser = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))

    def read(self):
        self.parser.read(self.config_file)
        return self.parser

@dataclass(frozen=True)
class SREMConfig:
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout_rate: float
    embed_dim: int

@dataclass(frozen=True)
class APIConfig:
    api_key: str

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    actor_lr: float
    critic_lr: float
    gamma: float
    sharpe_lambda: float
    lookback: int
    rollout_len: int
    critic_updates: int
    grad_clip: float
    # reward_window: int
    # input_window: int


@dataclass(frozen=True)
class DataConfig:
    symbols: str
    data_path: str
    max_assets: int
    min_obs: int

@dataclass(frozen=True)
class GenConfig:
    fraction: float

@dataclass(frozen=True)
class EnvConfig:
    num_envs: int
    train_frac: float
    val_frac: float
    test_frac: float

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
        # num_features=parser.getint("SREM", "num_features"),
        embed_dim=parser.getint("SREM", "embed_dim")
    )

    training = TrainingConfig(
        batch_size=parser.getint("TRAINING", "batch_size"),
        epochs=parser.getint("TRAINING", "epochs"),
        actor_lr=parser.getfloat("TRAINING", "actor_lr"),
        critic_lr=parser.getfloat("TRAINING", "critic_lr"),
        gamma = parser.getfloat("TRAINING", "gamma"),
        sharpe_lambda = parser.getfloat("TRAINING", "sharpe_lambda"),
        lookback = parser.getint("TRAINING", "lookback"),
        rollout_len = parser.getint("TRAINING","rollout_len"),
        critic_updates = parser.getint("TRAINING","critic_updates"),
        grad_clip = parser.getint("TRAINING","grad_clip") 
        # reward_window = parser.getint("TRAINING", "reward_window"),
        # input_window = parser.getint("TRAINING", "input_window")
    )

    api = APIConfig(
        api_key = parser.get("API", "api_key")
    )

    data = DataConfig(
        symbols = parser.get("DATA",  "symbols"),
        data_path=parser.get("DATA", "data_path"),
        max_assets=parser.getint("DATA", "max_assets"),
        min_obs=parser.getint("DATA", "min_obs")
    )

    gen = GenConfig(
        fraction=parser.getfloat("GEN", "fraction")
    )

    env = EnvConfig(
        num_envs=parser.getint("ENV", "num_envs"),
        train_frac=parser.getfloat("ENV", "train_frac"),
        val_frac=parser.getfloat("ENV", "val_frac"),
        test_frac=parser.getfloat("ENV", "test_frac")
    )

    return AppConfig(
        srem=srem,
        training=training,
        api=api,        
        data=data,
        gen=gen,
        env=env
    )
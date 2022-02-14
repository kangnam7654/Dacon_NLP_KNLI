import yaml
from box import Box
import pandas as pd


def load_config(cfg_path=None):
    with open(cfg_path, 'r') as f:
        config = Box(yaml.full_load(f))
    return config


def load_csv(path=None):
    csv = pd.read_csv(path, encoding='utf-8')
    return csv


if __name__ == '__main__':
    pass
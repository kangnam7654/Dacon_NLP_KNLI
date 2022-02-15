import yaml
from box import Box
import pandas as pd


def load_config(cfg_path=None):
    with open(cfg_path, 'r') as f:
        config = Box(yaml.full_load(f))
    return config


def load_csv(path=None):
    csv = pd.read_csv(path, encoding='utf-8', index_col=False)
    return csv


if __name__ == '__main__':
    a = load_csv('D:\project\dacon\Dacon_knlp_classification\data\\train_data.csv')
    line = a.iloc[0]
    text = line['premise'] + ' ' + '[SEP]' + ' ' + line['hypothesis']
    b = sorted(list(set(a['label'])))

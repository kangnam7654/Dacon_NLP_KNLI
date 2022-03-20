import pandas as pd

from utils.core.trainer import Trainer
from configs.setting import global_setting
from model.model_load import load_roberta
from dataload.dataloader import build_dataloader


def main():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')
    model, tokenizer = load_roberta(config=config, device=device)

    # dataframes
    test_df = pd.read_csv('./data/test_data.csv')

    # 데이터 로더
    test_loader = build_dataloader(cfg=config, df=test_df, tokenizer=tokenizer, device=device, mode='test')

    train = Trainer(config=config,
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    tokenizer=tokenizer)
    cls_label_map = test_loader.dataset.map_cls_to_label
    train.inference(cls_label_map=cls_label_map)


if __name__ == '__main__':
    main()

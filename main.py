import pandas as pd

from utils.core.trainer import Trainer
from configs.setting import global_setting
from model.model_load import load_roberta
from model.opt_load import opt_load
from model.loss_load import loss_load
from utils.logger.csv_logger import CSVLogger
from utils.callbacks.early_stopping import EarlyStopping
from utils.callbacks.save_checkpoint import SaveCheckPoint
from utils import train_utils
from dataload.dataloader import build_dataloader


def main():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')
    model, tokenizer = load_roberta(config=config, device=device)

    # dataframes
    data_path = config['DATA']['TRAIN_CSV']
    train_df, valid_df = train_utils.split_dataset(data_path, 'label', valid_size=config['TRAIN']['VALID_SPLIT_SIZE'])
    test_df = pd.read_csv(config['DATA']['TEST_CSV'])

    # 데이터 로더
    train_loader = build_dataloader(cfg=config, df=train_df, tokenizer=tokenizer, device=device, mode="train")
    valid_loader = build_dataloader(cfg=config, df=valid_df, tokenizer=tokenizer, device=device, mode="valid")
    test_loader = build_dataloader(cfg=config, df=test_df, tokenizer=tokenizer, device=device, mode='test')

    # loss, optimizer
    criterion = loss_load(config=config, device=device)
    optimizer = opt_load(config=config, model=model)

    # callbacks
    logger = CSVLogger(
        path=config['TRAIN']['LOGGING_SAVE_PATH'], sep=config['TRAIN']['LOGGING_SEP']
    )
    checkpoint = SaveCheckPoint(path=config['TRAIN']['MODEL_SAVE_PATH'])
    early_stopping = EarlyStopping(
        patience=config['TRAIN']['EARLYSTOP_PATIENT'], verbose=True
    )

    train = Trainer(config=config,
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    tokenizer=tokenizer,
                    logger=logger,
                    checkpoint=checkpoint,
                    early_stopping=early_stopping,
                    fp16=True)
    train.train()


if __name__ == '__main__':
    main()

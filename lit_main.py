import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from configs.setting import global_setting
from model.lit_model import LitRoberta
from utils import train_utils
from dataload.lit_dataloader import LitDataLoader

def main():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')
    model = LitRoberta(config)
    tokenizer = model.tokenizer

    # dataframes
    data_path = config['DATA']['TRAIN_CSV']
    train_df, valid_df = train_utils.split_dataset(data_path, 'label', valid_size=config['TRAIN']['VALID_SPLIT_SIZE'])
    test_df = pd.read_csv(config['DATA']['TEST_CSV'])

    # 데이터 로더
    lit_loaders = LitDataLoader(cfg=config,
                                train_df=train_df,
                                valid_df=valid_df,
                                test_df=test_df,
                                tokenizer=tokenizer)
    train_loader = lit_loaders.train_dataloader()
    valid_loader = lit_loaders.val_dataloader()
    # test_loader = lit_loaders.test_dataloader()

    csv_logger = pl_loggers.CSVLogger(save_dir='./logs/', name='')
    ckpt_callback = ModelCheckpoint(dirpath='./ckpt',
                                    filename=config['MODEL']['NAME'],
                                    monitor='valid_loss',
                                    save_top_k=1,
                                    save_weights_only=True,
                                    mode='min',
                                    save_last=False,
                                    verbose=True)
    early_stop = EarlyStopping(monitor='valid_loss', verbose=True, patience=10, mode='min')
    trainer = Trainer(max_epochs=config['TRAIN']['EPOCHS'],
                      gpus=2,
                      logger=csv_logger,
                      strategy='ddp',
                      callbacks=[ckpt_callback, early_stop],
                      precision=16)
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    main()
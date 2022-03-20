from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataload.dataset import CustomDataset


class LitDataLoader(LightningDataModule):
    def __init__(self, cfg=None, train_df=None, valid_df=None, test_df=None, tokenizer=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.device = device

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return CustomDataset(cfg=self.cfg, dataframe_set=self.train_df, tokenizer=self.tokenizer, train_stage=True, device=self.device)
        elif mode == 'valid':
            return CustomDataset(cfg=self.cfg, dataframe_set=self.valid_df, tokenizer=self.tokenizer, train_stage=True, device=self.device)
        elif mode == 'test':
            return CustomDataset(cfg=self.cfg, dataframe_set=self.test_df, tokenizer=self.tokenizer, train_stage=False, device=self.device)

    def train_dataloader(self):
        dataset = self.__create_dataset('train')
        return DataLoader(dataset, batch_size=self.cfg['TRAIN']['BATCH_SIZE'], shuffle=True, drop_last=False, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        dataset = self.__create_dataset('valid')
        return DataLoader(dataset, batch_size=self.cfg['VALID']['BATCH_SIZE'], shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    def predict_dataloader(self):
        raise Exception('predict_dataloader는 구현되어 있지 않습니다. test_dataloader를 이용해주세요.')

    def test_dataloader(self):
        dataset = self.__create_dataset('test')
        return DataLoader(dataset, batch_size=self.cfg['TEST']['BATCH_SIZE'], shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
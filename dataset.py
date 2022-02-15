import cv2

from utils.common.project_paths import GetPaths

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_dataloader(df, cfg, mode):
    """
    Augmentation을 불러오고 Dataset을 불러와 적용하여 Dataloader로 wrapping하는 함수입니다.

    :args
        df: 데이터 프레임
        cfg: config. 본 함수에서는 batch size를 받습니다.
        mode: 'train, valid, test'의 값들 중 하나를 받습니다.
    :return dataloader
    """
    mode = mode.lower()
    assert mode in ['train', 'valid', 'test'], 'mode의 입력값은 train, valid, test 중 하나여야 합니다.'
    if mode in ['train', 'valid']:
        param = True
    else:
        param = False
    trans = get_transforms(cfg)[mode]
    dataset = CustomDataset(df=df, transform=trans, train=param)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=param, drop_last=False)
    return dataloader


class CustomDataset(Dataset):
    """
    Dataset을 만드는 클래스입니다.
    param train: train, validation에는 True를, Inference에는 False를 주면 됩니다.
    """
    def __init__(self, df,  train=True):
        self.df = df
        self.train = train
        self.name_cls_map, cls_name_map = self.maps()
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        text = line['premise'] + ' ' + self.tokenizer.sep_token + ' ' + line['hypothesis']
        if self.train:
            label = self.name_cls_map[line['label']]
            return text, label
        else:
            return text

    def maps(self):
        name_cls_map = {}
        cls_name_map = {}
        labels = sorted(list(set(self.raw_data['label'])))
        for idx, i in enumerate(labels):
            name_cls_map[idx] = i
            cls_name_map[i] = idx
        return name_cls_map, cls_name_map

# class PetFinderDataModule(LightningDataModule):
#     """Data module of Petfinder profiles."""
#     def __init__(self, train_df=None, valid_df=None, test_df=None, cfg=None):
#         super().__init__()
#         self._train_df = train_df
#         self._valid_df = valid_df
#         self._test_df = test_df
#         self._cfg = cfg
#         self.trans = get_default_transforms()
#         self.ttas = get_tta_transforms()
#
#     def __create_dataset(self, mode='train'):
#         if mode == 'train':
#             return PetFinderDataset(self._train_df, train=True, transform=self.trans['train'])
#         elif mode == 'valid':
#             return PetFinderDataset(self._valid_df, train=True, transform=self.trans['valid'])
#         elif mode == 'test':
#             return PetFinderDataset(self._test_df, train=False, transform=self.trans['test'], tta=self.ttas)
#         elif mode == 'predict':
#             return PetFinderDataset(self._train_df, train=False, predict=True, transform=self.trans['test'], tta=self.ttas)
#
#     def train_dataloader(self):
#         dataset = self.__create_dataset('train')
#         return DataLoader(dataset, **self._cfg.train_loader)
#
#     def val_dataloader(self):
#         dataset = self.__create_dataset('valid')
#         return DataLoader(dataset, **self._cfg.valid_loader)
#
#     def predict_dataloader(self):
#         dataset = self.__create_dataset('predict')
#         return DataLoader(dataset, **self._cfg.test_loader)
#
#     def test_dataloader(self):
#         dataset = self.__create_dataset('test')
#         return DataLoader(dataset, **self._cfg.test_loader)





if __name__ == '__main__':
    a = AutoTokenizer.from_pretrained('klue/roberta-large')

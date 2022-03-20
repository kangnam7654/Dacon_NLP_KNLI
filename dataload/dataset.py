from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, cfg, dataframe_set, tokenizer, train_stage=True, device=None):
        self.cfg = cfg
        self.df = dataframe_set
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.device = device
        self.labels = None
        self.cls = None
        self._maps()
        self._preprocess()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        concat_sentence = self.df.loc[idx, 'concat_sentences']
        tokenized = self._tokenize_function(concat_sentence)
        input_values = self._input_values(tokenized)

        if self.train_stage:
            label = torch.zeros(self.cfg['MODEL']['N_CLASSES']).to(self.device)
            if self.device is not None:
                label = label.to(self.device)
            answer = self.df.loc[idx, 'class']
            label[answer] = 1
            return input_values, label

        else:
            return input_values

    def _tokenize_function(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=256)

    def _maps(self):
        self.map_label_to_cls = {}
        self.map_cls_to_label = {}
        for cls, label in enumerate(self.cfg['DATA']['PAIR']):
            self.map_label_to_cls[label] = cls
            self.map_cls_to_label[cls] = label

    def _preprocess(self):
        self.df['premise'] = self.df['premise'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')
        self.df['hypothesis'] = self.df['hypothesis'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '')

        self.df['concat_sentences'] = self.df["premise"] + " " + self.df["hypothesis"]
        if self.train_stage:
            self.df['class'] = self.df['label'].apply(lambda x: self.map_label_to_cls[x])
            self.label = self.df['label']
            self.cls = self.df['class']

    def _input_values(self, tokenized):
        input_values = torch.tensor(tokenized['input_ids'])
        if self.device is not None:
            return input_values.to(self.device)
        else:
            return input_values

import torch
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.loss_load import loss_load
from model.opt_load import opt_load


class LitRoberta(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg['MODEL']['NAME']
        self.n_classes = self.cfg['MODEL']['N_CLASSES']
        self.ckpt = self.cfg['MODEL']['CHECKPOINT']
        self.model = None
        self.tokenizer = None
        self.criterion = loss_load(config=self.cfg)
        self.__build_model()

    def forward(self, x):
        out = self.model(input_ids=x).logits
        return out

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__share_step(batch)
        results = {'loss': train_loss, 'acc': train_acc}
        return results

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_acc = self.__share_step(batch)
        results = {'loss': valid_loss, 'acc': valid_acc}
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out = torch.softmax(self(batch), dim=1)
        pred = torch.argmax(out, dim=1)
        return pred

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def configure_optimizers(self):
        opt = opt_load(self.cfg, self.model)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=2,
                                                         min_lr=1e-8,
                                                         verbose=True)
        monitor = 'valid_loss'
        returns = {'optimizer': opt, 'lr_scheduler': sch, 'monitor': monitor}
        return returns

    def __share_step(self, batch):
        data, label = batch
        out = torch.softmax(self(data), dim=1)
        loss = self.criterion(out, label).view(1)
        acc = self.compute_accuracy(out, label).view(1)
        return loss, acc

    def __share_epoch_end(self, outputs, mode):
        all_loss = []
        all_acc = []
        for out in outputs:
            loss, acc = out['loss'], out['acc']
            all_loss.append(loss)
            all_acc.append(acc)
        avg_loss = torch.mean(torch.cat(all_loss))
        avg_acc = torch.mean(torch.cat(all_acc))
        self.log_dict({f'{mode}_loss': avg_loss, f'{mode}_acc': avg_acc})

    def __build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_classes)
        if self.ckpt is not None:
            self.apply_ckpt()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def apply_ckpt(self):
        ckpt = torch.load(self.ckpt)
        self.model.load_state_dict(ckpt)
        print(f'모델을 성공적으로 불러왔습니다.')

    @staticmethod
    def compute_accuracy(out, labels):  # for classification
        max_indices = torch.argmax(out, dim=1)
        label = torch.argmax(labels, dim=1)
        acc = (max_indices == label).to(torch.float).mean() * 100
        return acc

import os
import numpy as np
import time
import torch


class SaveCheckPoint:
    """ 모델 학습 진행상황을 저장합니다. """
    def __init__(self, path=None, verbose=False, delta=0, mode='min'):
        """
        :param path: 모델을 저장할 경로입니다. None일 경우, YYMMDDHHMM.pth 로 저장이 됩니다.
        :param verbose: 모델을 저장할 경우, 해당 내용을 출력할지 정하는 변수입니다.
        :param delta: monitor할 score의 최소 improvement를 정하는 변수입니다. delta 이상의 improvement가 있어야 checkpoint가 저장이 됩니다.

        :param mode: improvement의 방향성을 결정합니다. 'min', 'max'의 값을 받으며,
                     min일 경우 score의 감소가 improvement, max일 경우 score의 증가가 improvement입니다.
        """
        self.verbose = verbose
        self.path = path
        self.delta = delta
        self.mode = mode.lower()
        self.best_score = np.Inf
        self.val_loss_min = np.Inf

    def __call__(self, score, model):
        score_ = abs(score)
        if self.mode == 'min':
            improve_condition = self.best_score - score_ > self.delta
        elif self.mode == 'max':
            improve_condition = score_ - self.best_score > self.delta
        else:
            raise Exception('mode의 변수는 min 또는 max여야 합니다.')

        if improve_condition:
            self.best_score = score_
            self.save_ckpt(model)

    def save_ckpt(self, model):
        """
        checkpoint를 저장하는 메서드입니다.
        :param model:
        :return:
        """
        t = time.strftime("%y%m%d%H%M")
        if self.path is None:
            self.path = f'./ckpt/model/{t}.pth'
        else:
            pass
        directory = os.path.split(self.path)[0]
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        if self.verbose:
            self.verbose_print()
        torch.save(model.state_dict(), self.path)  # state dict로 저장

    def verbose_print(self):
        print(f"best score가 {round(self.best_score, 3)}로 갱신되었습니다. 모델을 저장합니다...")
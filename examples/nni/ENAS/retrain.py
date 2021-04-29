import math
import numpy as np
import torch
from torch import nn as nn
from nni.nas.pytorch.utils import AverageMeter, to_device
from utils import accuracy_metrics, f1_score, accuracy

import logging

logger = logging.getLogger('ENAS')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='retrain_model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss

class Retrain:
    def __init__(self, model, optimizer, criterion, lr_scheduler, device, train_loader, test_loader, n_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def train(self,epoch):
        f1 = AverageMeter("f1")
        acc = AverageMeter("acc")
        losses = AverageMeter("losses")
        
        self.model.train()
        cur_step = epoch * len(self.train_loader)
        cur_lr = self.optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d LR %.6f", epoch, cur_lr)
        for step, (x, y) in enumerate(self.train_loader):
            bs = x.size(0)
            self.optimizer.zero_grad()
            logits = self.model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = criterion(aux_logits, y)
            else:
                aux_loss = 0.
            metrics = accuracy_metrics(logits, y)
            loss = self.criterion(logits, y)
            loss = loss + 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            losses.update(loss.item(), bs)
            acc.update(metrics["acc_score"], bs)
            f1.update(metrics["f1_score"], bs)
            cur_step += 1
        logger.info(
                    "Train: [{:3d}/{}] Loss {losses.avg:.3f} "
                    "acc {acc.avg:.2%}, f1 {f1.avg:.2%}".format(
                        epoch + 1, 100, losses=losses,
                        acc=acc, f1=f1))
        return losses.avg

    def validate(self, epoch):
        f1 = AverageMeter("f1")
        acc = AverageMeter("acc")
        losses = AverageMeter("losses")
        
        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.test_loader):
                bs = x.size(0)
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits, _ = logits
                metrics = accuracy_metrics(logits, y)
                loss = self.criterion(logits, y)
                losses.update(loss.item(), bs)
                acc.update(metrics["acc_score"], bs)
                f1.update(metrics["f1_score"], bs)

        logger.info("Valid: [{:3d}/{}] Loss {losses.avg:.3f} "
                        "acc {acc.avg:.2%}, f1 {f1.avg:.2%}".format(
                            epoch + 1, 100, losses=losses,
                            acc=acc, f1=f1))
        return f1.avg,losses.avg

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.best_top1 = 0.
        self.early_stopping = EarlyStopping(patience=15, verbose=True)
        for epoch in range(self.n_epochs):
            # training
            train_loss = self.train(epoch)
            # validation
            top1, val_loss = self.validate(epoch)
            self.best_top1 = max(self.best_top1, top1)
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            self.lr_scheduler.step()

        logger.info("Final best f1_score = {:.4%}".format(self.best_top1))
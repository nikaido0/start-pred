#!/usr/bin/env python
#coding=gbk

import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
import os
import shutil
import hiddenlayer as hl


# import tensorboard

class DataTrain:
    # ѵ��ģ��
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device

    def train_step(self, train_iter, test_iter, model_name, epochs=None, threshold=0.5):
        steps = 1
        train_fea = []
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', 'best.pth')
        early_stop = 10
        history1 = hl.History()
        canvas1 = hl.Canvas()
        print_step = 100
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            i = 0
            for train_data, seq_data, chr_data, train_label in train_iter: # ������������
                # print(train_data.shape)
                self.model.train()  # ����ѵ��ģʽ
                # ʹ������ģ����ͬһ�豸��
                train_data, seq_data, chr_data, train_label = train_data.to(self.device), seq_data.to(self.device), chr_data.to(self.device), train_label.to(self.device)
                # ģ��Ԥ��
                y_hat, train_feature = self.model(train_data, seq_data, chr_data)
                # ������ʧ
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # ���֮ǰ���ݶ�
                self.optimizer.zero_grad()
                # ���򴫲���ʧ
                loss.backward()
                # ���²���
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1
            # ���һ�ε���ѵ��
            end_time = time.time()
            epoch_time = end_time - start_time

            model_predictions, true_labels = predict(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            print(f'{model_name}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}')
            print(f'Train acc:{acc1}')

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break

        model = self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        # canvas1.save('./save_img/train_test' + model_name + '.pdf')

    def train_step_val(self, train_iter, val_iter, test_iter, modelname, epochs=None, model_num=0, early_stop=10000,
                       threshold=0.5):
        print("train with val")
        steps = 1
        bestlos_acc1 = 0.
        best_loss = 100000
        bestlos_epoch = 0
        PATH = os.getcwd()
        latest_model = os.path.join(PATH, 'saved_models', 'latest.pth')
        best_model = os.path.join(PATH, 'saved_models', 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        print_step = 100

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_data, seq_data, chr_data, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # ?????????

                train_data, seq_data, chr_data, train_label = train_data.to(self.device), seq_data.to(self.device), chr_data.to(self.device), train_label.to(self.device)

                y_hat = self.model(train_data, seq_data, chr_data)
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # ??????????
                self.optimizer.zero_grad()
                # ?????????
                loss.backward()
                # ???2???
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

                if steps % print_step == 0:
                    # ????????????????
                    model_predictions, true_labels = predict(self.model, val_iter, device=self.device)
                    y_hat = model_predictions
                    for i in range(len(model_predictions)):
                        if model_predictions[i] < threshold:  # threshold
                            y_hat[i] = 0
                        else:
                            y_hat[i] = 1

                    acc1 = accuracy_score(true_labels, y_hat)

                    test_model_predictions, test_true_labels = predict(self.model, test_iter, device=self.device)
                    test_y_hat = test_model_predictions
                    for i in range(len(test_model_predictions)):
                        if test_model_predictions[i] < threshold:  # threshold
                            test_y_hat[i] = 0
                        else:
                            test_y_hat[i] = 1

                    acc2 = accuracy_score(test_true_labels, test_y_hat)

                    # ???????epoch??step????????????
                    history1.log((epoch, steps),
                                 train_loss=loss.item(),  # ????????
                                 val_acc=acc1,
                                 test_acc=acc2)  # ?????????
                    # ?????????????????
                    with canvas1:
                        canvas1.draw_plot(history1["train_loss"])
                        canvas1.draw_plot(history1["val_acc"])
                        canvas1.draw_plot(history1["test_acc"])

            # self.model.eval()
            model_predictions, true_labels = predict(self.model, train_iter, device=self.device)

            y_hat = model_predictions
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    y_hat[i] = 0
                else:
                    y_hat[i] = 1

            acc1 = accuracy_score(true_labels, y_hat)
            # print(len(train_iter))
            train_loss = total_loss / len(train_iter)

            torch.save(self.model.state_dict(), latest_model)
            # latest
            if train_loss < best_loss:
                shutil.copy(latest_model, best_model)
                bestlos_epoch = epoch
                bestlos_acc1 = acc1
                best_loss = train_loss

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f'Model {model_num + 1}|Epoch:{epoch:002} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}')
            print(f'Train acc:{acc1}')

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break

        print("best_loss = {}".format(best_loss))
        print("best_loss_acc = {}".format(bestlos_acc1))
        self.model.load_state_dict(torch.load(best_model))

        canvas1.save('./save_img/train_val_' + str(model_num + 1) + modelname + '.pdf')


# model_predictions, true_labels
def predict(model, data, device="cuda"):
    # ģ��Ԥ��
    model.to(device)
    model.eval()  # ��������ģʽ
    predictions = []
    labels = []

    with torch.no_grad():  # ȡ���ݶȷ��򴫲�
        for x, f, f2, y in data:
            x = x.to(device)
            f = f.to(device)
            f2 = f2.to(device)
            y = y.to(device).unsqueeze(1)

            score, _ = model(x, f, f2)
            label = torch.sigmoid(score)  # ��ģ��Ԥ��ֵӳ����0-1֮��
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    # �˻�ѧϰ��
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def feature(model, data, device="cuda"):
    # ��ȡģ�ͱ���������
    model.eval()  # ��������ģʽ
    extract_fea = []

    with torch.no_grad():  # ȡ���ݶȷ��򴫲�
        for x, f, f2, y in data:
            x = x.to(device)
            f = f.to(device)
            f2 = f2.to(device)
            score, fea = model(x, f, f2)
            extract_fea.extend(fea.tolist())
    return torch.Tensor(extract_fea)

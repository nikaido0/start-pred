#!/usr/bin/env python
# coding=gbk


# ����ָ��
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
import pandas as pd


def scores(score_label, Y_True, th=0.5):
    # ��Ԥ����ʷ���תΪ��ǩ
    y_hat = [1 if score >= th else 0 for score in score_label]

    # ����ģ�ͣ�Y_True��ʵ��ǩ��y_hatԤ���ǩ��score_labelԤ�������
    tn, fp, fn, tp = confusion_matrix(Y_True, y_hat).ravel()
    Recall = recall_score(Y_True, y_hat)
    SPE = tn / (tn + fp)
    MCC = matthews_corrcoef(Y_True, y_hat)
    Precision = precision_score(Y_True, y_hat)
    wujianlv = fp / (fp + tn)
    loujianlv = fn / (tp + fn)
    F1 = f1_score(Y_True, y_hat)
    Acc = accuracy_score(Y_True, y_hat)
    AUC = roc_auc_score(Y_True, score_label)
    precision_aupr, recall_aupr, _ = precision_recall_curve(Y_True, score_label)
    AUPR = auc(recall_aupr, precision_aupr)

    return Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR
